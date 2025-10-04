import numpy as np
import torch
import time
from MMTO_examples import METALSTOExamples, getMETALSTOProblem
from materialEncoder import MaterialEncoder
from MMTO_obj_cons_sensitivities import (
    compute_mmto_objective_and_gradient,
    compute_mmto_constraint_and_gradient,
)
from PyTOImports import *



# The main code for METALS topology optimization
def run_topopt(
    to_problem,
    debug=False,
    nIterationsWithoutPenalization=50,
    nIterationsWithPenalization= 50,
    timeLimit=7200,
    saveNet=None,
    use_pretrained_vae=True,
    rel_conv_tol=1e-7,
    nDOFDesired=5000,
    gamma_init = 1e-3,
    gamma_max = 1000,
    gamma_factor = 2,
    apply_filter_to_materials=True):
    
    history = {
        "objective": [],
        "constraints": []
    }
    # --- Get the TO problem
    mesh_structural, mat_prop_struct, bc_struct, elem_body_force, to_params, vae_params = getMETALSTOProblem(to_problem, nDOFDesired=nDOFDesired)
    
    # --- Read the materials excel file ---
    if to_params.MaterialsExcelFile  is None:
        print("Please provide a valid MaterialsExcelFile in to_params.")
        return
    matEncoder = MaterialEncoder(vae_params)
    matEncoder.readExcel(to_params.MaterialsExcelFile)

    numAttributes = matEncoder.nAttributes

    # Train the autoencoder or save/load from file
    if saveNet is None:
        base, _ = os.path.splitext(to_params.MaterialsExcelFile)
        saveNet = base + ".nt"
    vae_file_exists = os.path.exists(saveNet) and os.path.getsize(saveNet) > 0

    if use_pretrained_vae and vae_file_exists:
        print(f"Loading pre-trained autoencoder from file: {saveNet}")
        matEncoder.loadAutoencoderFromFile(saveNet)
        with torch.no_grad():
            z_real_np = matEncoder.vaeNet.encoder(matEncoder.scaledMaterialData).cpu().numpy()
    else:
        print(f"Training autoencoder from scratch and saving to: {saveNet}")
        matEncoder.trainAutoencoder(vae_params.numEpochs, vae_params.klFactor, saveNet, vae_params.learningRate)
        with torch.no_grad():
            z_real_np = matEncoder.vaeNet.encoder(matEncoder.scaledMaterialData).cpu().numpy()
        matEncoder.printEncodingErrors()
        for attributeId in range(numAttributes):# Optionally plot the latent space
            matEncoder.plotLSRContours(attributeId=attributeId)
        
    with torch.no_grad():
        matEncoder.training_latents = matEncoder.vaeNet.encoder(matEncoder.scaledMaterialData).cpu()

    zRealTorch = matEncoder.vaeNet.encoder.z
    
    # Set up the FEA solver
    solver = linear_solvers.Solvers.PARDISO
    dsolver = deflation.DeflationSolver()
    fe_solver_structural = hex_structural_fea.HexStructuralFEA(
        mesh=mesh_structural,
        mat_prop=mat_prop_struct,
        bc=bc_struct,
        solver=solver,
        dsolver=dsolver,
        rtol=1e-8,
        elem_body_force=elem_body_force)
    
    KETemplate = hex_element_stiffness.hex8_stiffness_matrix_structural(
            mat_prop_struct.youngs_modulus, mat_prop_struct.poissons_ratio, mesh_structural.elem_size)
    
    num_elems = mesh_structural.num_elems
    num_design_var = num_elems + num_elems * 2

    # Create the filter for density and material variables
    print("Creating filter...")
    [H, Hs] = createFilters(fe_solver_structural, to_params)
    shared_vars = {}

    iterationCount = 0
    obj0 = None # will get updated in the first iteration
    gamma = gamma_init
    def METALS_optimization_function(zeta):
        nonlocal iterationCount, obj0, gamma, zRealTorch
        zeta = np.asarray(zeta).flatten()
        print("-------------- Iteration", iterationCount, "-----------------")
        
        # Prepare tensors and decode material properties
        zetaTensor = torch.tensor(zeta, dtype=torch.float32, requires_grad=True)
        xDesign = zetaTensor[0:num_elems]
        zD = zetaTensor[num_elems:]
        zDesign = zD.view(2, -1).T

        decoded = matEncoder.vaeNet.decoder(zDesign)
        material_properties = matEncoder.getMaterialProperties(decoded)
        zValues = zDesign.detach().numpy()

        #fe_solver_structural.plot_elem_field(zValues[:,0], title='z', colormap='viridis')

        # Set material properties for FEA solver
        Youngs_Modulus = material_properties['Youngs_Modulus'].detach().numpy()

        fe_solver_structural.mat_prop = [
            mat_lib.create_material_with_defaults(name=f"Material_{i+1}", youngs_modulus=Youngs_Modulus[i])
            for i in range(len(Youngs_Modulus))]
        fe_solver_structural.set_structural_material(fe_solver_structural.mat_prop)

        # Solve FEA and compute objective/constraints/gradients
        sol = fe_solver_structural.solve(xDesign.detach().cpu().numpy(), MaterialModel.SIMP)
        fe_solver_structural.postprocess() # to compute stresses etc.

        obj, grad_obj = compute_mmto_objective_and_gradient(
            to_params, sol, zeta, fe_solver_structural, KETemplate, matEncoder)
        cons, grad_cons = compute_mmto_constraint_and_gradient(
            to_params, sol, zeta, fe_solver_structural, KETemplate, matEncoder)

        if (obj0 is None):
            obj0 = obj
        
        obj = obj / obj0  # Normalize objective
        grad_obj = grad_obj / obj0

        # Apply filter to sensitivities
        grad_obj[0:num_elems] = (H * grad_obj[0:num_elems]) / Hs
        if apply_filter_to_materials:
            grad_obj[num_elems:2*num_elems] = (H * grad_obj[num_elems:2*num_elems]) / Hs
            grad_obj[2*num_elems:3*num_elems] = (H * grad_obj[2*num_elems:3*num_elems]) / Hs

        for i in range(grad_cons.shape[0]):  # For each constraint
            grad_cons[i, 0:num_elems] = (H * grad_cons[i, 0:num_elems]) / Hs
            if apply_filter_to_materials:
                grad_cons[i, num_elems:2*num_elems] = (H * grad_cons[i, num_elems:2*num_elems]) / Hs
                grad_cons[i, 2*num_elems:3*num_elems] = (H * grad_cons[i, 2*num_elems:3*num_elems]) / Hs

        grad_obj = np.array([grad_obj]).reshape((num_design_var, 1))
        cons = np.array(cons).reshape((-1, 1))
        grad_cons = np.array(grad_cons).reshape((len(cons), num_design_var))

     
        shared_vars['zDesign'] = zDesign.detach().cpu().numpy()
        shared_vars['Youngs_Modulus'] = material_properties.get('Youngs_Modulus', None).detach().numpy()

        # Extract names for printing
        objective_name = getattr(to_params.Objective[0], 'name', str(to_params.Objective[0]))
        constraint_names = [getattr(c[0], 'name', str(c[0])) for c in to_params.Constraints]

        # Print objective and constraints for this iteration
        print(f"Objective ({objective_name}): {obj*obj0:.6f}")
        for idx, val in enumerate(cons.flatten()):
            print(f"Constraint {idx+1} ({constraint_names[idx]}): {val:.6f}")

        # Store history
        history["objective"].append(obj)
        history["constraints"].append(cons.flatten().copy())

        # Add penalty to objective to keep designs close to training data
        if (iterationCount >= nIterationsWithoutPenalization):
            p_softmin = -6
            d_ij = torch.cdist(zDesign, zRealTorch, p=2) + 1e-12
            min_i = torch.sum(d_ij ** p_softmin, dim=1).pow(1.0/p_softmin)
            min_i = min_i * xDesign
            penalty = gamma * torch.sum(min_i) / num_elems
            zetaTensor.grad = None
            penalty.backward(retain_graph=True)
            grad_obj[num_elems:,0] += zetaTensor.grad[num_elems:].detach().numpy()
            obj = obj + penalty.item()

            # # Apply filter to grad_obj for the penalty term
            if False and apply_filter_to_materials: # Don't use for now
                grad_obj[num_elems:2*num_elems, 0] = (H * grad_obj[num_elems:2*num_elems, 0]) / Hs
                grad_obj[2*num_elems:3*num_elems, 0] = (H * grad_obj[2*num_elems:3*num_elems, 0]) / Hs
            gamma = min(gamma*gamma_factor, gamma_max)


        iterationCount += 1
        return obj, grad_obj, cons, grad_cons


    # Initialize the design variables
    nConstraints = len(to_params.Constraints)
    x0 = 0.5 * np.ones(num_elems) 
    x0 = (H * x0) / Hs

    #z0 = np.random.uniform(-2,2, size=(2 * num_elems,))  
    z0 = np.max(zRealTorch.cpu().numpy()) * np.ones(2 * num_elems)

    if apply_filter_to_materials:
        z0[0:num_elems] = (H * z0[0:num_elems])/Hs
        z0[num_elems:2*num_elems] = (H * z0[num_elems:2*num_elems]) / Hs

    zeta0 = np.concatenate((x0, z0), axis=0).reshape(-1, 1)  # shape: (3*num_elems, 1)
    lowerBound = np.zeros(num_design_var, dtype=float).reshape(-1, 1)
    upperBound = np.ones(num_design_var, dtype=float).reshape(-1, 1)
    # Set bounds for material latent variables
   
    lowerBound[num_elems:3*num_elems] = np.min(zRealTorch.cpu().numpy())
    upperBound[num_elems:3*num_elems] = np.max(zRealTorch.cpu().numpy())

    nVariables = num_design_var
    tStart = time.time()
    maxMMAIterations = nIterationsWithoutPenalization + nIterationsWithPenalization


    # Run the MMA optimization
    optResults = runMMA(nVariables, nConstraints, METALS_optimization_function, zeta0.reshape(-1, 1), lowerBound,
        upperBound, maxIterations=maxMMAIterations, timeLimitSecs=timeLimit,
        move_limit=0.2, kktTol=1e-6, fTolerance=rel_conv_tol, gTolerance=rel_conv_tol, verbose=False)
    zetaOptimal = optResults[0]
    tEnd = time.time()
    print(f"Total optimization time: {tEnd - tStart:.2f} seconds")
   
    # post process the results
    if 'history' in shared_vars and 'mass' in shared_vars['history'] and len(shared_vars['history']['mass']) > 0:
        shared_vars['final_mass'] = shared_vars['history']['mass'][-1]
    else:
        shared_vars['final_mass'] = shared_vars.get('current_mass', None)
    zetaOptimal = np.asarray(zetaOptimal).flatten()
    xDesign = zetaOptimal[0:num_elems]
    
    zDesign = shared_vars['zDesign']
    Youngs_Modulus = shared_vars['Youngs_Modulus']
 
    fe_solver_structural.mesh.setPseudoDensity(xDesign)
    fe_solver_structural.solve()
    fe_solver_structural.postprocess()
    fe_solver_structural.plot_elem_field(Youngs_Modulus, title='YoungModulus', colormap='viridis')
    fe_solver_structural.plot_vonMisesStress()
    
    with torch.no_grad():
        z_real_np = matEncoder.vaeNet.encoder(matEncoder.scaledMaterialData).cpu().numpy()
    z_opt = zDesign if isinstance(zDesign, np.ndarray) else zDesign.detach().cpu().numpy()
    matEncoder.plotLSR(z_real_np, zDesign=z_opt.reshape(-1, 2), xDesign=xDesign)

if __name__ == "__main__":
    
    to_problem = METALSTOExamples.LBracketMidLoadStressSafetyFactor

    nDOFDesired = 10000
    run_topopt(
        to_problem=to_problem,
        nIterationsWithoutPenalization = 150,
        nIterationsWithPenalization = 0,
        use_pretrained_vae=True,
        nDOFDesired=nDOFDesired,
        apply_filter_to_materials=True
    )