import numpy as np
import torch
import time
from MMTO_examples import MMTOExamples, getMMTOProblem
from materialEncoder import MaterialEncoder
import matplotlib.pyplot as plt
from MMTO_TempDependent_obj_cons_sensitivities import (
    compute_mmto_objective_and_gradient,
    compute_mmto_constraint_and_gradient,
)
from PyTOImports import *
from HermiteFunction import hermiteInterpolation
from enum import Enum
class Z0InitMethod(Enum):
    LIGHTEST = 'lightest'
    HEAVIEST = 'heaviest'
    ORIGIN = 'origin'
    UNIFORM = 'uniform'


# The main code for MMTO topology optimization
def run_topopt(
    to_problem,
    nIterationsWithoutPenalization=50,
    nIterationsWithPenalization= 50,
    turnOnThermal=True,
    timeLimit=7200,
    saveNet=None,
    use_pretrained_vae=True,
    z0_init_method = Z0InitMethod.UNIFORM,  # options: Z0InitMethod.LIGHTEST, etc.
    rel_conv_tol=1e-10,
    gamma_init = 1e-4,
    gamma_max = 1000,
    gamma_factor = 2):
    
   
    # --- Get the TO problem ---
    mesh_structural, mesh_thermal, mat_prop_struct, mat_prop_thermal, \
    bc_struct, bc_thermal, elem_body_force, to_params, vae_params = getMMTOProblem(to_problem)

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
    else:
        print(f"Training autoencoder from scratch and saving to: {saveNet}")
        matEncoder.trainAutoencoder(vae_params.numEpochs, vae_params.klFactor, saveNet, vae_params.learningRate)
        matEncoder.printEncodingErrors()


    with torch.no_grad():
        matEncoder.training_latents = matEncoder.vaeNet.encoder(matEncoder.scaledMaterialData).cpu()

    zRealTorch = matEncoder.training_latents


    if (False): # optionally plot the latent space
        matEncoder.plotTemperatureVsMaterialProperty("E")
        matEncoder.plotTemperatureVsMaterialProperty("Y")
        for attributeId in range(numAttributes):# Optionally plot the latent space
            matEncoder.plotLSRContours(attributeId=attributeId)
        

    # Set up the FEA solver
    solver = linear_solvers.Solvers.PARDISO
    dsolver = deflation.DeflationSolver()
    fe_solver_structural = hex_structural_fea.HexStructuralFEA(
        mesh=mesh_structural,
        mat_prop=mat_prop_struct,
        bc=bc_struct,
        solver=solver,
        dsolver=dsolver,
        rtol=1e-4,
        elem_body_force=elem_body_force)
    

    fe_solver_thermal = hex_thermal_fea.HexThermalFEA(
            mesh=mesh_thermal,
            mat_prop=mat_prop_thermal,
            bc=bc_thermal,
            solver=solver,
            rtol=1e-8)
    

    KETemplate = hex_element_stiffness.hex8_stiffness_matrix_structural(
            mat_prop_struct.youngs_modulus, mat_prop_struct.poissons_ratio, mesh_structural.elem_size)
    
    num_elems = mesh_structural.num_elems
    num_design_var = num_elems + num_elems * 2

    # Create the filter for density and material variables
    print("Creating filter...")
    [H, Hs] = createFilters(fe_solver_structural, to_params)
    

    iterationCount = 0
    obj0 = None # will get updated in the first iteration
    gamma = gamma_init
    def MMTO_optimization_function(zeta):
        nonlocal iterationCount, obj0, gamma, zRealTorch
        zeta = np.asarray(zeta).flatten()
        print("-------------- Iteration", iterationCount, "-----------------")
        
        # Prepare tensors and decode material properties
        zetaTensor = torch.tensor(zeta, dtype=torch.float32, requires_grad=True)
        xDesign = zetaTensor[0:num_elems]
        zD = zetaTensor[num_elems:]
        zPts = zD.view(2, -1).T

        decoded = matEncoder.vaeNet.decoder(zPts)
        material_properties = matEncoder.getMaterialProperties(decoded)
       
        if (turnOnThermal):
            # Set material properties for FEA solver
            thermalConductivity = material_properties['K']
            thermalConductivity_elem = thermalConductivity.detach().numpy()
            #print(f"Thermal Conductivity (min, max): {np.min(thermalConductivity_elem):.3g}, {np.max(thermalConductivity_elem):.3g}")
            fe_solver_thermal.mat_prop = [
                mat_lib.create_material_with_defaults(
                    name=f"K{i+1}",
                    thermal_conductivity=thermalConductivity_elem[i].item())
                for i in range(num_elems)]

            #print("Solving thermal FEA...")
            fe_solver_thermal.set_thermal_material(fe_solver_thermal.mat_prop)
          
            T_full = fe_solver_thermal.solve(xDesign.detach().cpu().numpy())

            #fe_solver_thermal.plot_temperature()
            edofMat = fe_solver_thermal.mesh.edofMat
            T = np.mean(T_full[edofMat], axis=1)
        else:
            T = np.ones(num_elems) * 20.0  # uniform temperature of 20 degrees C
        #print(f"Temperature (min, max): {np.min(T):.3g}, {np.max(T):.3g}")


        # --- STRUCTURAL ANALYSIS WITH TEMPERATURE-DEPENDENT MATERIALS ---
        E  = matEncoder.getMaterialPropertyAtTemperature( "E",  zPts, T)
        Y = matEncoder.getMaterialPropertyAtTemperature( "Y",  zPts, T)

        #print(f"E (min, max): {np.min(E):.3g}, {np.max(E):.3g}")
        #input("Press Enter to continue...")
        #print(f"Y (min, max): {np.min(Y):.3g}, {np.max(Y):.3g}")

       
        # Compute Young's modulus based on temperature
        fe_solver_structural.mat_prop = [
            mat_lib.create_material_with_defaults(
                name=f"Material_{i+1}", 
                youngs_modulus=E[i].item(), 
                yield_strength=Y[i].item(),
                poissons_ratio=0.3
            )
            for i in range(num_elems)
        ]
       
        fe_solver_structural.set_structural_material(fe_solver_structural.mat_prop)

        # Solve FEA and compute objective/constraints/gradients
        #print("Solving structural FEA...")
        uvw = fe_solver_structural.solve(xDesign.detach().cpu().numpy(), MaterialModel.SIMP)

       
        fe_solver_structural.postprocess() # to compute stresses etc.
        #fe_solver_structural.plot_deformation()
        obj, grad_obj = compute_mmto_objective_and_gradient(
            to_params, uvw, T, zeta, fe_solver_structural, KETemplate, matEncoder)
     
        cons, grad_cons = compute_mmto_constraint_and_gradient(
            to_params, uvw, T, zeta, fe_solver_structural, KETemplate, matEncoder)

        if (obj0 is None): # For the first iteration
            obj0 = obj
            # Check if any constraints are violated (>0) and print a warning
            if np.any(cons > 0):
                print(50 * "-")
                print("Warning: Constraint(s) violated at start of optimization!")
                print("GCMMA may not converge for this problem. Consider changing constraints if convergence issues occur.")
                print(50 * "-")
        
        obj = obj / obj0  # Normalize objective
        grad_obj = grad_obj / obj0

        # Apply filter to sensitivities
        grad_obj[0:num_elems] = (H * grad_obj[0:num_elems]) / Hs
        grad_obj[num_elems:2*num_elems] = (H * grad_obj[num_elems:2*num_elems]) / Hs
        grad_obj[2*num_elems:3*num_elems] = (H * grad_obj[2*num_elems:3*num_elems]) / Hs

        for i in range(grad_cons.shape[0]):  # For each constraint
            grad_cons[i, 0:num_elems] = (H * grad_cons[i, 0:num_elems]) / Hs
            grad_cons[i, num_elems:2*num_elems] = (H * grad_cons[i, num_elems:2*num_elems]) / Hs
            grad_cons[i, 2*num_elems:3*num_elems] = (H * grad_cons[i, 2*num_elems:3*num_elems]) / Hs

        grad_obj = np.array([grad_obj]).reshape((num_design_var, 1))
        cons = np.array(cons).reshape((-1, 1))
        grad_cons = np.array(grad_cons).reshape((len(cons), num_design_var))

     
        # Extract names for printing
        objective_name = getattr(to_params.Objective[0], 'name', str(to_params.Objective[0]))
        constraint_names = [getattr(c[0], 'name', str(c[0])) for c in to_params.Constraints]

        # Print objective and constraints for this iteration
        print(f"Min. Objective ({objective_name}): {obj*obj0:.3g}")
        for idx, val in enumerate(cons.flatten()):
            print(f"Constraint {idx+1} ({constraint_names[idx]}): {(val+1)*to_params.Constraints[idx][2]:.3g} <= {to_params.Constraints[idx][2]:.3g}?")


       
        # Add penalty to objective to keep designs close to training data
        if (iterationCount > nIterationsWithoutPenalization):
            p_softmin = -6
            d_ij = torch.cdist(zPts, zRealTorch, p=2) + 1e-12
            min_i = torch.sum(d_ij ** p_softmin, dim=1).pow(1.0/p_softmin)
            min_i = min_i * xDesign
            penalty = gamma * torch.sum(min_i) / num_elems
            zetaTensor.grad = None
            penalty.backward(retain_graph=True)
            grad_obj[num_elems:,0] += zetaTensor.grad[num_elems:].detach().numpy()
            obj = obj + penalty.item()

            # # Apply filter to grad_obj for the penalty term
            if False:
                grad_obj[num_elems:2*num_elems, 0] = (H * grad_obj[num_elems:2*num_elems, 0]) / Hs
                grad_obj[2*num_elems:3*num_elems, 0] = (H * grad_obj[2*num_elems:3*num_elems, 0]) / Hs
            gamma = min(gamma*gamma_factor, gamma_max)


        iterationCount += 1
        return obj, grad_obj, cons, grad_cons


    # Initialize the design variables
    
    x0 = 0.5 * np.ones(num_elems) 
    x0 = (H * x0) / Hs

    #z0 = np.random.uniform(-2,2, size=(2 * num_elems,))  
    z0 = np.zeros(2 * num_elems)
    if z0_init_method == Z0InitMethod.LIGHTEST:
        zLightest = matEncoder.getLightestMaterial()
        z0[0:num_elems] = zLightest[0]
        z0[num_elems:2*num_elems] = zLightest[1]
    elif z0_init_method == Z0InitMethod.HEAVIEST:
        zHeaviest = matEncoder.getHeaviestMaterial()
        z0[0:num_elems] = zHeaviest[0]
        z0[num_elems:2*num_elems] = zHeaviest[1]
    elif z0_init_method == Z0InitMethod.ORIGIN:
        z0[0:num_elems] = 0.0
        z0[num_elems:2*num_elems] = 0.0
    elif z0_init_method == Z0InitMethod.UNIFORM:
        z0[0:num_elems] = np.random.uniform(-0.5, 0.5, size=num_elems)
        z0[num_elems:2*num_elems] = np.random.uniform(-0.5, 0.5, size=num_elems)
    else:
        raise ValueError(f"Unknown z0_init_method: {z0_init_method}")


    z0[0:num_elems] = (H * z0[0:num_elems])/Hs
    z0[num_elems:2*num_elems] = (H * z0[num_elems:2*num_elems]) / Hs
    zeta0 = np.concatenate((x0, z0), axis=0).reshape(-1, 1)  # shape: (3*num_elems, 1)

    lowerBound = np.zeros(num_design_var, dtype=float).reshape(-1, 1)
    upperBound = np.ones(num_design_var, dtype=float).reshape(-1, 1)
    # Set bounds for material latent variables
   
    lowerBound[num_elems:3*num_elems] = np.min(zRealTorch.cpu().numpy())
    upperBound[num_elems:3*num_elems] = np.max(zRealTorch.cpu().numpy())

    nVariables = num_design_var
    nConstraints = len(to_params.Constraints)
    tStart = time.time()
    maxMMAIterations = nIterationsWithoutPenalization + nIterationsWithPenalization


    # Run the MMA optimization
    optResults = runMMA(nVariables, nConstraints, MMTO_optimization_function, zeta0.reshape(-1, 1), lowerBound,
        upperBound, maxIterations=maxMMAIterations, timeLimitSecs=timeLimit,
        move_limit=0.2, kktTol=1e-6, fTolerance=rel_conv_tol, gTolerance=rel_conv_tol, verbose=False)
    zetaOptimal = optResults[0]
    tEnd = time.time()
    print(f"Total optimization time: {tEnd - tStart:.2f} seconds")
   
  
    # get design variables
    zetaOptimal = np.asarray(zetaOptimal).flatten()
    xDesign = zetaOptimal[0:num_elems]
    zDesign =  zetaOptimal[num_elems:]
    zDesignTensor = torch.tensor(zDesign).float()
    zPts = zDesignTensor.view(2, -1).T
    # get final material properties
    decoded = matEncoder.vaeNet.decoder(zPts)
    material_properties = matEncoder.getMaterialProperties(decoded)
    closest_index = matEncoder.getClosestRealMaterialIndex(zPts)

    if (turnOnThermal):
        # Set material properties for FEA solver
        thermalConductivity = material_properties['K']
        thermalConductivity_elem = thermalConductivity.detach().numpy()
        #print(f"Thermal Conductivity (min, max): {np.min(thermalConductivity_elem):.3g}, {np.max(thermalConductivity_elem):.3g}")
        fe_solver_thermal.mat_prop = [
            mat_lib.create_material_with_defaults(
                name=f"K{i+1}",
                thermal_conductivity=thermalConductivity_elem[i].item())
            for i in range(num_elems)]
        
        #print("Solving thermal FEA...")
        fe_solver_thermal.set_thermal_material(fe_solver_thermal.mat_prop)
        T_full = fe_solver_thermal.solve(xDesign)
        #fe_solver_thermal.plot_temperature()
        edofMat = fe_solver_thermal.mesh.edofMat
        T = np.mean(T_full[edofMat], axis=1)
    else:
        T = np.ones(num_elems) * 20.0  # uniform temperature of 20 degrees C
  
    # --- STRUCTURAL ANALYSIS WITH TEMPERATURE-DEPENDENT MATERIALS ---
    E = matEncoder.getMaterialPropertyAtTemperature( "E",  zPts, T)
    Y = matEncoder.getMaterialPropertyAtTemperature( "Y",  zPts, T)
    
    # Solve and plot
    fe_solver_structural.mesh.setPseudoDensity(xDesign)
    fe_solver_structural.solve(xDesign)
    fe_solver_structural.postprocess()
    
    fe_solver_structural.plot_elem_field(closest_index, title='Mat ID', colormap='tab20')
    fe_solver_structural.plot_elem_field(T, title='Temperature', colormap='plasma')
    fe_solver_structural.plot_elem_field(E, title='Youngs Modulus', colormap='plasma')
    
    
    # Plot latent space with designs and training data
    matEncoder.plotLSR(zRealTorch.detach().cpu().numpy(), zDesign.reshape(2, -1).T, xDesign=xDesign)

if __name__ == "__main__":
    
    to_problem = MMTOExamples.LBracket_TempDependent_ComplianceMassCost

    run_topopt(
        to_problem=to_problem,
        nIterationsWithoutPenalization= 30,
        nIterationsWithPenalization= 70,
        turnOnThermal=True,
        use_pretrained_vae=False
    )