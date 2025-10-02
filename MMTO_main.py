import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
from METALS_TO_examples import METALSTOExamples, getMETALSTOProblem
from materialEncoder import MaterialEncoder
from ReadMaterialData import ReadMaterialData
from MMTO_obj_cons_sensitivities import (
    compute_mmto_objective_and_gradient,
    compute_mmto_constraint_and_gradient,
)
from PyTOImports import *


iterationCount = 0
gamma_init = 1e-3
gamma_max = 1000
gamma_factor = 2


# The main code for METALS topology optimization
def run_topopt(
    to_problem,
    debug=False,
    nIterationsWithoutPenalization=50,
    nIterationsWithPenalization=50,
    timeLimit=7200,
    saveNet=None,
    use_pretrained_vae=True,
    rel_conv_tol=1e-7,
    nDOFDesired=5000,
    apply_filter_to_materials=True):
    
    # --- Get the TO problem
    mesh_structural, mat_prop_struct, bc_struct, elem_body_force, to_params, vae_params = getMETALSTOProblem(to_problem, nDOFDesired=nDOFDesired)
    
    # --- Read the materials excel file ---
    if to_params.MaterialsExcelFile  is None:
        print("Please provide a valid MaterialsExcelFile in to_params.")
        return
    material_data = ReadMaterialData(to_params.MaterialsExcelFile)


    # Push the material data to VAE
    scaledMaterialData = material_data.scaledMaterialData
    materialAttributes = material_data.materialAttributes
    materialNames = material_data.materialNames


    numAttributes = scaledMaterialData.shape[1]
    vaeSettings = {
        'encoder': {'inputDim': numAttributes, 'hiddenDim': vae_params.vae_hiddenDim, 'latentDim': vae_params.latentDim},
        'decoder': {'latentDim': vae_params.latentDim, 'hiddenDim': vae_params.vae_hiddenDim, 'outputDim': numAttributes}
    }
    
    matEncoder = MaterialEncoder(scaledMaterialData, materialAttributes, materialNames, vaeSettings)


    # Train the autoencoder or save/load from file
    if saveNet is None:
        base, _ = os.path.splitext(to_params.MaterialsExcelFile)
        saveNet = base + ".nt"
    vae_file_exists = os.path.exists(saveNet) and os.path.getsize(saveNet) > 0

    if use_pretrained_vae and vae_file_exists:
        print(f"Loading pre-trained autoencoder from file: {saveNet}")
        matEncoder.loadAutoencoderFromFile(saveNet)
        with torch.no_grad():
            z_real_np = matEncoder.vaeNet.encoder(scaledMaterialData).cpu().numpy()
    else:
        print(f"Training autoencoder from scratch and saving to: {saveNet}")
        matEncoder.trainAutoencoder(vae_params.numEpochs, vae_params.klFactor, saveNet, vae_params.learningRate)
        with torch.no_grad():
            z_real_np = matEncoder.vaeNet.encoder(scaledMaterialData).cpu().numpy()
    with torch.no_grad():
        matEncoder.training_latents = matEncoder.vaeNet.encoder(scaledMaterialData).cpu()


    # Optionally plot the latent space
    if (False):
        for attributeId in range(numAttributes):
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
        rtol=1e-8,
        elem_body_force=elem_body_force
    )
    KETemplate = hex_element_stiffness.hex8_stiffness_matrix_structural(
            mat_prop_struct.youngs_modulus, mat_prop_struct.poissons_ratio, mesh_structural.elem_size)
    
    num_elems = mesh_structural.num_elems
    num_design_var = num_elems + num_elems * 2

    # Create the filter for density and material variables
    print("Creating filter...")
    [H, Hs] = createFilters(fe_solver_structural, to_params)
    shared_vars = {}
    

    def METALS_optimization_function(zeta): # zeta contains both density and latent variables
        print("-------------- ", iterationCount, " -----------------")
        zeta = np.asarray(zeta).flatten()
        #zeta = matEncoder.unnormalize_last_n(arr=zeta, n=2*num_elems); # KS: Removed
        zetaTensor = torch.tensor(zeta, dtype=torch.float32, requires_grad=True)
        xDesign = zetaTensor[0:num_elems]
        zD = zetaTensor[num_elems:]
        zDesign = zD.view(num_elems, -1)
        decoded = matEncoder.vaeNet.decoder(zDesign)
        material_properties = matEncoder.getMaterialProperties(decoded)
        if 'youngs_modulus' in material_properties:
            EDesign = material_properties['youngs_modulus']
            fe_solver_structural.mat_prop = [
                mat_lib.create_material_with_defaults(name=f"Material_{i+1}", youngs_modulus=EDesign[i])
                for i in range(len(EDesign))]
            fe_solver_structural.set_structural_material(fe_solver_structural.mat_prop)
        else: # we need at least Young's modulus to proceed
            print("Young's modulus not found in decoded material properties.")
            return None
        
        sol = fe_solver_structural.solve(xDesign.detach().cpu().numpy(), MaterialModel.SIMP)
        obj, grad_obj = compute_mmto_objective_and_gradient(to_params, sol, zeta, fe_solver_structural, KETemplate, matEncoder)
        cons, grad_cons = compute_mmto_constraint_and_gradient(to_params, sol, zeta, fe_solver_structural, KETemplate, matEncoder)

        grad_obj[0:num_elems] = (H * grad_obj[0:num_elems]) / Hs
        grad_cons[0:num_elems] = (H * grad_cons[0:num_elems]) / Hs
        if apply_filter_to_materials:
            grad_obj[num_elems:2*num_elems] = (H * grad_obj[num_elems:2*num_elems]) / Hs
            grad_obj[2*num_elems:3*num_elems] = (H * grad_obj[2*num_elems:3*num_elems]) / Hs
            grad_cons[num_elems:2*num_elems] = (H * grad_cons[num_elems:2*num_elems]) / Hs
            grad_cons[2*num_elems:3*num_elems] = (H * grad_cons[2*num_elems:3*num_elems]) / Hs
        grad_obj = np.array([grad_obj]).reshape((num_design_var, 1))
        cons = np.array(cons).reshape((-1, 1))
        grad_cons = np.array(grad_cons).reshape((len(cons), num_design_var))
        shared_vars['zDesign'] = zDesign.detach().cpu().numpy() if hasattr(zDesign, 'detach') else zDesign
        shared_vars['EDesign'] = material_properties.get('youngs_modulus', None)
        if 'safety_factor' in material_properties:
            shared_vars['safety_factor'] = material_properties['safety_factor']
        return obj, grad_obj, cons, grad_cons


    # Initialize the design variables
    nConstraints = len(to_params.Constraints)
    x0 = 0.5 * np.ones((num_elems, 1))
    x0= (H * x0) / Hs

    z0 = np.random.uniform(0, 1, size=(2 * num_elems, 1))
    if (apply_filter_to_materials): 
        z0[0:num_elems,0] = (H * z0[0:num_elems,0]) / Hs
        z0[num_elems:2*num_elems,0] = (H * z0[num_elems:2*num_elems,0]) / Hs

    zeta0 = np.concatenate((x0, z0), axis=0)
    lowerBound = np.zeros(num_design_var, dtype=float).reshape(-1, 1)
    upperBound = np.ones(num_design_var, dtype=float).reshape(-1, 1)
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
    EDesign = shared_vars['EDesign']
    fe_solver_structural.mesh.setPseudoDensity(xDesign)
    fe_solver_structural.solve()
    fe_solver_structural.postprocess()
    fe_solver_structural.plot_elem_field(EDesign, title='YoungModulus', colormap='viridis')
    fe_solver_structural.plot_vonMisesStress()
    if 'safety_factor' in shared_vars:
        fe_solver_structural.plot_elem_field(shared_vars['safety_factor'], title=' Stress SF', colormap='viridis')
    if hasattr(fe_solver_structural, "plot_elem_field") and 'thermalConductivity' in shared_vars:
        fe_solver_structural.plot_elem_field(
            np.asarray(shared_vars.get('thermalConductivity', None) if 'thermalConductivity' in shared_vars else None),
            title='Thermal Conductivity',
            colormap='plasma'
        )
    with torch.no_grad():
        z_real_np = matEncoder.vaeNet.encoder(scaledMaterialData).cpu().numpy()
    z_opt = zDesign if isinstance(zDesign, np.ndarray) else zDesign.detach().cpu().numpy()
    matEncoder.plotLSR(z_real_np, zDesign=z_opt.reshape(-1, 2))

if __name__ == "__main__":
    
    to_problem = METALSTOExamples.Bridge

    nDOFDesired = 5000
    run_topopt(
        to_problem=to_problem,
        debug=False,
        nIterationsWithoutPenalization=50,
        nIterationsWithPenalization=0,
        use_pretrained_vae=True,
        rel_conv_tol=1e-7,
        nDOFDesired=nDOFDesired,
        apply_filter_to_materials=True
    )