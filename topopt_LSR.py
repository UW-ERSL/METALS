from sympy import gamma
from LSRSupportFunctions import *
from METALS_TO_examples import METALSTOExamples, getMETALSTOProblem
from METALS_thermal_examples import METALSThermalExamples, getMETALSThermalProblem
import hex_structural_fea
import hex_thermal_fea
import torch
import matplotlib.pyplot as plt
import os
import pickle
from materialEncoder import MaterialEncoder
from LSRImports import *
from ReadMaterialData import ReadMaterialData
from enum import Enum, auto
from scipy.spatial import ConvexHull


import numpy as np


def plotLSRContour(zReal,matEncoder,id = 0):
    # Sample the latent space from -3 to 3 in 2D
    n_points = 100
    z1 = np.linspace(-3, 3, n_points)
    z2 = np.linspace(-3, 3, n_points)
    Z1, Z2 = np.meshgrid(z1, z2)
    Z_grid = np.stack([Z1.ravel(), Z2.ravel()], axis=1)

    # Decode each latent point and get Young's modulus
    QOI = []
    with torch.no_grad():
        for z in Z_grid:
            z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
            decoded = matEncoder.vaeNet.decoder(z_tensor)
            decodedValues = matEncoder.getMaterialProperties_structuralyield(decoded)
            QOI.append(decodedValues[id].item())
     

    QOI = np.array(QOI).reshape(Z1.shape)

    # Plot the contour
    plt.figure(figsize=(6, 5))
    contour = plt.contourf(Z1, Z2, QOI, levels=30, cmap='viridis')
    plt.colorbar(contour, label="id")
    plt.scatter(zReal[:, 0], zReal[:, 1], c='black', marker='*', s=200, label='Real Materials', alpha=1.0)
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    plt.title("Contour of id in Latent Space")
    plt.legend()
    plt.show()
   

def plotLatentSpace(zReal, dataIdentifier, zDesign=None):
    """Plot the latent space with real and designed materials.

    Args:
        zReal: Numpy array of shape (num_real_materials, latentDim) for real materials.
        zDesign: Optional numpy array of shape (num_design_materials, latentDim) for designed materials.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    if zDesign is not None:
        ax.scatter(zDesign[:, 0], zDesign[:, 1], c='red', marker='o', s=20, label='Optimized Materials', alpha=0.2)
    # Plot real material points with labels
    ax.scatter(zReal[:, 0], zReal[:, 1], c='black', marker='*', s=200, label='real materials', alpha=1.0)
    for i, label in enumerate(dataIdentifier['name']):
        ax.text(zReal[i, 0] + 0.1, zReal[i, 1], str(label), fontsize=12, color='black', ha='center', va='bottom')
    ax.set_xlabel('$z_1$')
    ax.set_ylabel('$z_2$')
    ax.set_title('Latent Space')
    ax.legend()
    ax.set_aspect('equal', 'box')
    plt.grid(True)
    plt.show()

class ProblemType(Enum):
    PURE_STRUCTURAL = auto()
    TEMP_DEPENDENT = auto()
    BENCHMARK = auto()
    BENCHMARK_COST = auto()
    STRUCTURAL_YIELD = auto()

def run_topopt(
    to_problem,
    thermal_problem=None,
    problem_type=ProblemType.PURE_STRUCTURAL,
    nPatchesDesired=0,
    debug=False,
    nIterationsWithoutPenalization=50,
    nIterationsWithPenalization=50,
    timeLimit =7200,
    klFactor= 5e-6,
    learningRate = 2e-4,
    numEpochs = 20000,
    vae_hiddenDim=500,
    latentDim=2,
    saveNet=None,
    use_pretrained_vae=True,
    rel_conv_tol=1e-7,
    nDOFDesired=5000,
    apply_filter_to_materials=True,
    material_excel_file=None,
    results_filename="topopt_results.pkl"
):
    # --- Set gamma and normalization based on penalization flag ---
   
    gamma_init = 1e-3
    gamma_max = 1000
    gamma_factor = 2 

    # --- Problem-type-dependent settings ---
    if problem_type == ProblemType.PURE_STRUCTURAL:
        default_excel = './data/TeledyneDatabase.xlsx'
        default_vae = './data/vaeNet_ref_purestructural.nt'
    elif problem_type == ProblemType.TEMP_DEPENDENT:
        default_excel = './data/TeledyneDatabase2_Temp_scaled.xlsx'
        default_vae = './data/vaeNet_ref_tempdependent.nt'
    elif problem_type == ProblemType.BENCHMARK:
        default_excel = './data/BenchmarkDatabase.xlsx'
        default_vae = './data/vaeNet_ref_benchmark.nt'
    elif problem_type == ProblemType.BENCHMARK_COST:
        default_excel =  './data/BenchmarkDatabaseCost.xlsx' #'./data/TeledyneDatabase_Cost.xlsx'
        default_vae = './data/vaeNet_ref_benchmark_cost.nt'
    elif problem_type == ProblemType.STRUCTURAL_YIELD:
        default_excel = './data/LBracketDatabase.xlsx'
        default_vae = './data/vaeNet_ref_structural_yield.nt'
    else:
        raise ValueError("Unknown problem type.")

    # --- Data preprocessing ---
    if material_excel_file is None:
        material_excel_file = default_excel
    if saveNet is None:
        saveNet = default_vae

    material_data = ReadMaterialData(material_excel_file)
    trainingData = material_data.trainingData

    dataInfo = material_data.dataInfo
    
    dataIdentifier = material_data.dataIdentifier
  
    numFeatures = trainingData.shape[1]

    vaeSettings = {
        'encoder': {'inputDim': numFeatures, 'hiddenDim': vae_hiddenDim, 'latentDim': latentDim},
        'decoder': {'latentDim': latentDim, 'hiddenDim': vae_hiddenDim, 'outputDim': numFeatures}
    }
    matEncoder = MaterialEncoder(trainingData, dataInfo, dataIdentifier, vaeSettings)

    # --- Check if VAE file exists and is non-empty ---
    vae_file_exists = os.path.exists(saveNet) and os.path.getsize(saveNet) > 0

    # Train or load VAE
    if use_pretrained_vae and vae_file_exists:
        print(f"Loading pre-trained autoencoder from file: {saveNet}")
        matEncoder.loadAutoencoderFromFile(saveNet)
        with torch.no_grad():
            z_real_np = matEncoder.vaeNet.encoder(trainingData).cpu().numpy()
    else:
        print(f"Training autoencoder from scratch and saving to: {saveNet}")
        matEncoder.trainAutoencoder(numEpochs, klFactor, saveNet, learningRate)
        with torch.no_grad():
            z_real_np = matEncoder.vaeNet.encoder(trainingData).cpu().numpy()
        
    # After loading or training the VAE
    with torch.no_grad():
        matEncoder.training_latents = matEncoder.vaeNet.encoder(trainingData).cpu()
    zReal = matEncoder.vaeNet.encoder.z.detach().numpy()
    #plotLatentSpace(zReal,dataIdentifier=matEncoder.dataIdentifier)

    #plotLSRContour(zReal,matEncoder,id = 2)
    matEncoder.constraints = {}

    # --- Problem setup ---
    mesh_structural, mat_prop_struct, bc_struct, elem_body_force, to_params = getMETALSTOProblem(
        to_problem, nDOFDesired=nDOFDesired)

    solver = lin_solv.Solvers.PARDISO
    dsolver = deflation.DeflationSolver()
    if solver == lin_solv.Solvers.DPCG:
        nGroups = min(dsolver.maxGroups, max(dsolver.minGroups, round(3 * mesh_structural.num_nodes / dsolver.dofPerGroup)))
        dsolver.create_deflation_groups(mesh_structural, nGroups)
        dsolver.create_deflation_matrix(mesh_structural)
        dsolver.W = dsolver.W[bc_struct.free_dofs, :]

    fe_solver_structural = hex_structural_fea.HexStructuralFEA(
        mesh=mesh_structural,
        mat_prop=mat_prop_struct,
        bc=bc_struct,
        solver=solver,
        dsolver=dsolver,
        rtol=1e-8,
        elem_body_force=elem_body_force
    )
    #fe_solver_structural.plot_mesh(title="Structural Mesh with Boundary Conditions", plot_bc=True)
    if isinstance(mat_prop_struct, list):
        KE = hex_element_stiffness.hex8_stiffness_matrix_structural(
            mat_prop_struct[0].youngs_modulus, mat_prop_struct[0].poissons_ratio, mesh_structural.elem_size
        )
    else:
        KE = hex_element_stiffness.hex8_stiffness_matrix_structural(
            mat_prop_struct.youngs_modulus, mat_prop_struct.poissons_ratio, mesh_structural.elem_size
        )

    # --- Thermal Problem ---
    if problem_type == ProblemType.TEMP_DEPENDENT and thermal_problem is not None:
        mesh_thermal, mat_prop_thermal, bc_thermal = getMETALSThermalProblem(
            thermal_problem, nDOFDesired=nDOFDesired
        )
        fe_solver_thermal = hex_thermal_fea.HexThermalFEA(
            mesh=mesh_thermal,
            mat_prop=mat_prop_thermal,
            bc=bc_thermal,
            solver=solver,
            rtol=1e-8
        )
    else:
        fe_solver_thermal = None

    # --- Patch ID ---
    patch_id = patchwork(mesh_structural, nPatchesDesired=nPatchesDesired)
    num_patches = len(np.unique(patch_id))
    num_elems = mesh_structural.num_elems
    num_design_var = num_elems + num_patches * 2

    # --- MMA Optimization ---
    print("Creating filter...")
    [H, Hs] = createFilters(fe_solver_structural, to_params)
    shared_vars = {}

    # --- Gamma as mutable object for update ---
    gammaStruct = {'value': 0}
    iterationCount = 0
    # --- MMA Objective Functions ---
    if problem_type == ProblemType.TEMP_DEPENDENT:
        def mma_obj(x):
            nonlocal iterationCount
            print("-------------- ", iterationCount, " -----------------")
            obj, grad_obj, cons, grad_cons = optimizationFunction_tempdependent(
                x, fe_solver_structural, fe_solver_thermal, to_params, matEncoder,
                patch_id, num_patches, num_elems, num_design_var, H, Hs, KE, shared_vars,
                gamma=gammaStruct['value'],
                debug=debug,
                apply_filter_to_materials=apply_filter_to_materials,
            )
            iterationCount += 1
            if (iterationCount < nIterationsWithoutPenalization):
                gammaStruct['value'] = 0
            elif (iterationCount == nIterationsWithoutPenalization):
                gammaStruct['value'] = gamma_init
                print(f"Gamma updated to: {gammaStruct['value']}")

            gammaStruct['value'] = min(gammaStruct['value'] * gamma_factor, gamma_max)
            return obj, grad_obj, cons, grad_cons
        nConstraints = 1
    elif problem_type == ProblemType.BENCHMARK_COST:
        def mma_obj(x):
            nonlocal iterationCount
            print("-------------- ", iterationCount, " -----------------")
            obj, grad_obj, cons, grad_cons = optimizationFunction_structuralcost(
                x, fe_solver_structural, to_params,
                num_elems, num_design_var, H, Hs, KE, matEncoder, shared_vars,
                gamma=gammaStruct['value'],
                debug=debug,
                apply_filter_to_materials=apply_filter_to_materials,
            )
            iterationCount += 1
            if (iterationCount < nIterationsWithoutPenalization):
                gammaStruct['value'] = 0
            elif (iterationCount == nIterationsWithoutPenalization):
                gammaStruct['value'] = gamma_init
                print(f"Gamma updated to: {gammaStruct['value']}")

            gammaStruct['value'] = min(gammaStruct['value'] * gamma_factor, gamma_max)
            return obj, grad_obj, cons, grad_cons
        nConstraints = 2
    elif problem_type == ProblemType.STRUCTURAL_YIELD:
        def mma_obj(x):
            nonlocal iterationCount
            print("-------------- ", iterationCount, " -----------------")
            obj, grad_obj, cons, grad_cons = optimizationFunction_structuralyield(
                x, fe_solver_structural, to_params,
                num_elems, num_design_var, H, Hs, KE, matEncoder, shared_vars,
                gamma=gammaStruct['value'],
                debug=debug,
                apply_filter_to_materials=apply_filter_to_materials,
            )
            
            iterationCount += 1
            if (iterationCount < nIterationsWithoutPenalization):
                gammaStruct['value'] = 0
            elif (iterationCount == nIterationsWithoutPenalization):
                gammaStruct['value'] = gamma_init
                print(f"Gamma updated to: {gammaStruct['value']}")

            gammaStruct['value'] = min(gammaStruct['value'] * gamma_factor, gamma_max)
            
            return obj, grad_obj, cons, grad_cons
        nConstraints = 2
    else:
        def mma_obj(x):
            nonlocal iterationCount
            print("-------------- ", iterationCount, " -----------------")
            obj, grad_obj, cons, grad_cons = optimizationFunction_structural(
                x, fe_solver_structural, to_params,
                patch_id, num_patches, num_elems, num_design_var, H, Hs, KE, matEncoder, shared_vars,
                gamma=gamma['value'],
                debug=debug,
                apply_filter_to_materials=apply_filter_to_materials,
            )
            iterationCount += 1
            if (iterationCount < nIterationsWithoutPenalization):
                gammaStruct['value'] = 0
            elif (iterationCount == nIterationsWithoutPenalization):
                gammaStruct['value'] = gamma_init

            gammaStruct['value'] = min(gammaStruct['value'] * gamma_factor, gamma_max)
            return obj, grad_obj, cons, grad_cons
        nConstraints = 1

    # Initial guess

    latent_init = np.random.uniform(0, 1, size=(2 * num_patches, 1))
    #latent_init = 0.1 * np.ones((2 * num_patches, 1))

    #plotLatentSpace(zReal, dataIdentifier=matEncoder.dataIdentifier,latent_init.reshape(-1, 2))
       
    if (apply_filter_to_materials): 
        latent_init[0:num_patches,0] = (H * latent_init[0:num_patches,0]) / Hs
        latent_init[num_patches:2*num_patches,0] = (H * latent_init[num_patches:2*num_patches,0]) / Hs
    mma_init = np.concatenate((0.5 * np.ones((num_elems, 1)), latent_init), axis=0)
    lowerBound = np.zeros(num_design_var, dtype=float).reshape(-1, 1)
    upperBound = np.ones(num_design_var, dtype=float).reshape(-1, 1)
    nVariables = num_design_var
   
    ## Run MMA Optimization ---
    tStart = time.time()
    maxMMAIterations = nIterationsWithoutPenalization + nIterationsWithPenalization
    [xOptimal, f0val, df0dx, gval, dgdx, nFEAs] = runMMA(
        nVariables, nConstraints, mma_obj, mma_init.reshape(-1, 1), lowerBound,
        upperBound, maxIterations=maxMMAIterations, timeLimitSecs=timeLimit,
        move_limit=0.2, kktTol=1e-6, fTolerance=rel_conv_tol, gTolerance=rel_conv_tol, verbose=False
    )
    tEnd = time.time()
    print(f"Total optimization time: {tEnd - tStart:.2f} seconds")
    # --- Store final mass after optimization ---
    if 'history' in shared_vars and 'mass' in shared_vars['history'] and len(shared_vars['history']['mass']) > 0:
        shared_vars['final_mass'] = shared_vars['history']['mass'][-1]
    else:
        shared_vars['final_mass'] = shared_vars.get('current_mass', None)
    # --- Postprocess and Plot ---
    x = np.asarray(xOptimal).flatten()
    xDesign = x[0:num_elems]
    zDesign = shared_vars['zDesign']
    EDesign = shared_vars['EDesign']
    fe_solver_structural.mesh.setPseudoDensity(xDesign)
    fe_solver_structural.plot_elem_field(EDesign, title='YoungModulus', colormap='viridis')
    # Plot thermal conductivity field
    if hasattr(fe_solver_structural, "plot_elem_field") and 'thermalConductivity' in shared_vars:
        fe_solver_structural.plot_elem_field(
            np.asarray(shared_vars.get('thermalConductivity', None) if 'thermalConductivity' in shared_vars else None),
            title='Thermal Conductivity',
            colormap='plasma'
        )

    # Plot latent space
    with torch.no_grad():
        z_real_np = matEncoder.vaeNet.encoder(trainingData).cpu().numpy()
    z_opt = zDesign if isinstance(zDesign, np.ndarray) else zDesign.detach().cpu().numpy()
    plotLatentSpace(z_real_np, dataIdentifier=matEncoder.dataIdentifier, zDesign=z_opt.reshape(-1, 2))

if __name__ == "__main__":
    run_topopt(
        to_problem=METALSTOExamples.BridgeMMTOCost,
        thermal_problem=None,
        problem_type=ProblemType.BENCHMARK_COST,
        debug=False,
        nIterationsWithoutPenalization= 50,
        nIterationsWithPenalization = 50,
        use_pretrained_vae=False,
        rel_conv_tol=1e-7,
        nDOFDesired=10000,
        apply_filter_to_materials=True,
        results_filename="Temp.pkl"
    )
    """
    Runs topology optimization with VAE-based material design.

     - Results are saved to `results_filename` and plots are shown.
    - You can view and compare results by running the postprocess_topopt_lsr.py file.

    | Parameter                  | Type      | Default                  | Description                                                                 |
    |----------------------------|-----------|--------------------------|-----------------------------------------------------------------------------|
    | to_problem                 | object    | (required)               | Topology optimization problem definition object                             |
    | thermal_problem            | object    | None                     | Thermal problem definition object (optional, for temp-dependent problems)   |
    | problem_type               | enum      | PURE_STRUCTURAL          | Problem type: PURE_STRUCTURAL, TEMP_DEPENDENT, BENCHMARK (may add more)                    |
    | nPatchesDesired            | int       | 8                        | Number of patches for patchwork coloring                                    |
    | random_latent_init         | bool      | True                     | Randomly initialize latent variables                                        |
    | debug                      | bool      | False                    | Enable debug mode                                                           |
    | maxMMAIterations           | int       | 200                      | Maximum number of MMA iterations                                            |
    | timeLimit                  | int/float | 7200                     | Time limit for MMA optimization (seconds)                                   |
    | klFactor                   | float     | 5e-5                     | KL divergence factor for VAE training                                       |
    | learningRate               | float     | 2e-3                     | Learning rate for VAE training                                              |
    | numEpochs                  | int       | 40000                    | Number of epochs for VAE training                                           |
    | vae_hiddenDim              | int       | 250                      | Hidden layer dimension for VAE                                              |
    | latentDim                  | int       | 2                        | Latent space dimension for VAE                                              |
    | saveNet                    | str       | None                     | Path to save/load VAE network                                               |
    | use_pretrained_vae         | bool      | False                    | Use a pre-trained VAE if available                                          |
    | plot_patches_flag          | bool      | False                    | Plot patchwork coloring                                                     |
    | use_penalization           | bool      | True                     | Use penalization for LSR constraint                                         |
    | rel_conv_tol               | float     | 1e-3                     | Relative convergence tolerance for MMA                                      |
    | nDOFDesired                | int       | 5000                     | Desired number of degrees of freedom in mesh                                |
    | apply_filter_to_materials  | bool      | True                     | Apply filter to material properties                                         |
    | material_excel_file        | str       | None                     | Path to material property Excel file                                        |
    | results_filename           | str       | "topopt_results.pkl"     | Output filename for saving results                                          |

    """