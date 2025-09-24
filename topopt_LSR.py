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

class ProblemType(Enum):
    PURE_STRUCTURAL = auto()
    TEMP_DEPENDENT = auto()
    BENCHMARK = auto()
    BENCHMARK_COST = auto()

def run_topopt(
    to_problem,
    thermal_problem=None,
    problem_type=ProblemType.PURE_STRUCTURAL,
    nPatchesDesired=8,
    random_latent_init=True,
    debug=False,
    maxMMAIterations=200,
    timeLimit=7200,
    klFactor=5e-5,
    learningRate=2e-3,
    numEpochs=40000,
    vae_hiddenDim=250,
    latentDim=2,
    saveNet=None,
    use_pretrained_vae=False,
    plot_patches_flag=False,
    use_penalization=True,
    rel_conv_tol=1e-3,
    nDOFDesired=5000,
    apply_filter_to_materials=True,
    material_excel_file=None,
    results_filename="topopt_results.pkl"
):
    # --- Set gamma and normalization based on penalization flag ---
    if use_penalization:
        gamma_init = 0
        gamma_max = 1000
        gamma_factor = 0
        print(f"Penalization is ENABLED (gamma_init={gamma_init}, gamma_max={gamma_max}, gamma_factor={gamma_factor}).")
    else:
        gamma_init = 0
        gamma_max = 0
        gamma_factor = 1
        print(f"Penalization is DISABLED. Ellipse-based normalization will be used for latent variables.")

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
        default_excel = './data/BenchmarkDatabaseCost.xlsx'
        default_vae = './data/vaeNet_ref_benchmark_cost.nt'
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
    materialEncoder = MaterialEncoder(trainingData, dataInfo, dataIdentifier, vaeSettings)

    # --- Check if VAE file exists and is non-empty ---
    vae_file_exists = os.path.exists(saveNet) and os.path.getsize(saveNet) > 0

    # Train or load VAE
    if use_pretrained_vae and vae_file_exists:
        print(f"Loading pre-trained autoencoder from file: {saveNet}")
        materialEncoder.loadAutoencoderFromFile(saveNet)
    else:
        print(f"Training autoencoder from scratch and saving to: {saveNet}")
        materialEncoder.trainAutoencoder(numEpochs, klFactor, saveNet, learningRate)
    # After loading or training the VAE
    with torch.no_grad():
        materialEncoder.training_latents = materialEncoder.vaeNet.encoder(trainingData).cpu()
    zReal = materialEncoder.vaeNet.encoder.z.detach().numpy()

    # Ellipse constraint setup (only if using ellipse LSR)
    if not use_penalization:
        enclosing_ellipse = welzl(np.array(zReal, dtype=float))
        center,a,b,t = enclosing_ellipse
        constraints = {'distance': {'isOn':False, 'center':center, 'a':a, 'b':b, 'theta':t, 'delta':0.0, 'beta':20}}
        materialEncoder.constraints = constraints
    else:
        materialEncoder.constraints = {}

    # --- Problem setup ---
    mesh_structural, mat_prop_struct, bc_struct, elem_body_force, to_params = getMETALSTOProblem(
        to_problem, nDOFDesired=nDOFDesired
    )

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
    [H, Hs] = createFilters(fe_solver_structural, to_params)
    shared_vars = {}

    # --- Patch plotting ---
    if plot_patches_flag:
        from LSRSupportFunctions import plot_patches
        plot_patches(mesh_structural, nPatchesDesired=nPatchesDesired, title_prefix="Patch ID Coloring of Structural Mesh")

    # --- Gamma as mutable object for update ---
    gamma = {'value': gamma_init}

    # --- MMA Objective Functions ---
    if problem_type == ProblemType.TEMP_DEPENDENT:
        def mma_obj(x):
            obj, grad_obj, cons, grad_cons = optimizationFunction_tempdependent(
                x, fe_solver_structural, fe_solver_thermal, to_params, materialEncoder,
                patch_id, num_patches, num_elems, num_design_var, H, Hs, KE, shared_vars,
                gamma=gamma['value'],
                debug=debug,
                apply_filter_to_materials=apply_filter_to_materials,
                use_penalization=use_penalization
            )
            if use_penalization:
                gamma['value'] = min(gamma['value'] * gamma_factor, gamma_max)
                print(f"Gamma updated to: {gamma['value']}")
            return obj, grad_obj, cons, grad_cons
        nConstraints = 1
    elif problem_type == ProblemType.BENCHMARK_COST:
        def mma_obj(x):
            obj, grad_obj, cons, grad_cons = optimizationFunction_structuralcost(
                x, fe_solver_structural, to_params, materialEncoder,
                patch_id, num_patches, num_elems, num_design_var, H, Hs, KE, materialEncoder, shared_vars,
                gamma=gamma['value'],
                debug=debug,
                apply_filter_to_materials=apply_filter_to_materials,
                use_penalization=use_penalization
            )
            if use_penalization:
                gamma['value'] = min(gamma['value'] * gamma_factor, gamma_max)
                print(f"Gamma updated to: {gamma['value']}")
            return obj, grad_obj, cons, grad_cons
        nConstraints = 2
    else:
        def mma_obj(x):
            obj, grad_obj, cons, grad_cons = optimizationFunction_structural(
                x, fe_solver_structural, to_params, materialEncoder,
                patch_id, num_patches, num_elems, num_design_var, H, Hs, KE, materialEncoder, shared_vars,
                gamma=gamma['value'],
                debug=debug,
                apply_filter_to_materials=apply_filter_to_materials,
                use_penalization=use_penalization
            )
            if use_penalization:
                gamma['value'] = min(gamma['value'] * gamma_factor, gamma_max)
                print(f"Gamma updated to: {gamma['value']}")
            return obj, grad_obj, cons, grad_cons
        nConstraints = 1

    # Initial guess
    if random_latent_init: 
        latent_init = np.random.uniform(0, 1, size=(2 * num_patches, 1))
        # latent_init = 0.1*np.ones((2 * num_patches, 1))
    else:
        latent_init = np.zeros((2 * num_patches, 1))
    if (apply_filter_to_materials): 
        latent_init[0:num_patches,0] = (H * latent_init[0:num_patches,0]) / Hs
        latent_init[num_patches:2*num_patches,0] = (H * latent_init[num_patches:2*num_patches,0]) / Hs
    mma_init = np.concatenate((0.5 * np.ones((num_elems, 1)), latent_init), axis=0)
    lowerBound = np.zeros(num_design_var, dtype=float).reshape(-1, 1)
    upperBound = np.ones(num_design_var, dtype=float).reshape(-1, 1)
    nVariables = num_design_var
    # --- Run MMA ---
    if problem_type == ProblemType.TEMP_DEPENDENT:
        print("Running MMA optimization with temperature-dependent LSR...")
        nConstraints = 1
    elif problem_type == ProblemType.PURE_STRUCTURAL:
        print("Running MMA optimization with pure structural LSR...")
        nConstraints = 1
    elif problem_type == ProblemType.BENCHMARK:
        print("Running MMA optimization with benchmark LSR...")
        nConstraints = 1
    elif problem_type == ProblemType.BENCHMARK_COST:
        print("Running MMA optimization with benchmark cost LSR...")
        nConstraints = 2
    else:
        raise ValueError("Unknown problem type.")
    [xOptimal, f0val, df0dx, gval, dgdx, nFEAs] = runMMA(
        nVariables, nConstraints, mma_obj, mma_init.reshape(-1, 1), lowerBound,
        upperBound, maxIterations=maxMMAIterations, timeLimitSecs=timeLimit,
        move_limit=0.2, kktTol=1e-6, fTolerance=rel_conv_tol, gTolerance=rel_conv_tol, verbose=True
    )
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
        z_real_np = materialEncoder.vaeNet.encoder(trainingData).cpu().numpy()
    z_opt = zDesign if isinstance(zDesign, np.ndarray) else zDesign.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(z_real_np[:, 0], z_real_np[:, 1], c='black', marker='*', s=80, label='Real Materials', alpha=1.0)
    ax.scatter(z_opt[:, 0], z_opt[:, 1], c='red', marker='o', s=40, label='Optimized Materials', alpha=0.5)
    ax.set_xlabel('$z_1$')
    ax.set_ylabel('$z_2$')
    ax.set_title('Optimized Materials vs Real Materials in Latent Space')
    ax.legend()
    ax.set_aspect('equal', 'box')
    plt.grid(True)
    plt.show()

    # --- Plot compliance and volume fraction history ---
    history = shared_vars.get('history', {})
    if 'compliance' in history and 'volfrac' in history:
        fig, ax1 = plt.subplots()
        ax1.plot(history['compliance'], 'b-', label='Compliance')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Compliance', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.plot(history['volfrac'], 'r--', label='Volume Fraction')
        ax2.set_ylabel('Volume Fraction', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        plt.title('Compliance and Volume Fraction vs Iteration')
        fig.tight_layout()
        plt.show()
    else:
        print("No compliance/volume fraction history found in shared_vars['history'].")

    # --- Print compliance and mass summary ---
    initial_compliance = shared_vars.get('J0', None)
    final_compliance = history['compliance'][-1] if 'compliance' in history else None
    final_mass = shared_vars.get('final_mass', None)
    target_mass = to_params.Constraints[0][2]
    if 'mass' in history:
        shared_vars['final_mass'] = history['mass'][-1]
    else:
        shared_vars['final_mass'] = shared_vars.get('current_mass', None)

    print("\n--- Optimization Summary ---")
    if initial_compliance is not None:
        print(f"Initial compliance: {initial_compliance:.4f}")
    if final_compliance is not None:
        print(f"Final compliance: {final_compliance:.4f}")
        if initial_compliance is not None:
            percent_change = 100 * (final_compliance - initial_compliance) / initial_compliance
            print(f"Percent change in compliance: {percent_change:+.2f}%")
    if final_mass is not None:
        print(f"Final mass: {final_mass:.4f}")
    print("Target mass: ", target_mass)
    print("--- End of Summary ---\n")
    # Save results
    results_to_save = {
        'xDesign': xDesign,
        'EDesign': EDesign,
        'zDesign': zDesign,
        'history': history,
        'thermalConductivity': shared_vars.get('thermalConductivity', None),
        'massDensity': shared_vars.get('massDensity', None),
        'z_real': z_real_np,
        'initial_compliance': initial_compliance,
        'final_compliance': final_compliance,
        'final_mass': final_mass,
        'target_mass': target_mass,
        'to_problem': to_problem,
        'thermal_problem': thermal_problem,
        'nDOFDesired': nDOFDesired,
        'nPatchesDesired': nPatchesDesired,
        'latentDim': latentDim,
        'vae_hiddenDim': vae_hiddenDim,
        'apply_filter_to_materials': apply_filter_to_materials,
        'problem_type': problem_type,
        'results_filename': results_filename,
    }
    with open(results_filename, 'wb') as f:
        pickle.dump(results_to_save, f)
    print(f"Results saved to {results_filename}")

if __name__ == "__main__":
    # Example: Run BridgeMMTOCost problem with BENCHMARK_COST type
    run_topopt(
        to_problem=METALSTOExamples.BridgeMMTOCost,
        thermal_problem=None,
        problem_type=ProblemType.BENCHMARK_COST,
        nPatchesDesired=0,
        random_latent_init=True,
        debug=False,
        maxMMAIterations=100,
        use_pretrained_vae=True,
        plot_patches_flag=False,
        use_penalization=True,
        rel_conv_tol=1e-7,
        nDOFDesired=50000,
        apply_filter_to_materials=True,
        results_filename="BridgeMMTOCost_YesPenalization0pt035kg_150iter_50000DOF.pkl"
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