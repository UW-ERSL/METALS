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

def run_topopt(
    use_temp_dependent=False,
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
    gamma_init=100,
    gamma_max=100,
    gamma_factor=1,
    calls_per_stage=10,
    rel_conv_tol=1e-3,
    nDOFDesired=5000,
    to_problem_name="EdgeCantilever",
    thermal_problem_name=None,
    apply_filter_to_materials=True,
    results_filename="topopt_results.pkl"
):
    # --- Set VAE save/load path based on mode ---
    if use_temp_dependent:
        saveNet = './data/vaeNet_ref_tempdependent.nt'
    else:
        saveNet = './data/vaeNet_ref_purestructural.nt'

    # --- Data preprocessing ---
    if use_temp_dependent:
        trainingData, dataInfo, dataIdentifier, trainInfo, Emax = preprocessData_tempdependent()
        numFeatures = trainingData.shape[1]
    else:
        trainingData, dataInfo, dataIdentifier, trainInfo, EMax = preprocessData_structural()
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
        enc_out = materialEncoder.vaeNet.encoder(trainingData)
        if isinstance(enc_out, tuple):
            z_mu = enc_out[0]
        else:
            z_mu = enc_out
        materialEncoder.training_latents = z_mu.cpu()
    # --- Problem setup ---
    # Select TO problem based on user input
    if to_problem_name == "EdgeCantilever":
        to_problem = METALSTOExamples.EdgeCantilever
    elif to_problem_name == "BliskWithBladeMass":
        to_problem = METALSTOExamples.BliskWithBladeMass
    else:
        raise ValueError(f"Unknown TO problem name: {to_problem_name}")
    nDOFDesired = nDOFDesired if nDOFDesired is not None else 20000
    mesh_structural, mat_prop_struct, bc_struct, elem_body_force, to_params = getMETALSTOProblem(to_problem, nDOFDesired=nDOFDesired)

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
    fe_solver_structural.plot_mesh(title="Structural Mesh with Boundary Conditions", plot_bc=True)
    if isinstance(mat_prop_struct, list):
        KE = hex_element_stiffness.hex8_stiffness_matrix_structural(
            mat_prop_struct[0].youngs_modulus, mat_prop_struct[0].poissons_ratio, mesh_structural.elem_size
        )
    else:
        KE = hex_element_stiffness.hex8_stiffness_matrix_structural(
            mat_prop_struct.youngs_modulus, mat_prop_struct.poissons_ratio, mesh_structural.elem_size
        )
    # --- Thermal Problem Selection ---
    print("Selecting thermal problem...")
    if use_temp_dependent:
        valid_thermal = False
        if thermal_problem_name is not None:
            if to_problem == METALSTOExamples.EdgeCantilever:
                if thermal_problem_name == "EdgeCantilever":
                    thermal_problem = METALSThermalExamples.EdgeCantilever
                    valid_thermal = True
                elif thermal_problem_name == "EdgeCantilever_TempBC":
                    thermal_problem = METALSThermalExamples.EdgeCantilever_TempBC
                    valid_thermal = True
            elif to_problem == METALSTOExamples.BliskWithBladeMass:
                if thermal_problem_name == "BliskBlade":
                    thermal_problem = METALSThermalExamples.BliskBlade
                    valid_thermal = True

            if not valid_thermal:
                print(f"Invalid thermal problem '{thermal_problem_name}' for selected TO problem '{to_problem_name}'. Auto-selecting correct thermal problem.")
                if to_problem == METALSTOExamples.EdgeCantilever:
                    thermal_problem = METALSThermalExamples.EdgeCantilever_TempBC
                elif to_problem == METALSTOExamples.BliskWithBladeMass:
                    thermal_problem = METALSThermalExamples.BliskBlade
                else:
                    raise ValueError("No matching thermal problem for selected TO problem.")
                print(f"Selected thermal problem: {thermal_problem.name}")
            else:
                print(f"Selected thermal problem: {thermal_problem.name}")
        else:
            # Auto-select based on TO problem
            if to_problem == METALSTOExamples.EdgeCantilever:
                thermal_problem = METALSThermalExamples.EdgeCantilever_TempBC
            elif to_problem == METALSTOExamples.BliskWithBladeMass:
                thermal_problem = METALSThermalExamples.BliskBlade
            else:
                raise ValueError("No matching thermal problem for selected TO problem.")
            print(f"Selected thermal problem: {thermal_problem.name}")

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
    # --- Patch ID ---
    patch_id = patchwork(mesh_structural, nPatchesDesired=nPatchesDesired)
    num_patches = len(np.unique(patch_id))
    num_elems = mesh_structural.num_elems
    num_design_var = num_elems + num_patches * 2

    # --- MMA Optimization ---
    [H, Hs] = createFilters(fe_solver_structural, to_params)
    shared_vars = {}
    # --- Plot patch id coloring ---
    if plot_patches_flag is True:
        from LSRSupportFunctions import plot_patches
        plot_patches(mesh_structural, nPatchesDesired=nPatchesDesired, title_prefix="Patch ID Coloring of Structural Mesh")
    if use_temp_dependent:
        def build_itertrack_obj(func, gamma_init, calls_per_stage, gamma_max, gamma_factor):
            state = {'calls': 0, 'gamma': gamma_init}
            def wrapper(x):
                result = func(x, state['gamma'])
                state['calls'] += 1
                if state['calls'] % calls_per_stage == 0:
                    if state['gamma'] < gamma_max:
                        state['gamma'] *= gamma_factor
                        if state['gamma'] > gamma_max:
                            state['gamma'] = gamma_max
                    print(f"Stage {state['calls']//calls_per_stage}, gamma = {state['gamma']}")
                return result
            return wrapper

        def mma_obj(x, gamma):
            return optimizationFunction_tempdependent(
                x, fe_solver_structural, fe_solver_thermal, to_params, materialEncoder,
                patch_id, num_patches, num_elems, num_design_var, H, Hs, KE, shared_vars, gamma=gamma, debug=debug, apply_filter_to_materials=apply_filter_to_materials
            )

        mma_obj_wrapped = build_itertrack_obj(mma_obj, gamma_init, calls_per_stage, gamma_max, gamma_factor)
    else:
        def build_itertrack_obj(func, gamma_init, calls_per_stage, gamma_max, gamma_factor):
            state = {'calls': 0, 'gamma': gamma_init}
            def wrapper(x):
                result = func(x, state['gamma'])
                state['calls'] += 1
                if state['calls'] % calls_per_stage == 0:
                    if state['gamma'] < gamma_max:
                        state['gamma'] *= gamma_factor
                        if state['gamma'] > gamma_max:
                            state['gamma'] = gamma_max
                    print(f"Stage {state['calls']//calls_per_stage}, gamma = {state['gamma']}")
                return result
            return wrapper
        def mma_obj(x,gamma):
            return optimizationFunction_structural(
                x, fe_solver_structural, to_params, materialEncoder,
                patch_id, num_patches, num_elems, num_design_var, H, Hs, KE, materialEncoder, shared_vars,gamma=gamma, debug=debug, apply_filter_to_materials=apply_filter_to_materials
            )
        mma_obj_wrapped = build_itertrack_obj(mma_obj, gamma_init, calls_per_stage, gamma_max, gamma_factor)

    # Initial guess
    if random_latent_init:
        latent_init = np.random.uniform(0, 1, size=(2 * num_patches, 1))
    else:
        latent_init = np.zeros((2 * num_patches, 1))
    mma_init = np.concatenate((0.5 * np.ones((num_elems, 1)), latent_init), axis=0)
    lowerBound = np.zeros(num_design_var, dtype=float).reshape(-1, 1)
    upperBound = np.ones(num_design_var, dtype=float).reshape(-1, 1)
    nVariables = num_design_var
    nConstraints = 1

    # --- Run MMA ---
    if use_temp_dependent:
        print("Running MMA optimization (temperature-dependent case)...")
    else:
        print("Running MMA optimization (pure structural case)...")
    [xOptimal, f0val, df0dx, gval, dgdx, nFEAs] = runMMA(
        nVariables, nConstraints, mma_obj_wrapped, mma_init.reshape(-1, 1), lowerBound,
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

    # # Plot mass density field
    # if hasattr(fe_solver_structural, "plot_elem_field"):
    #     fe_solver_structural.plot_elem_field(
    #         np.asarray(shared_vars.get('massDensity', None) if 'massDensity' in shared_vars else None),
    #         title='Mass Density',
    #         colormap='cividis'
    #     )

    # Plot latent space
    with torch.no_grad():
        z_real = materialEncoder.vaeNet.encoder(trainingData)
        if isinstance(z_real, tuple):
            z_real = z_real[0]
        z_real_np = z_real.cpu().numpy()
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
    if 'history' in shared_vars:
        history = shared_vars['history']
    else:
        history = {}

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
    final_compliance = None
    final_mass = shared_vars.get('final_mass', None)
    target_mass = to_params.Constraints[0][2]
    if 'history' in shared_vars and 'compliance' in shared_vars['history']:
        final_compliance = shared_vars['history']['compliance'][-1]
    if 'history' in shared_vars and 'mass' in shared_vars['history']:
        shared_vars['final_mass'] = shared_vars['history']['mass'][-1]
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
        'to_problem_name': to_problem_name,
        'thermal_problem_name': thermal_problem_name,
        'nDOFDesired': nDOFDesired,
        'nPatchesDesired': nPatchesDesired,
        'latentDim': latentDim,
        'vae_hiddenDim': vae_hiddenDim,
        'apply_filter_to_materials': apply_filter_to_materials,
        'use_temp_dependent': use_temp_dependent,
        'results_filename': results_filename,
    }
    with open(results_filename, 'wb') as f:
        pickle.dump(results_to_save, f)
    print(f"Results saved to {results_filename}")

if __name__ == "__main__":

    run_topopt(
        use_temp_dependent=False,
        nPatchesDesired=0,
        random_latent_init=True,
        debug=False,
        maxMMAIterations=50,
        use_pretrained_vae=True,  
        plot_patches_flag=False,
        gamma_init=1e-2,
        gamma_max=1000,
        gamma_factor=100,
        calls_per_stage=10,
        rel_conv_tol=1e-4,
        nDOFDesired=10000,
        to_problem_name="EdgeCantilever", #EdgeCantilever or BliskWithBladeMass
        thermal_problem_name="EdgeCantilever_TempBC", # "BliskBlade", "EdgeCantilever", "EdgeCantilever_TempBC"
        apply_filter_to_materials=False,
        results_filename="EdgeCantilever_NoFilterMat.pkl"
    # For EdgeCantilever, the correct thermal problem(s) are "EdgeCantilever_TempBC" or "EdgeCantilever"
    # For BliskWithBladeMass, the correct thermal problem(s) are "BliskBlade".
    # If incorrect thermal problem is specified wrt the selected TO problem, the correct one will be used after issuing a warning.
    # For EdgeCantilever, "EdgeCantilever_TempBC" will be used by default unless specified
    )
    """
    -------------------------------------------------------------------------------
    | Parameter            | Description                                         | Default Value         |
    -------------------------------------------------------------------------------
    | use_temp_dependent   | Use temperature-dependent optimization              | False                |
    | nPatchesDesired      | Number of patches for mesh coloring                 | 8                    |
    | random_latent_init   | Random initialization of latent variables           | True                 |
    | debug                | Enable debug mode                                   | False                |
    | maxMMAIterations     | Maximum MMA optimization iterations                 | 200                  |
    | timeLimit            | Time limit for optimization (seconds)               | 7200                 |
    | klFactor             | KL divergence factor for VAE training               | 5e-5                 |
    | learningRate         | Learning rate for VAE training                      | 2e-3                 |
    | numEpochs            | Number of epochs for VAE training                   | 40000                |
    | vae_hiddenDim        | Hidden layer dimension for VAE                      | 250                  |
    | latentDim            | Latent space dimension for VAE                      | 2                    |
    | saveNet              | Path to save/load VAE network                       | None                 |
    | use_pretrained_vae   | Use pre-trained VAE network                         | False                |
    | plot_patches_flag    | Plot mesh patch coloring                            | False                |
    | gamma_init           | Initial penalty factor for distance penalization    | 100                  |
    | gamma_max            | Maximum penalty factor for distance penalization    | 100                  |
    | gamma_factor         | Multiplicative factor for penalty update            | 1                    |
    | calls_per_stage      | MMA calls per penalty update stage                  | 10                   |
    | rel_conv_tol         | Relative convergence tolerance                      | 1e-3                 |
    | nDOFDesired          | Desired number of mesh degrees of freedom           | 5000                 |
    | to_problem_name      | Name of topology optimization problem               | "EdgeCantilever"     |
    | thermal_problem_name | Name of thermal problem (if applicable)             | None                 |
    -------------------------------------------------------------------------------
    """