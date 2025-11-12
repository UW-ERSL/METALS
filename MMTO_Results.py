import numpy as np
import torch
import time
import os
from MMTO_examples import MMTOExamples, getMMTOProblem
from materialEncoder import MaterialEncoder
from MMTO_obj_cons_sensitivities import (
    compute_mmto_objective_and_gradient,
    compute_mmto_constraint_and_gradient,
)
from PyTOImports import *
import matplotlib.pyplot as plt

from enum import Enum
class Z0InitMethod(Enum):
    LIGHTEST = 'lightest'
    HEAVIEST = 'heaviest'
    ORIGIN = 'origin'
    UNIFORM = 'uniform'


# The main code for MMTO topology optimization
def run_topopt(
    to_problem,
    timeLimit=7200,
    saveNet=None,
    use_pretrained_vae=False,
    use_penalization=False,
    snap_to_real_material=True,
    rel_conv_tol = 1e-7,
    maxIterations = 100,
    z0_init_method = Z0InitMethod.ORIGIN,  
    gamma_init = 1e-3, # penalization
    gamma_max = 100,#100
    gamma_factor =1.1):#1.1
    
    history = {
        "objective": [],
        "constraints": []
    }


    # --- Get the TO problem
    mesh_structural, mat_prop_struct, bc_struct,\
          elem_body_force, to_params, vae_params = getMMTOProblem(to_problem)
    # Format filter size for folder name (replace . with 'pt')
    filter_size = to_params.RelativeFilterRadius
    filter_size_str = str(filter_size).replace('.', 'pt')

    # Get init method name
    init_method_str = str(z0_init_method.name) if isinstance(z0_init_method, Enum) else str(z0_init_method)

    # Build results directory name
    results_dir_name = f"{to_problem.name}_{init_method_str}_{filter_size_str}"
    results_dir = os.path.join(os.getcwd(), results_dir_name)
    os.makedirs(results_dir, exist_ok=True)
    # --- Read the materials excel file ---
    if to_params.MaterialsExcelFile  is None:
        print("Please provide a valid MaterialsExcelFile in to_params.")
        return
    matEncoder = MaterialEncoder(vae_params)
    matEncoder.readExcel(to_params.MaterialsExcelFile)

    # Train the autoencoder or save/load from file
    if saveNet is None:
        base, _ = os.path.splitext(to_params.MaterialsExcelFile)
        saveNet = base + ".nt"
    vae_file_exists = os.path.exists(saveNet) and os.path.getsize(saveNet) > 0

    if use_pretrained_vae and vae_file_exists:
        print(f"Loading pre-trained autoencoder from file: {saveNet}")
        matEncoder.loadAutoencoderFromFile(saveNet)
    else:
        print(f"Training autoencoder and saving to: {saveNet}")
        time_start = time.time()
        matEncoder.trainAutoencoder(vae_params.numEpochs, vae_params.klFactor, saveNet, vae_params.learningRate, vae_params.maxAttributeErrorPercent)
        time_end = time.time()
        print(f"Autoencoder training time: {time_end - time_start:.2f} seconds")
        
    with torch.no_grad():
        matEncoder.training_latents = matEncoder.vaeNet.encoder(matEncoder.scaledMaterialData).cpu()

    matEncoder.printEncodingErrors()
    # Save contour plots for latent space
    for attr, fname in [("Youngs_Modulus", "contour_youngs_modulus"), ("Density", "contour_density")]:
        fig = matEncoder.plotLSRContours(attr, title=f"{attr} Contours in Latent Space")
        if fig is not None:
            fig.tight_layout(pad=0.1)
            fig.savefig(os.path.join(results_dir, f"{fname}.png"), bbox_inches='tight')
            fig.savefig(os.path.join(results_dir, f"{fname}.pdf"), bbox_inches='tight')
            plt.close(fig)
    zRealPoints = matEncoder.training_latents

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
    
    num_dof = fe_solver_structural.bc.num_dofs
    print(f"Number of DOF: {num_dof}")
    num_elems = mesh_structural.num_elems
    num_design_var = num_elems + num_elems * 2
    fe_solver_structural.plot_mesh(plot_bc=True, offsetArrow=True)  
    # Create the filter for density and material variables
    print("Creating filter...")
    [H, Hs] = createFilters(fe_solver_structural, to_params)

    iterationCount = 0
    obj0 = None # will get updated in the first iteration
    gamma = gamma_init

    def MMTO_optimization_function(zeta):
        nonlocal iterationCount, obj0, gamma, zRealPoints
        zeta = np.asarray(zeta).flatten()
        print("-------------- Iteration", iterationCount, "-----------------")
        
        # Prepare tensors and decode material properties
        zetaTensor = torch.tensor(zeta, dtype=torch.float32, requires_grad=True)
        xDesign = zetaTensor[0:num_elems]
        zDesign = zetaTensor[num_elems:]
        zPoints = zDesign.view(2, -1).T

        decoded = matEncoder.vaeNet.decoder(zPoints)
        material_properties = matEncoder.getMaterialProperties(decoded)
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

        if (obj0 is None): # For the first iteration
            obj0 = obj
        
        obj = obj / obj0  # Normalize objective
        grad_obj = grad_obj / obj0

        if (to_params.ElemsToKeep is not None):
            grad_obj[to_params.ElemsToKeep] = min(grad_obj) # also retain elements that are in the keep list

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
        print(f"Min. Objective ({objective_name}): {obj*obj0:.5g}")
        for idx, val in enumerate(cons.flatten()):
            print(f"Constraint {idx+1} ({constraint_names[idx]}): {(val+1)*to_params.Constraints[idx][2]:.3g} <= {to_params.Constraints[idx][2]:.3g}?")

        # Store history
        history["objective"].append(obj)
        history["constraints"].append(cons.flatten().copy())
    
        if (use_penalization):
            d_ij = torch.cdist(zPoints, zRealPoints)
            min_i = torch.min(d_ij, dim=1).values
            min_i = min_i * xDesign # only relevant if element is present
            nActiveElems = np.ceil(torch.sum((xDesign).float()).item())
            avgClosestDistance = torch.sum(min_i) / nActiveElems   
            maxClosestDistance = torch.max(min_i).item()
            penalty = gamma * avgClosestDistance
            zetaTensor.grad = None
            penalty.backward(retain_graph=True)
            gradMeanDistance = zetaTensor.grad[num_elems:].detach().numpy()
            if False: # Apply filter to grad_obj for the penalty term
                gradMeanDistance[0:num_elems] = (H * gradMeanDistance[0:num_elems]) / Hs
                gradMeanDistance[num_elems:2*num_elems] = (H * gradMeanDistance[num_elems:2*num_elems]) / Hs
                
            grad_obj[num_elems:,0] += gradMeanDistance
            obj = obj + penalty.item()
            gamma = min(gamma*gamma_factor, gamma_max)

        iterationCount += 1
        return obj, grad_obj, cons, grad_cons

    # Initialize the design variables
    x0 = 0.5 * np.ones(num_elems) 
    x0 = (H * x0) / Hs

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

    # Apply filter to initial design variables
    z0[0:num_elems] = (H * z0[0:num_elems])/Hs
    z0[num_elems:2*num_elems] = (H * z0[num_elems:2*num_elems]) / Hs
    zeta0 = np.concatenate((x0, z0), axis=0).reshape(-1, 1)  # shape: (3*num_elems, 1)

    # Set bounds for design variables
    lowerBound = np.zeros(num_design_var, dtype=float).reshape(-1, 1)
    upperBound = np.ones(num_design_var, dtype=float).reshape(-1, 1)

    # Set bounds for material latent variables
    lowerBound[num_elems:3*num_elems] = np.min(zRealPoints.cpu().numpy())
    upperBound[num_elems:3*num_elems] = np.max(zRealPoints.cpu().numpy())

    # Set MMA parameters
    nVariables = num_design_var
    nConstraints = len(to_params.Constraints)
    tStart = time.time()
    maxMMAIterations = maxIterations

    # Run the MMA optimization
    optResults = runMMA(nVariables, nConstraints, MMTO_optimization_function, zeta0.reshape(-1, 1), lowerBound,
        upperBound, maxIterations=maxMMAIterations, timeLimitSecs=timeLimit,
        move_limit=0.05, kktTol=1e-6, fTolerance=rel_conv_tol, gTolerance=rel_conv_tol, verbose=False)
    zetaOptimal = optResults[0]
    tEnd = time.time()
    print(f"Total optimization time: {tEnd - tStart:.2f} seconds")
   
    # get design variables
    zetaOptimal = np.asarray(zetaOptimal).flatten()
    xOptimal = zetaOptimal[0:num_elems]
    zOptimal =  zetaOptimal[num_elems:]
    zOptimalPts = torch.tensor(zOptimal).view(2, -1).T.float()

    if (snap_to_real_material): # optionally snap to closest real material
        zSnappedPts = torch.tensor(matEncoder.getClosestRealMaterialZValues(zOptimalPts))
        zetaOptimal[num_elems:] = zSnappedPts.T.flatten().numpy()
        print(50 * "-")
        print("After snapping:")
        print(50 * "-")
        MMTO_optimization_function(zetaOptimal)
        zOptimalPts = zSnappedPts

    decoded = matEncoder.vaeNet.decoder(zOptimalPts)
    material_properties = matEncoder.getMaterialProperties(decoded)
    Youngs_Modulus = material_properties['Youngs_Modulus'].detach().cpu().numpy()
    fe_solver_structural.mesh.setPseudoDensity(xOptimal)
    # Save Young's modulus field plot
    # Save real materials field plot
    custom_camera_position = [
        (64.69850660883577, -71.34025053053703, 66.88944997959193),
        (25.98076211353316, 25.98076211353316, 0.0),
        (-0.16137552824275206, 0.5145899236321491, 0.8421135014833997)
    ]
    fig = fe_solver_structural.plot_elem_field(Youngs_Modulus, title='YoungModulus', colormap='viridis',save_path=os.path.join(results_dir, "youngs_modulus_field.png"), camera_position=custom_camera_position)
    fig2 =  fe_solver_structural.plot_elem_field(Youngs_Modulus, title='YoungModulus', colormap='viridis',save_path=os.path.join(results_dir, "youngs_modulus_field.tif"), camera_position=custom_camera_position)
    if fig is not None:
        fig.tight_layout(pad=0.1)
        fig.savefig(os.path.join(results_dir, "youngs_modulus_field.png"), bbox_inches='tight')
        fig.savefig(os.path.join(results_dir, "youngs_modulus_field.pdf"), bbox_inches='tight')
        plt.close(fig)
    # After optimization, before plotting
    material_indices = matEncoder.getClosestRealMaterialIndex(zOptimalPts)  # shape: (num_elems,)
    if to_problem == MMTOExamples.CantileverBenchmark_Compliance_Mass:
        material_colors = {
            0: '#fe4d02', 
            1: '#e6fd1a',
            2: '#1dfde1',
            3: '#004fff',
            4: '#020a86',
        }
    elif to_problem == MMTOExamples.Bridge_Compliance_MassCost or to_problem == MMTOExamples.Bridge_Compliance_Mass or to_problem == MMTOExamples.MBBBeam_Compliance_Mass:
        material_colors = {
            0: '#04fd05', 
            1: '#0505f0',
            2: '#ef0711',
        }
    elif to_problem == MMTOExamples.Bridge_Compliance_MassCost_Saitou:
        material_colors = {
            0: '#0201fc', 
            1: '#f60004',
            2: '#080101',
        }
    # elif to_problem == MMTOExamples.MBBBeam_Compliance_Mass:
    #     material_colors = {
    #         0: '#228B22',  # forest green
    #         1: '#FFD700',  # golden
    #         2: '#800080',  # bright purple
    #     }
    elif to_problem == MMTOExamples.Table_Compliance_Mass or to_problem == MMTOExamples.CenterCantilever_Compliance_Mass or to_problem == MMTOExamples.Table_Compliance_Mass_Cost or to_problem == MMTOExamples.CenterCantilever_Compliance_Mass_Cost:
        material_colors = {
            0: '#e6194b',  # vibrant red
            1: '#3cb44b',  # vibrant green
            2: '#ffe119',  # vibrant yellow
            3: '#4363d8',  # vibrant blue
            4: '#f58231',  # vibrant orange
            5: '#911eb4',  # vibrant purple
            6: '#42d4f4',  # vibrant cyan
            7: '#f032e6',  # vibrant magenta
        }
    else:
        default_colors = plt.cm.get_cmap('tab20', matEncoder.nMaterials)
        material_colors = {i: default_colors(i) for i in range(matEncoder.nMaterials)} 

    colors = [material_colors[int(idx)] for idx in material_indices]
    import matplotlib.colors as mcolors
    rgb_colors = np.array([mcolors.to_rgb(c) for c in colors])

    fig = fe_solver_structural.plot_elem_field(material_indices, title='Real Materials', colors=rgb_colors,save_path=os.path.join(results_dir, "real_materials_field.png"), camera_position=custom_camera_position)
    fig2 = fe_solver_structural.plot_elem_field(material_indices, title='Real Materials', colors=rgb_colors,save_path=os.path.join(results_dir, "real_materials_field.tif"), camera_position=custom_camera_position)
    if fig is not None:
        fig.tight_layout(pad=0.1)
        fig.savefig(os.path.join(results_dir, "real_materials_field.png"), bbox_inches='tight')
        fig.savefig(os.path.join(results_dir, "real_materials_field.tif"), bbox_inches='tight')
        plt.close(fig)
    # Plot latent space with designs and training data
    fig = matEncoder.plotLSR(zRealPoints.detach().cpu().numpy(), zOptimalPts, xDesign=xOptimal)
    if fig is not None:
        fig.tight_layout(pad=0.1)
        fig.savefig(os.path.join(results_dir, "latent_space.png"), bbox_inches='tight')
        fig.savefig(os.path.join(results_dir, "latent_space.tif"), bbox_inches='tight')
        plt.close(fig)

    # Combined plot for objective and constraints vs. iterations
    plt.figure(figsize=(12, 6))
    plt.plot(
        range(len(history["objective"])),
        history["objective"],
        label="Objective",
        color="blue",
        linewidth=2,
        marker="o",
        markevery=5
    )
    markers = ['s', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
    colors = plt.cm.tab10.colors
    for i in range(len(history["constraints"][0])):
        constraint_values = [history["constraints"][j][i] for j in range(len(history["constraints"]))]
        plt.plot(
            range(len(constraint_values)),
            constraint_values,
            label=f"Constraint {i+1}",
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            markevery=5
        )
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Objective and Constraints vs. Iterations")
    plt.legend(loc='best', fontsize=10)
    plt.grid()
    plt.tight_layout(pad=0.1)
    fig = plt.gcf()
    fig.savefig(os.path.join(results_dir, "objective_constraints.png"), bbox_inches='tight')
    fig.savefig(os.path.join(results_dir, "objective_constraints.pdf"), bbox_inches='tight')
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    to_problem = MMTOExamples.Table_Compliance_Mass
    run_topopt(
        to_problem=to_problem,
        use_penalization=True,
        use_pretrained_vae=True,
        snap_to_real_material=True,
    )