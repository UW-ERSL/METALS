import numpy as np
import torch
import time
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
    z0_init_method = Z0InitMethod.HEAVIEST,
    gamma_init = 1e-3, # penalization
    gamma_max = 100,#100
    gamma_factor =1.1,
    massfrac=None):#1.1
    
    history = {
        "objective": [],
        "constraints": []
    }
    # --- Get the TO problem
 
    mesh_structural, mat_prop_struct, bc_struct,\
          elem_body_force, to_params, vae_params = getMMTOProblem(to_problem)

    # --- Read the materials excel file ---
    if to_params.MaterialsExcelFile  is None:
        print("Please provide a valid MaterialsExcelFile in to_params.")
        return
    matEncoder = MaterialEncoder(vae_params)
    matEncoder.readExcel(to_params.MaterialsExcelFile)
    if massfrac is not None:
        to_params.Constraints = [(TO_QOI.MASS, None, massfrac * mesh_structural.num_elems)]
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
    matEncoder.plotLSRContours("Youngs_Modulus", title="Young's Modulus Contours in Latent Space")
    matEncoder.plotLSRContours("Density", title="Density Contours in Latent Space")
    # matEncoder.plotLSRContours("Cost", title="Cost Contours in Latent Space")
    zRealPoints = matEncoder.training_latents
    
    if (False): # optionally plot latent space contours
        matEncoder.plotLSRContours("Youngs_Modulus")
        matEncoder.plotLSRContours("Density")
        matEncoder.plotLSRContours("Cost")

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
    fe_solver_structural.plot_elem_field(Youngs_Modulus, title='YoungModulus', colormap='viridis')
    # After optimization, before plotting
    material_indices = matEncoder.getClosestRealMaterialIndex(zOptimalPts)  # shape: (num_elems,)
    # if to_problem == MMTOExamples.CantileverBenchmark_Compliance_Mass or to_problem == MMTOExamples.CenterCantilever_Compliance_Mass:
    # # Cantilever benchmark colors
    #     material_colors = {
    #         0: '#fe4d02', 
    #         1: '#e6fd1a',
    #         2: '#1dfde1',
    #         3: '#004fff',
    #         4: '#020a86',
    #     }
    # elif to_problem == MMTOExamples.Bridge_Compliance_MassCost or to_problem == MMTOExamples.Bridge_Compliance_Mass:
    # # Bridge benchmark colors
    #     material_colors = {
    #         0: '#04fd05', 
    #         1: '#0505f0',
    #         2: '#ef0711',
    #     }
    # elif to_problem == MMTOExamples.Bridge_Compliance_MassCost_Saitou:
    # # Bridge benchmark colors (Saitou et al.)
    #     material_colors = {
    #         0: '#0201fc', 
    #         1: '#f60004',
    #         2: '#080101',
    #     }
    # elif to_problem == MMTOExamples.Table_Compliance_Mass or to_problem == MMTOExamples.CenterCantilever_Compliance_Mass or to_problem == MMTOExamples.Table_Compliance_Mass_Cost or to_problem == MMTOExamples.CenterCantilever_Compliance_Mass_Cost:
    #     material_colors = {
    #         0: '#e6194b',  # vibrant red
    #         1: '#3cb44b',  # vibrant green
    #         2: '#ffe119',  # vibrant yellow
    #         3: '#4363d8',  # vibrant blue
    #         4: '#f58231',  # vibrant orange
    #         5: '#911eb4',  # vibrant purple
    #         6: '#42d4f4',  # vibrant cyan
    #         7: '#f032e6',  # vibrant magenta
    #     }
    # else:
    #     # Default colors for up to 20 materials
    #     default_colors = plt.cm.get_cmap('tab20', matEncoder.nMaterials)
    #     material_colors = {i: default_colors(i) for i in range(matEncoder.nMaterials)} 
    excel_file = to_params.MaterialsExcelFile

    if excel_file == './DataConstantTemperature/5MaterialsCantilever.xlsx':
        material_colors = {
            0: '#fe4d02', 
            1: '#e6fd1a',
            2: '#1dfde1',
            3: '#004fff',
            4: '#020a86',
        }
    elif excel_file == './DataConstantTemperature/3MaterialsBridgev2.xlsx' or excel_file == './DataConstantTemperature/3MaterialsBridge.xlsx':
        material_colors = {
            0: '#04fd05', 
            1: '#0505f0',
            2: '#ef0711',
        }
    elif excel_file == './DataConstantTemperature/20MaterialsTeledyne.xlsx':
        default_colors = plt.cm.get_cmap('tab20', matEncoder.nMaterials)
        material_colors = {i: default_colors(i) for i in range(matEncoder.nMaterials)}
    elif excel_file == './DataConstantTemperature/8Materials.xlsx':
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
    elif excel_file == './DataConstantTemperature/3Materials.xlsx':
        material_colors = {
            0: '#0201fc', 
            1: '#f60004',
            2: '#080101',
        }
    else:
        # Default colors for up to 20 materials
        default_colors = plt.cm.get_cmap('tab20', matEncoder.nMaterials)
        material_colors = {i: default_colors(i) for i in range(matEncoder.nMaterials)}
    colors = [material_colors[int(idx.item()) if hasattr(idx, "item") else int(idx)] for idx in material_indices]
    import matplotlib.colors as mcolors
    rgb_colors = np.array([mcolors.to_rgb(c) for c in colors])
    fe_solver_structural.plot_elem_field(material_indices, title='Real Materials', colors=rgb_colors)
    # Plot latent space with designs and training data
    matEncoder.plotLSR(zRealPoints.detach().cpu().numpy(), zOptimalPts, xDesign=xOptimal)

    # Combined plot for objective and constraints vs. iterations
    plt.figure(figsize=(12, 6))
    
    # Plot objective
    plt.plot(
        range(len(history["objective"])),
        history["objective"],
        label="Objective",
        color="blue",
        linewidth=2,
        marker="o",
        markevery=5
    )
    
    # Plot constraints
    markers = ['s', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors
    for i in range(len(history["constraints"][0])):
        constraint_values = [history["constraints"][j][i] for j in range(len(history["constraints"]))]
        plt.plot(
            range(len(constraint_values)),
            constraint_values,
            label=f"Constraint {i+1}",
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            markevery=5  # Add markers every 5 points for clarity
        )
    
    # Add labels, title, and legend
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Objective and Constraints vs. Iterations")
    plt.legend()

    plt.grid()
    plt.show()

if __name__ == "__main__":
    
    # Temperature independent Mult-Material TO Problems :
    
    # Example 1 (30000 DOF) uses 3 materials, 4 attributes from './DataConstantTemperature/3MaterialsBridge.xlsx'
    # Examples 2-5 (13000 DOF) use 3 materials, 5 attributes from './DataConstantTemperature/3Materials.xlsx'
    # Examples 6-9 (137000 DOF) use 20 materials, 5 attributes from './DataConstantTemperature/20MaterialsTeledyne.xlsx'

    # See MMTO_examples.py for additional details

    # 1. Bridge_Compliance_MassCost (Benchmark Bridge, Minimize Compliance with Mass and Cost constraints)

    # 2. LBracketTopLoad_Compliance_Mass (L-Bracket with Top Load, Minimize Compliance with Mass constraint)
    # 3. LBracketTopLoad_Compliance_MassCost (L-Bracket with Top Load, Minimize Compliance with Mass and Cost constraints)
    # 4. LBracketTopLoad_Compliance_MassCriticality (L-Bracket with Top Load, Minimize Compliance with Mass and Criticality constraints)
    # 5. LBracketTopLoad_Stress_Mass (L-Bracket with Top Load, Minimize Stress with Mass constraint)
   
    # 6. BliskSection_Compliance_MassCost (Blisk Section, Minimize Compliance with Mass and Cost constraints)
    # 7. BliskSection_Compliance_MassCriticality (Blisk Section, Minimize Mass  with Compliance and Criticality constraints)
    # 8. BliskSection_Stress_Mass (Blisk Section, Minimize Stress with Mass constraints)
    
    to_problem = MMTOExamples.Table_Compliance_Mass_Cost
    mass_fracs = [0.05, 0.15, 0.3, 0.45, 0.6, 0.75]
    pareto_compliance = []
    pareto_mass = []

    for mf in mass_fracs:
        # Get the problem and mesh
        mesh_structural, mat_prop_struct, bc_struct, elem_body_force, to_params, vae_params = getMMTOProblem(to_problem)
        # Set the mass constraint for this fraction
        to_params.Constraints = [(TO_QOI.MASS, None, mf * mesh_structural.num_elems)]
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
        
        # Run optimization
        result = run_topopt(
            to_problem=to_problem,
            timeLimit=7200,
            saveNet=None,
            use_pretrained_vae=True,
            use_penalization=True,
            snap_to_real_material=True,
            rel_conv_tol=1e-7,
            maxIterations=100,
            z0_init_method=Z0InitMethod.ORIGIN,
        )
        
        # After optimization, get compliance and mass
        # (Assume run_topopt returns compliance and mass, or you can extract from the solver)
        # For example, if you have access to fe_solver_structural and xOptimal:
        # compliance = fe_solver_structural.get_compliance()
        # used_mass = np.sum(xOptimal)
        # If not, adapt this to your return values

        # Save Pareto data (replace with actual values from your run_topopt)
        pareto_compliance.append(compliance)
        pareto_mass.append(used_mass)

        # Save topology plot
        save_folder = f"results_massfrac_{mf:.2f}"
        os.makedirs(save_folder, exist_ok=True)
        fe_solver_structural.plot_elem_field(
            material_indices,
            title=f"Topology_massfrac_{mf:.2f}",
            colors=rgb_colors,
            save_path=os.path.join(save_folder, f"topology_massfrac_{mf:.2f}.png")
        )

    # Plot Pareto curve
    plt.figure()
    plt.plot(pareto_mass, pareto_compliance, marker='o')
    plt.xlabel('Mass')
    plt.ylabel('Compliance')
    plt.title('Pareto Curve: Compliance vs. Mass Fraction')
    plt.grid()
    plt.savefig("pareto_curve.png")
    plt.show()