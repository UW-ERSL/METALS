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


def run_topopt(
    to_problem,
    timeLimit=10*60*60,
    saveNet=None,
    plot_progress=True,
    use_pretrained_vae=False,
    use_penalization=False,
    snap_to_real_material=True,
    rel_conv_tol = 1e-7,
    maxIterations = 150,
    binarize_topology = True,
    z0_init_method = Z0InitMethod.ORIGIN,  
    gamma_init = 1e-6, # penalization
    gamma_max = 100,
    gamma_factor = 1.25):#1.1
    
    history = {
        "objective": [],
        "constraints": []
    }

    mesh_structural, mat_prop_struct, bc_struct,\
        elem_body_force, to_params, vae_params = getMMTOProblem(to_problem)

    if to_params.MaterialsExcelFile is None:
        print("Please provide a valid MaterialsExcelFile in to_params.")
        return
    matEncoder = MaterialEncoder(vae_params)
    matEncoder.readExcel(to_params.MaterialsExcelFile)

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
    # matEncoder.plotLSRContours("Youngs_Modulus", title="Young's Modulus Contours in Latent Space")
    # matEncoder.plotLSRContours("Density", title="Density Contours in Latent Space")
    # matEncoder.plotLSRContours("Cost", title="Cost Contours in Latent Space")
    zRealPoints = matEncoder.training_latents

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
    # --- Generalize latent dimension ---
    latentDim = matEncoder.vae_params.latentDim
    num_design_var = num_elems + num_elems * latentDim
    print(f"Using latent dimension: {latentDim}")
    #fe_solver_structural.plot_mesh(plot_bc=True, offsetArrow=True)  
    # Create the filter for density and material variables
    print("Creating filter...")
    [H, Hs] = createFilters(fe_solver_structural, to_params)

    iterationCount = 0
    obj0 = None
    gamma = gamma_init

    def MMTO_optimization_function(zeta):
        nonlocal iterationCount, obj0, gamma, zRealPoints

        zeta = np.asarray(zeta).flatten()
        print("-------------- Iteration", iterationCount, "-----------------")
        
        # Prepare tensors and decode material properties
        zetaTensor = torch.tensor(zeta, dtype=torch.float32, requires_grad=True)
        xDesign = zetaTensor[0:num_elems]
        zDesign = zetaTensor[num_elems:]
        zPoints = zDesign.view(latentDim, -1).T

        xNumpy = xDesign.detach().cpu().numpy()
        grey_elements = np.sum((xNumpy > 0.1) & (xNumpy < 0.9))
        fraction_grey = (grey_elements / num_elems)
        print(f"Percentange grey elements:", f"{fraction_grey*100:.2f}%")

        decoded = matEncoder.vaeNet.decoder(zPoints)
        material_properties = matEncoder.getMaterialProperties(decoded)
        Youngs_Modulus = material_properties['Youngs_Modulus'].detach().numpy()

        fe_solver_structural.mat_prop = [
            mat_lib.create_material_with_defaults(name=f"Material_{i+1}", youngs_modulus=Youngs_Modulus[i])
            for i in range(len(Youngs_Modulus))]
        fe_solver_structural.set_structural_material(fe_solver_structural.mat_prop)
        
        sol = fe_solver_structural.solve(xDesign.detach().cpu().numpy(), MaterialModel.SIMP)
        fe_solver_structural.mesh.setPseudoDensity(xDesign.detach().cpu().numpy())
        fe_solver_structural.postprocess()

        if (plot_progress):
           fe_solver_structural.plot_pseudo_density(
                    plotter=None,
                   auto_close=False,
                   title=f"Iter {len(history['objective']) + 1} - Density"
               )


        obj, grad_obj = compute_mmto_objective_and_gradient(
            to_params, sol, zeta, fe_solver_structural, KETemplate, matEncoder)
        cons, grad_cons = compute_mmto_constraint_and_gradient(
            to_params, sol, zeta, fe_solver_structural, KETemplate, matEncoder)

        if (obj0 is None):
            obj0 = obj
        
        if any(c > 0.5 for c in cons.flatten()): # if any constraint is significantly violated, zero out objective gradient
            grad_obj *= 0 # MMA step will try to reduce constraint violation first

        obj = obj / obj0
        grad_obj = grad_obj / obj0

        if (to_params.ElemsToKeep is not None):
            grad_obj[to_params.ElemsToKeep] = min(grad_obj)

        # Apply filter to sensitivities
        grad_obj[0:num_elems] = (H * grad_obj[0:num_elems]) / Hs
        for i in range(latentDim):
            grad_obj[num_elems + i*num_elems : num_elems + (i+1)*num_elems] = \
                (H * grad_obj[num_elems + i*num_elems : num_elems + (i+1)*num_elems]) / Hs

        for i in range(grad_cons.shape[0]):
            grad_cons[i, 0:num_elems] = (H * grad_cons[i, 0:num_elems]) / Hs
            for j in range(latentDim):
                grad_cons[i, num_elems + j*num_elems : num_elems + (j+1)*num_elems] = \
                    (H * grad_cons[i, num_elems + j*num_elems : num_elems + (j+1)*num_elems]) / Hs

        grad_obj = np.array([grad_obj]).reshape((num_design_var, 1))
        cons = np.array(cons).reshape((-1, 1))
        grad_cons = np.array(grad_cons).reshape((len(cons), num_design_var))

        objective_name = getattr(to_params.Objective[0], 'name', str(to_params.Objective[0]))
        constraint_names = [getattr(c[0], 'name', str(c[0])) for c in to_params.Constraints]

        print(f"Min. Objective ({objective_name}): {obj*obj0:.5g}")
        for idx, val in enumerate(cons.flatten()):
            inequality = '<='
            if constraint_names[idx] == "STRESS_SAFETY_FACTOR" or constraint_names[idx] == "TEMPERATURE_SAFETY_FACTOR":
                inequality = '>='
            print(f"Constraint {idx+1} ({constraint_names[idx]}): {(val+1)*to_params.Constraints[idx][2]:.3g} {inequality} {to_params.Constraints[idx][2]:.3g}?")

        history["objective"].append(obj)
        history["constraints"].append(cons.flatten().copy())
    
        if (use_penalization):
            d_ij = torch.cdist(zPoints, zRealPoints)
            min_i = torch.min(d_ij, dim=1).values
            min_i = min_i * xDesign
            nActiveElems = np.ceil(torch.sum((xDesign).float()).item())
            avgClosestDistance = torch.sum(min_i) / nActiveElems   
            maxClosestDistance = torch.max(min_i).item()
            penalty = gamma * avgClosestDistance
            zetaTensor.grad = None
            penalty.backward(retain_graph=True)
            gradMeanDistance = zetaTensor.grad[num_elems:].detach().numpy()
            # for i in range(latentDim):
            #     gradMeanDistance[i*num_elems:(i+1)*num_elems] = \
            #         (H * gradMeanDistance[i*num_elems:(i+1)*num_elems]) / Hs
            grad_obj[num_elems:,0] += gradMeanDistance
            obj = obj + penalty.item()
            gamma = min(gamma*gamma_factor, gamma_max)

        iterationCount += 1
        #print(np.linalg.norm(grad_obj[:num_elems]), np.linalg.norm(grad_obj[num_elems:]),np.linalg.norm(grad_cons))
        return obj, grad_obj, cons, grad_cons

       # Check if there's a volume fraction constraint and set initial density accordingly
    initialDensity = 0.5
    for constraint in to_params.Constraints:
        if constraint[0] == TO_QOI.VOLUME_FRACTION:
            initialDensity = constraint[2]  # Use the constraint value as initial density
            break
    # Initialize the design variables
    x0 = initialDensity * np.ones(num_elems) 
    x0 = (H * x0) / Hs

    z0 = np.zeros(latentDim * num_elems)
    if z0_init_method == Z0InitMethod.LIGHTEST:
        zLightest = matEncoder.getLightestMaterial()
        for i in range(latentDim):
            z0[i*num_elems:(i+1)*num_elems] = zLightest[i]
    elif z0_init_method == Z0InitMethod.HEAVIEST:
        zHeaviest = matEncoder.getHeaviestMaterial()
        for i in range(latentDim):
            z0[i*num_elems:(i+1)*num_elems] = zHeaviest[i]
    elif z0_init_method == Z0InitMethod.ORIGIN:
        z0[:] = 0.0
    elif z0_init_method == Z0InitMethod.UNIFORM:
        for i in range(latentDim):
            z0[i*num_elems:(i+1)*num_elems] = np.random.uniform(-0.5, 0.5, size=num_elems)
    else:
        raise ValueError(f"Unknown z0_init_method: {z0_init_method}")

    for i in range(latentDim):
        z0[i*num_elems:(i+1)*num_elems] = (H * z0[i*num_elems:(i+1)*num_elems]) / Hs
    zeta0 = np.concatenate((x0, z0), axis=0).reshape(-1, 1)

    lowerBound = np.zeros(num_design_var, dtype=float).reshape(-1, 1)
    upperBound = np.ones(num_design_var, dtype=float).reshape(-1, 1)

    lowerBound[num_elems:num_design_var] = np.min(zRealPoints.cpu().numpy())
    upperBound[num_elems:num_design_var] = np.max(zRealPoints.cpu().numpy())

    nVariables = num_design_var
    nConstraints = len(to_params.Constraints)
    tStart = time.time()
    maxMMAIterations = maxIterations

    optResults = runMMA(nVariables, nConstraints, MMTO_optimization_function, zeta0.reshape(-1, 1), lowerBound,
        upperBound, maxIterations=maxMMAIterations, timeLimitSecs=timeLimit,
        move_limit=0.05, kktTol=1e-6, fTolerance=rel_conv_tol, gTolerance=rel_conv_tol, verbose=False)
    zetaOptimal = optResults[0]
    tEnd = time.time()
    print(f"Total optimization time: {tEnd - tStart:.2f} seconds")
   
    zetaOptimal = np.asarray(zetaOptimal).flatten()
    xOptimal = zetaOptimal[0:num_elems]
    zOptimal =  zetaOptimal[num_elems:]
    zOptimalPts = torch.tensor(zOptimal).view(latentDim, -1).T.float()

    if (binarize_topology):
        x_sorted = np.sort(xOptimal)
        threshold = x_sorted[int((1-np.mean(xOptimal))*len(xOptimal))]
        xOptimal = np.where(xOptimal < threshold, 0.0, 1.0)
    
    if (snap_to_real_material):
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

    matEncoder.plotLSR(zRealPoints.detach().cpu().numpy(), zOptimalPts, xDesign=xOptimal)

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
    plt.legend()
    plt.grid()
    plt.show()
    material_indices = matEncoder.getClosestRealMaterialIndex(zOptimalPts)  # shape: (num_elems,)
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
    elif excel_file == './DataConstantTemperature/20MaterialsTeledyne.xlsx' or excel_file == './DataConstantTemperature/20MaterialsTeledyneSimple.xlsx':
        material_colors = {
            0: '#1b1b1b',  # charcoal black
            1: '#004d00',  # deep forest green
            2: '#800000',  # dark maroon
            3: '#ffb6c1',  # light pink
            4: '#008080',  # teal
            5: '#b8860b',  # dark goldenrod
            6: '#2f4f4f',  # dark slate gray
            7: '#ff4500',  # orange red
            8: '#6a5acd',  # slate blue
            9: '#228b22',  # forest green
            10: '#9932cc', # dark orchid
            11: '#8b0000', # dark red
            12: '#e6194b', # vibrant red
            13: '#3cb44b', # vibrant green
            14: '#ffe119', # vibrant yellow
            15: '#4363d8', # vibrant blue
            16: '#f58231', # vibrant orange
            17: '#911eb4', # vibrant purple
            18: '#42d4f4', # vibrant cyan
            19: '#f032e6', # vibrant magenta

        }
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
    
    to_problem = MMTOExamples.Bridge_Compliance_MassCost


    run_topopt(
        to_problem=to_problem,
        use_penalization=True,
        use_pretrained_vae=True,
        snap_to_real_material=True,
    )