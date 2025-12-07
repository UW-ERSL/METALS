import numpy as np
import torch
import time
import os
import matplotlib.colors as mcolors

from MMTO_TS_examples import MMTOThermostructuralExamples, getMMTOThermostructuralProblem, material_colors
from materialEncoder import MaterialEncoder
from MMTO_obj_cons_sensitivities_TS import (
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

def run_topopt_TS(
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
    use_continuation = True,
    gamma_init = 1e-3,
    gamma_max = 25,
    gamma_factor = 1.1, 
    plotter=None ):

    history = {
        "objective": [],
        "constraints": []
    }

    # Get problem setup
    mesh, mat_prop, structural_bc,thermal_bc, elem_body_force, to_params, vae_params = getMMTOThermostructuralProblem(to_problem)

  
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
    matEncoder.plotLSR(matEncoder.training_latents.detach().cpu().numpy())
    zRealPoints = matEncoder.training_latents

    solver = linear_solvers.Solvers.PARDISO
    dsolver = deflation.DeflationSolver()
    fe_solver_structural = hex_structural_fea.HexStructuralFEA(
        mesh=mesh,
        mat_prop=mat_prop,
        bc=structural_bc,
        solver=solver,
        dsolver=dsolver,
        rtol=1e-8,
        elem_body_force=elem_body_force)
    fe_solver_thermal = hex_thermal_fea.HexThermalFEA(
        mesh=mesh,
        mat_prop=mat_prop,
        bc=thermal_bc,
        solver=solver,
        dsolver=dsolver,
        rtol=1e-8)

    KETemplate = hex_element_stiffness.hex8_stiffness_matrix_structural(
            1.0, 0.3, mesh.elem_size)
    KTTemplate = hex_element_stiffness.hex8_stiffness_matrix_thermal(
            1.0, mesh.elem_size)

    num_elems = mesh.num_elems
    latentDim = vae_params.latentDim
    num_design_var = num_elems + num_elems * latentDim

    print(f"Using latent dimension: {latentDim}")
    print("Creating filter...")
    [H, Hs] = createFilters(fe_solver_structural, to_params)

    mmaIterations = 0
    obj0 = None
    gamma = gamma_init

    if (use_continuation):
        initialize_SIMP_STRUCTURAL_PENALTY(1.5)
        initialize_SIMP_THERMAL_PENALTY(1)
    else:
        initialize_SIMP_STRUCTURAL_PENALTY(3)   
        initialize_SIMP_THERMAL_PENALTY(1)

    def MMTO_TS_optimization_function(zeta):
        nonlocal mmaIterations, obj0, gamma, zRealPoints

        zeta = np.asarray(zeta).flatten()
        print("-------------- Iteration", mmaIterations, "-----------------")
        
        # Prepare tensors and decode material properties
        zetaTensor = torch.tensor(zeta, dtype=torch.float32, requires_grad=True)
        xDesign = zetaTensor[0:num_elems]
        zDesign = zetaTensor[num_elems:]
        zPoints = zDesign.view(latentDim, -1).T

        xNumpy = xDesign.detach().cpu().numpy()
        grey_elements = np.sum((xNumpy > 0.1) & (xNumpy < 0.9))
        fraction_grey = (grey_elements / num_elems)
        print(f"Percentage grey elements:", f"{fraction_grey*100:.2f}%")

        decoded = matEncoder.vaeNet.decoder(zPoints)
        material_properties = matEncoder.getMaterialProperties(decoded)
        Youngs_Modulus = material_properties['Youngs_Modulus'].detach().cpu().numpy()
        Thermal_Conductivity = material_properties['Conductivity'].detach().cpu().numpy()
        Thermal_Expansion = material_properties['Thermal_Expansion'].detach().cpu().numpy()

        # Set per-element material properties
        fe_solver_structural.mat_prop = [
            mat_lib.create_material_with_defaults(
                name=f"Material_{i+1}",
                youngs_modulus=Youngs_Modulus[i],
                thermal_expansion_coefficient=Thermal_Expansion[i],
                thermal_conductivity=Thermal_Conductivity[i]
            )
            for i in range(len(Youngs_Modulus))]
        fe_solver_structural.set_material(fe_solver_structural.mat_prop)
        fe_solver_thermal.mat_prop = [
            mat_lib.create_material_with_defaults(
                name=f"Material_{i+1}",
                youngs_modulus=Youngs_Modulus[i],
                thermal_expansion_coefficient=Thermal_Expansion[i],
                thermal_conductivity=Thermal_Conductivity[i]
            )
            for i in range(len(Thermal_Conductivity))]
        fe_solver_thermal.set_material(fe_solver_thermal.mat_prop)

        # Solve thermal problem
        temperature = fe_solver_thermal.solve(xDesign.detach().cpu().numpy(), MaterialModel.SIMP)
        fe_solver_thermal.postprocess()

        # Get thermoelastic force and solve structural problem
        thermo_elastic_force = fe_solver_thermal.get_thermoelastic_force(xDesign.detach().cpu().numpy(), MaterialModel.SIMP)
        fe_solver_structural.set_thermal_forces(thermo_elastic_force)
        displacement = fe_solver_structural.solve(xDesign.detach().cpu().numpy(), MaterialModel.SIMP)
        fe_solver_structural.mesh.setPseudoDensity(xDesign.detach().cpu().numpy())
        fe_solver_structural.postprocess()
        
        if (plot_progress):
           fe_solver_structural.plot_pseudo_density_realtime(
                   title=f"Iter {mmaIterations + 1}",
                   external_plotter=plotter  # Pass GUI plotter if available
               )
        # Compute sensitivities
        obj, grad_obj = compute_mmto_objective_and_gradient(
            to_params,
            displacement,           # sol (structural displacement)
            temperature,            # temperature field
            zeta,                   # design variable vector
            fe_solver_structural,   # structural solver
            KETemplate,             # template stiffness matrix
            KTTemplate,             # template thermal matrix
            matEncoder,             # material encoder
            fe_solver_thermal       # thermal solver
        )
        cons, grad_cons = compute_mmto_constraint_and_gradient(
            to_params, displacement, zeta, fe_solver_structural, KETemplate, matEncoder
        )

        if (obj0 is None):
            obj0 = obj
        
        if any(c > 0.5 for c in cons.flatten()):
            grad_obj *= 0

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
            print(f"Constraint {idx+1} ({constraint_names[idx]}): {(val+1)*to_params.Constraints[idx][2]:.3g} {inequality} {to_params.Constraints[idx][2]:.3g}?")

        history["objective"].append(obj)
        history["constraints"].append(cons.flatten().copy())
    
        if (use_penalization):
            d_ij = torch.cdist(zPoints, zRealPoints)
            min_i = torch.min(d_ij, dim=1).values
            min_i = min_i * torch.tensor(xDesign, dtype=torch.float32)
            nActiveElems = np.ceil(torch.sum((torch.tensor(xDesign).float())).item())
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

        print(f"Gamma value: {gamma:.5g}")
        print(f"Actual objective: {obj-penalty:.5g}")
        print(f"Penalizeation: {penalty:.5g}")
        mmaIterations += 1
        if (use_continuation) and (mmaIterations % 10 == 0):
            increment_SIMP_THERMAL_PENALTY(0.25)
            increment_SIMP_STRUCTURAL_PENALTY(0.5)
        return obj, grad_obj, cons, grad_cons

    # Initial design variable setup
    initialDensity = 0.5
    for constraint in to_params.Constraints:
        if constraint[0] == TO_QOI.VOLUME_FRACTION:
            initialDensity = constraint[2]
            break
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

    optResults = runMMA(nVariables, nConstraints, MMTO_TS_optimization_function, zeta0.reshape(-1, 1), lowerBound,
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
        MMTO_TS_optimization_function(zetaOptimal)
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
    material_indices = matEncoder.getClosestRealMaterialIndex(zOptimalPts)
    colors = [material_colors[int(idx.item()) if hasattr(idx, "item") else int(idx)] for idx in material_indices]
    rgb_colors = np.array([mcolors.to_rgb(c) for c in colors])
    fe_solver_structural.plot_elem_field(material_indices, title='Real Materials', colors=rgb_colors)

if __name__ == "__main__":
    # Example thermoelastic TO problem
    to_problem = MMTOThermostructuralExamples.MBBBeam

    run_topopt_TS(
        to_problem=to_problem,
        use_penalization=True,
        use_pretrained_vae=True,
        snap_to_real_material=False,
    )