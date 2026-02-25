import numpy as np
import torch
import time
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../0-Common')))

from ExamplesThermoStructural import MMTOThermostructuralExamples, getMMTOThermostructuralProblem
from materialEncoder import MaterialEncoder
from materialColors import material_colors

# IMPORTANT: use the temp-dependent sensitivities module you will create
from SensitivitiesThermoStructural_TempDep import (
    compute_mmto_objective_and_gradient_tempdep,
    compute_mmto_constraint_and_gradient,  # volume/mass constraints can stay same for now
)

from PyTOImports import (
    deflation, linear_solvers, hex_structural_fea, hex_element_stiffness,
    hex_thermal_fea, createFilters, mat_lib, MaterialModel, TO_QOI, runMMA,
    initialize_SIMP_STRUCTURAL_PENALTY, initialize_SIMP_THERMAL_PENALTY,
    increment_SIMP_STRUCTURAL_PENALTY, increment_SIMP_THERMAL_PENALTY,
)

from enum import Enum
class Z0InitMethod(Enum):
    LIGHTEST = 'lightest'
    HEAVIEST = 'heaviest'
    ORIGIN = 'origin'
    UNIFORM = 'uniform'


def run_topopt_ThermoStructural_TempDep(
    to_problem,
    timeLimit=10*60*60,
    saveNet=None,
    plot_progress=True,
    use_pretrained_vae=False,
    use_penalization=False,
    snap_to_real_material=True,
    rel_conv_tol=1e-7,
    maxIterations=175,
    binarize_topology=False,
    z0_init_method=Z0InitMethod.ORIGIN,
    use_continuation=True,
    gamma_init=1e-4,
    gamma_max=1,
    gamma_factor=1.25,
    plotter=None,
    # temperature normalization for bezier/logbezier
    Ta=23.0,
    Tf=823.0,
    # Picard iterations for k(T) thermal solve
    maxThermalPicard=3,
    thermalPicardTol=1e-6,
):
    history = {"objective": [], "constraints": []}

    mesh, mat_prop, structural_bc, thermal_bc, elem_body_force, to_params, vae_params = \
        getMMTOThermostructuralProblem(to_problem)

    if to_params.MaterialsExcelFile is None:
        print("Please provide a valid MaterialsExcelFile in to_params.")
        return

    matEncoder = MaterialEncoder(vae_params)
    matEncoder.readExcel(to_params.MaterialsExcelFile)


    matEncoder.plotTemperatureVsMaterialPropertyRaw("E", semilogy=True, colors=material_colors)
    matEncoder.plotTemperatureVsMaterialPropertyRaw("Y", semilogy=True, colors=material_colors)
    matEncoder.plotTemperatureVsMaterialPropertyRaw("K", semilogy=True, colors=material_colors)
    matEncoder.plotTemperatureVsMaterialPropertyRaw("Alpha", semilogy=True, colors=material_colors)
    if saveNet is None:
        base, _ = os.path.splitext(to_params.MaterialsExcelFile)
        saveNet = base + ".nt"
    vae_file_exists = os.path.exists(saveNet) and os.path.getsize(saveNet) > 0
    print("Materials loaded:", matEncoder.materialNames)
    
    if use_pretrained_vae and vae_file_exists:
        print(f"Loading pre-trained autoencoder from file: {saveNet}")
        matEncoder.loadAutoencoderFromFile(saveNet)
    else:
        print(f"Training autoencoder and saving to: {saveNet}")
        t0 = time.time()
        matEncoder.trainAutoencoder(
            vae_params.numEpochs, vae_params.klFactor, saveNet,
            vae_params.learningRate, vae_params.maxAttributeErrorPercent
        )
        print(f"Autoencoder training time: {time.time() - t0:.2f} seconds")

    with torch.no_grad():
        matEncoder.training_latents = matEncoder.vaeNet.encoder(matEncoder.scaledMaterialData).cpu()
    matEncoder.printEncodingErrors()
    zRealPoints = matEncoder.training_latents

    solver = linear_solvers.Solvers.PARDISO
    dsolver = deflation.DeflationSolver()

    fe_solver_structural = hex_structural_fea.HexStructuralFEA(
        mesh=mesh, mat_prop=mat_prop, bc=structural_bc, solver=solver, dsolver=dsolver, rtol=1e-8,
        elem_body_force=elem_body_force
    )
    fe_solver_thermal = hex_thermal_fea.HexThermalFEA(
        mesh=mesh, mat_prop=mat_prop, bc=thermal_bc, solver=solver, dsolver=dsolver, rtol=1e-8
    )

    # Unit templates (E=1, k=1)
    KETemplate_unitE = hex_element_stiffness.hex8_stiffness_matrix_structural(1.0, 0.3, mesh.elem_size)
    KTTemplate_unitK = hex_element_stiffness.hex8_stiffness_matrix_thermal(1.0, mesh.elem_size)

    num_elems = mesh.num_elems
    latentDim = vae_params.latentDim
    num_design_var = num_elems + num_elems * latentDim

    print(f"Using latent dimension: {latentDim}")
    print("Creating filter...")
    H, Hs = createFilters(fe_solver_structural, to_params)

    mmaIterations = 0
    obj0 = None
    gamma = gamma_init

    if use_continuation:
        initialize_SIMP_STRUCTURAL_PENALTY(1.5)
        initialize_SIMP_THERMAL_PENALTY(1.0)
    else:
        initialize_SIMP_STRUCTURAL_PENALTY(3.0)
        initialize_SIMP_THERMAL_PENALTY(1.0)

    def MMTO_TS_TempDep_optimization_function(zeta):
        nonlocal mmaIterations, obj0, gamma, zRealPoints

        zeta = np.asarray(zeta).flatten()
        print("-------------- Iteration", mmaIterations, "-----------------")

        zetaTensor = torch.tensor(zeta, dtype=torch.float32, requires_grad=True)
        xDesign = zetaTensor[0:num_elems]
        zDesign = zetaTensor[num_elems:]
        zPoints = zDesign.view(latentDim, -1).T  # (nelem, latentDim)

        xNumpy = xDesign.detach().cpu().numpy()
        grey_elements = np.sum((xNumpy > 0.1) & (xNumpy < 0.9))
        print(f"Percentage grey elements: {100.0*grey_elements/num_elems:.2f}%")

        # ---------------------------------------------------------------------
        # TEMP-DEPENDENT PROPERTY EVALUATION NEEDS element temperatures.
        # We solve thermal with k(T) by Picard iteration:
        #   - start with k at T=Ta (uniform) on first Picard step
        #   - update k(T_elem) and re-solve thermal a few times
        # ---------------------------------------------------------------------

        # helper: compute element average temperature
        def compute_Telem(T_nodes):
            T_e = T_nodes[mesh.edofMatThermal]        # (nelem, 8)
            return T_e.mean(axis=1)                   # (nelem,)

        # Initial guess for Telem = Ta
        Telem = Ta * np.ones(num_elems, dtype=float)

        temperature = None
        for pic in range(maxThermalPicard):
            # Evaluate kappa at current Telem
            with torch.no_grad():
                kappa = matEncoder.getMaterialPropertyAtTemperatureTorch("K", zPoints, torch.tensor(Telem, dtype=torch.float32))
            kappa = kappa.detach().cpu().numpy()

            # Set thermal materials with this kappa (E/alpha irrelevant to thermal solve)
            fe_solver_thermal.mat_prop = [
                mat_lib.create_material_with_defaults(
                    name=f"Material_{i+1}",
                    thermal_conductivity=float(kappa[i]),
                    youngs_modulus=1.0,
                    thermal_expansion_coefficient=0.0
                )
                for i in range(num_elems)
            ]
            fe_solver_thermal.set_material(fe_solver_thermal.mat_prop)

            # Solve thermal
            temperature = fe_solver_thermal.solve(xNumpy, MaterialModel.SIMP)
            fe_solver_thermal.postprocess()

            Telem_new = compute_Telem(temperature)
            rel = np.linalg.norm(Telem_new - Telem) / max(1e-12, np.linalg.norm(Telem))
            Telem = Telem_new
            if rel < thermalPicardTol:
                break

        # Now compute E(T), Alpha(T), K(T) at converged Telem
        Telem_torch = torch.tensor(Telem, dtype=torch.float32)

        with torch.no_grad():
            E = matEncoder.getMaterialPropertyAtTemperatureTorch("E", zPoints, Telem_torch).detach().cpu().numpy()
            alpha = matEncoder.getMaterialPropertyAtTemperatureTorch("Alpha", zPoints, Telem_torch).detach().cpu().numpy()
            kappa = matEncoder.getMaterialPropertyAtTemperatureTorch("K", zPoints, Telem_torch).detach().cpu().numpy()

        # Set materials for structural and thermal solvers consistently
        fe_solver_structural.mat_prop = [
            mat_lib.create_material_with_defaults(
                name=f"Material_{i+1}",
                youngs_modulus=float(E[i]),
                thermal_expansion_coefficient=float(alpha[i]),
                thermal_conductivity=float(kappa[i]),
            )
            for i in range(num_elems)
        ]
        fe_solver_structural.set_material(fe_solver_structural.mat_prop)

        fe_solver_thermal.mat_prop = [
            mat_lib.create_material_with_defaults(
                name=f"Material_{i+1}",
                youngs_modulus=float(E[i]),
                thermal_expansion_coefficient=float(alpha[i]),
                thermal_conductivity=float(kappa[i]),
            )
            for i in range(num_elems)
        ]
        fe_solver_thermal.set_material(fe_solver_thermal.mat_prop)

        # Thermoelastic coupling and structural solve
        thermo_elastic_force = fe_solver_thermal.get_thermoelastic_force(xNumpy, MaterialModel.SIMP)
        fe_solver_structural.set_thermal_forces(thermo_elastic_force)

        displacement = fe_solver_structural.solve(xNumpy, MaterialModel.SIMP)
        fe_solver_structural.mesh.setPseudoDensity(xNumpy)
        fe_solver_structural.postprocess()

        if plot_progress:
            fe_solver_structural.plot_pseudo_density_realtime(
                title=f"Iter {mmaIterations + 1}",
                external_plotter=plotter
            )

        # Compute objective + gradients using TEMP-DEPENDENT sensitivity module
        obj, grad_obj = compute_mmto_objective_and_gradient_tempdep(
            to_params=to_params,
            displacement=displacement,
            temperature=temperature,
            Telem=Telem,
            zeta=zeta,
            fe_solver_structural=fe_solver_structural,
            fe_solver_thermal=fe_solver_thermal,
            KETemplate_unitE=KETemplate_unitE,
            KTTemplate_unitK=KTTemplate_unitK,
            matEncoder=matEncoder
        )

        # Constraints unchanged (volume/mass)
        cons, grad_cons = compute_mmto_constraint_and_gradient(
            to_params, displacement, zeta, fe_solver_structural, KETemplate_unitE, matEncoder
        )

        if obj0 is None:
            obj0 = obj

        if any(c > 0.5 for c in cons.flatten()):
            grad_obj *= 0

        obj = obj / obj0
        grad_obj = grad_obj / obj0

        if to_params.ElemsToKeep is not None:
            grad_obj[to_params.ElemsToKeep] = min(grad_obj)

        # Filter
        grad_obj[0:num_elems] = (H * grad_obj[0:num_elems]) / Hs
        for i in range(latentDim):
            s = num_elems + i*num_elems
            grad_obj[s:s+num_elems] = (H * grad_obj[s:s+num_elems]) / Hs

        for i in range(grad_cons.shape[0]):
            grad_cons[i, 0:num_elems] = (H * grad_cons[i, 0:num_elems]) / Hs
            for j in range(latentDim):
                s = num_elems + j*num_elems
                grad_cons[i, s:s+num_elems] = (H * grad_cons[i, s:s+num_elems]) / Hs

        grad_obj = np.array([grad_obj]).reshape((num_design_var, 1))
        cons = np.array(cons).reshape((-1, 1))
        grad_cons = np.array(grad_cons).reshape((len(cons), num_design_var))

        print(f"Min. Objective: {obj*obj0:.5g}")
        history["objective"].append(obj)
        history["constraints"].append(cons.flatten().copy())

        # continuation
        mmaIterations += 1
        if use_continuation and (mmaIterations % 10 == 0):
            increment_SIMP_THERMAL_PENALTY(0.25)
            increment_SIMP_STRUCTURAL_PENALTY(0.25)

        return obj, grad_obj, cons, grad_cons

    # -------------------- Initial design --------------------
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

    tStart = time.time()
    optResults = runMMA(
        num_design_var,
        len(to_params.Constraints),
        MMTO_TS_TempDep_optimization_function,
        zeta0.reshape(-1, 1),
        lowerBound,
        upperBound,
        maxIterations=maxIterations,
        timeLimitSecs=timeLimit,
        move_limit=0.05,
        kktTol=1e-6,
        fTolerance=rel_conv_tol,
        gTolerance=rel_conv_tol,
        verbose=False
    )
    zetaOptimal = optResults[0]
    print(f"Total optimization time: {time.time() - tStart:.2f} seconds")

    # -------------------------
    # Post-processing (Temp-Dependent)
    # -------------------------
    zetaOptimal = np.asarray(zetaOptimal).flatten()
    xOptimal = zetaOptimal[0:num_elems].copy()
    zOptimal = zetaOptimal[num_elems:].copy()
    zOptimalPts = torch.tensor(zOptimal).view(latentDim, -1).T.float()

    fe_solver_structural.mesh.setPseudoDensity(xOptimal)

    # Optional: binarize
    if binarize_topology:
        x_sorted = np.sort(xOptimal)
        threshold = x_sorted[int((1 - np.mean(xOptimal)) * len(xOptimal))]
        xOptimal = np.where(xOptimal < threshold, 0.0, 1.0)

    # Optional: snap materials
    if snap_to_real_material:
        zSnappedPts = torch.tensor(matEncoder.getClosestRealMaterialZValues(zOptimalPts))
        zetaOptimal[num_elems:] = zSnappedPts.T.flatten().numpy()
        print(50 * "-")
        print("After snapping:")
        print(50 * "-")

        # Re-evaluate once for reporting and final plots
        _ = MMTO_TS_TempDep_optimization_function(zetaOptimal)  # if your tempdep function name differs, change it
        zOptimalPts = zSnappedPts
        zOptimal = zSnappedPts.T.flatten().numpy()
        xOptimal = zetaOptimal[0:num_elems].copy()

    # ---- Recompute final forward solution (Temp-Dependent) for plotting ----
    def compute_Telem_from_Tnodes(T_nodes):
        T_e = T_nodes[mesh.edofMatThermal]
        return T_e.mean(axis=1)

    # Picard thermal solve with K(Telem)
    Ta_plot = 23.0  # keep consistent with your problem definition
    maxThermalPicard_plot = 3
    thermalPicardTol_plot = 1e-6

    Telem = Ta_plot * np.ones(num_elems, dtype=float)
    temperature = None

    for pic in range(maxThermalPicard_plot):
        with torch.no_grad():
            K_T = matEncoder.getMaterialPropertyAtTemperatureTorch(
                "K", zOptimalPts, torch.tensor(Telem, dtype=torch.float32)
            ).detach().cpu().numpy()

        # Thermal solve uses K only
        fe_solver_thermal.mat_prop = [
            mat_lib.create_material_with_defaults(
                name=f"Material_{i+1}",
                thermal_conductivity=float(K_T[i]),
                youngs_modulus=1.0,
                thermal_expansion_coefficient=0.0
            )
            for i in range(num_elems)
        ]
        fe_solver_thermal.set_material(fe_solver_thermal.mat_prop)

        temperature = fe_solver_thermal.solve(xOptimal, MaterialModel.SIMP)
        fe_solver_thermal.postprocess()

        Telem_new = compute_Telem_from_Tnodes(temperature)
        rel = np.linalg.norm(Telem_new - Telem) / max(1e-12, np.linalg.norm(Telem))
        Telem = Telem_new
        if rel < thermalPicardTol_plot:
            break

    # Final E(Telem), Alpha(Telem), K(Telem) for plotting and thermoelastic solve
    Telem_torch = torch.tensor(Telem, dtype=torch.float32)
    with torch.no_grad():
        E_T = matEncoder.getMaterialPropertyAtTemperatureTorch("E", zOptimalPts, Telem_torch).detach().cpu().numpy()
        A_T = matEncoder.getMaterialPropertyAtTemperatureTorch("Alpha", zOptimalPts, Telem_torch).detach().cpu().numpy()
        K_T = matEncoder.getMaterialPropertyAtTemperatureTorch("K", zOptimalPts, Telem_torch).detach().cpu().numpy()

    # Set structural materials with final properties
    fe_solver_structural.mat_prop = [
        mat_lib.create_material_with_defaults(
            name=f"Material_{i+1}",
            youngs_modulus=float(E_T[i]),
            thermal_expansion_coefficient=float(A_T[i]),
            thermal_conductivity=float(K_T[i])
        )
        for i in range(num_elems)
    ]
    fe_solver_structural.set_material(fe_solver_structural.mat_prop)

    # Set thermal materials too (for thermoelastic force routine), then re-solve thermal ONCE for consistency
    fe_solver_thermal.mat_prop = [
        mat_lib.create_material_with_defaults(
            name=f"Material_{i+1}",
            youngs_modulus=float(E_T[i]),
            thermal_expansion_coefficient=float(A_T[i]),
            thermal_conductivity=float(K_T[i])
        )
        for i in range(num_elems)
    ]
    fe_solver_thermal.set_material(fe_solver_thermal.mat_prop)

    temperature = fe_solver_thermal.solve(xOptimal, MaterialModel.SIMP)
    fe_solver_thermal.postprocess()

    thermo_elastic_force = fe_solver_thermal.get_thermoelastic_force(xOptimal, MaterialModel.SIMP)
    fe_solver_structural.set_thermal_forces(thermo_elastic_force)

    displacement = fe_solver_structural.solve(xOptimal, MaterialModel.SIMP)
    fe_solver_structural.mesh.setPseudoDensity(xOptimal)
    fe_solver_structural.postprocess()

    # ---- Material distribution (same as old main) ----
    material_indices = matEncoder.getClosestRealMaterialIndex(zOptimalPts)  # (num_elems,)
    material_names = [matEncoder.materialNames[i] for i in range(len(matEncoder.materialNames))]

    fe_solver_structural.plot_material_distribution(
        material_indices=material_indices.cpu().numpy() if hasattr(material_indices, 'cpu') else material_indices,
        material_names=material_names,
        material_colors=material_colors,
        title='Material Distribution (Temp-Dep)',
        show_legend=True
    )

    # ---- Plot E(Telem) field (replaces 'Youngs_Modulus' plot) ----
    fe_solver_structural.plot_elem_field(E_T, title='Youngs Modulus E(Telem)', colormap='viridis')

    # ---- LSR plot ----
    matEncoder.plotLSR(zRealPoints.detach().cpu().numpy(), zOptimalPts, xDesign=xOptimal)

    # ---- Objective + constraints history plot (same as old main) ----
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

    if len(history["constraints"]) > 0:
        n_cons = len(history["constraints"][0])
        for i in range(n_cons):
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
    plt.title("Objective and Constraints vs. Iterations (Temp-Dep)")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    to_problem = MMTOThermostructuralExamples.MBBBeam
    run_topopt_ThermoStructural_TempDep(
        to_problem=to_problem,
        use_penalization=True,
        use_pretrained_vae=True,
        snap_to_real_material=False,
    )