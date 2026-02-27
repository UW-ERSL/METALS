import numpy as np
import torch
import time
import os
import sys
import matplotlib.colors as mcolors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../0-Common')))

from materialColors import material_colors # type: ignore[reportMissingImports]
from PyTOImports import (deflation,linear_solvers,hex_structural_fea, hex_element_stiffness,# type: ignore
                         createFilters,mat_lib,MaterialModel,TO_QOI,runMMA,
                         initialize_SIMP_STRUCTURAL_PENALTY,get_pNorm_exponent,initialize_SIMP_THERMAL_PENALTY,
                         increment_SIMP_STRUCTURAL_PENALTY,increment_SIMP_THERMAL_PENALTY,) # type: ignore) # type: ignore

from ExamplesPureStructural import MMTOExamplesPureStructural, getMMTOProblemPureStructural, material_colors
from materialEncoder import MaterialEncoder # type: ignore[reportMissingImports]
from SensitivitiesPureStructural import (compute_mmto_objective_and_gradient,compute_mmto_constraint_and_gradient,)
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
    use_continuation = True,
    use_constant_poissons_ratio: bool = True,
    constant_poissons_ratio: float = 0.3,
    snap_to_real_material=True,
    rel_conv_tol = 1e-7,
    maxIterations = 150,
    binarize_topology = True,
    z0_init_method = Z0InitMethod.ORIGIN,  
    gamma_init = 1e-4, # penalization
    gamma_max = 100,
    gamma_factor = 1.1,
    plotter = None):
    
    history = {
        "objective": [],
        "constraints": []
    }

    mesh_structural, mat_prop_struct, bc_struct,\
        elem_body_force, to_params, vae_params = getMMTOProblemPureStructural(to_problem)
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
    
    # Element stiffness template (only used in backwards-compatible constant-nu sensitivity path)
    nu_template = float(constant_poissons_ratio) if use_constant_poissons_ratio else float(mat_prop_struct.poissons_ratio)
    KETemplate = hex_element_stiffness.hex8_stiffness_matrix_structural(
            mat_prop_struct.youngs_modulus, nu_template, mesh_structural.elem_size)
    def eval_stress_compliance_metrics(x_vec, z_vec, label="", active_thresh=0.5):
        """
        Evaluate metrics on a given design (x_vec, z_vec):
          - p-norm von Mises stress (from fe_solver_structural.postprocess)
          - compliance
          - min/max von Mises stress on "active" elements (x > active_thresh)

        x_vec: (num_elems,)
        z_vec: (latentDim*num_elems,) flattened (same layout as zetaOptimal[num_elems:])
        """
        x_vec = np.asarray(x_vec, dtype=float).flatten()
        z_vec = np.asarray(z_vec, dtype=float).flatten()

        # --- Decode material properties from z ---
        zPts = torch.tensor(z_vec).view(latentDim, -1).T.float()  # (num_elems, latentDim)
        with torch.no_grad():
            decoded = matEncoder.vaeNet.decoder(zPts)
            material_properties = matEncoder.getMaterialProperties(decoded)
            Youngs_Modulus = material_properties['Youngs_Modulus'].detach().cpu().numpy()

        # --- Set elementwise materials (SIMP uses E per element in your setup) ---
        fe_solver_structural.mat_prop = [
            mat_lib.create_material_with_defaults(name=f"Material_{i+1}", youngs_modulus=float(Youngs_Modulus[i]))
            for i in range(len(Youngs_Modulus))
        ]
        fe_solver_structural.set_material(fe_solver_structural.mat_prop)

        # --- Solve + postprocess ---
        sol_local = fe_solver_structural.solve(x_vec, MaterialModel.SIMP)
        fe_solver_structural.mesh.setPseudoDensity(x_vec)
        fe_solver_structural.postprocess()

        vm = np.asarray(fe_solver_structural.vonMisesStress, dtype=float)  # (num_elems,)
        pnorm_vm = float(fe_solver_structural.pNormStress)                 # already uses your relaxation/modeling
        compliance = float(np.einsum('i,i->', fe_solver_structural.total_force, sol_local))

        active = x_vec > active_thresh
        if np.any(active):
            vm_min_active = float(np.min(vm[active]))
            vm_max_active = float(np.max(vm[active]))
        else:
            vm_min_active = float("nan")
            vm_max_active = float("nan")

        print("\n" + "="*70)
        print(f"[FINAL METRICS] {label}")
        print(f"  p-norm(vonMises)        : {pnorm_vm:.6g}")
        print(f"  Compliance              : {compliance:.6g}")
        print(f"  min(vonMises) active    : {vm_min_active:.6g}   (x > {active_thresh})")
        print(f"  max(vonMises) active    : {vm_max_active:.6g}   (x > {active_thresh})")
        print("="*70 + "\n")

        return {
            "pnorm_vm": pnorm_vm,
            "compliance": compliance,
            "vm_min_active": vm_min_active,
            "vm_max_active": vm_max_active,
        }

    num_dof = fe_solver_structural.bc.num_dofs
    print(f"Number of DOF: {num_dof}")
    num_elems = mesh_structural.num_elems
    # --- Generalize latent dimension ---
    latentDim = matEncoder.vae_params.latentDim
    num_design_var = num_elems + num_elems * latentDim
    print(f"Using latent dimension: {latentDim}")

####
    def plot_stress_ff_violations(x_vec, z_vec, label="", active_thresh=0.5):
        x_vec = np.asarray(x_vec, dtype=float).reshape(-1)
        z_vec = np.asarray(z_vec, dtype=float).reshape(-1)

        zPts = torch.tensor(z_vec).view(latentDim, -1).T.float()
        with torch.no_grad():
            decoded = matEncoder.vaeNet.decoder(zPts)
            props = matEncoder.getMaterialProperties(decoded)

        vm = np.asarray(fe_solver_structural.vonMisesStress, dtype=float).reshape(-1)

        # get yield strength
        ys_key = None
        for k in ("Yield_Strength", "YieldStrength", "yield_strength", "sigma_y", "Sy"):
            if k in props:
                ys_key = k
                break
        if ys_key is None:
            raise KeyError("Yield nt found.")
        Sy = props[ys_key].detach().cpu().numpy().astype(float).reshape(-1)

        ff = vm / np.maximum(Sy, 1e-12)

        solid = x_vec > active_thresh
        viol = (ff > 1.0) & solid # violating elements

        # voxel che
        print("\n" + "="*60)
        print(f"[STRESS FF VOXEL CHECK] {label}")
        print(f"Solid voxels     : {int(np.sum(solid))}")
        print(f"Violating voxels : {int(np.sum(viol))}")
        print(f"Max FF (solid)   : {float(np.max(ff[solid])):.6g}")
        print("="*60 + "\n")

        field = np.full_like(x_vec, np.nan, dtype=float)

        field[solid] = 1.0
        field[viol] = 0.0

        fe_solver_structural.plot_elem_field(
            field,
            title=f"Stress FF (1=OK, 0=VIOL) | {label}",
            colormap="RdYlGn"
        )
####



    #fe_solver_structural.plot_mesh(plot_bc=True, offsetArrow=True)  
    # Create the filter for density and material variables
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

    def MMTO_optimization_function(zeta):
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
        Youngs_Modulus = material_properties['Youngs_Modulus'].detach().numpy()

        # Decode Poisson's ratio per element if requested (otherwise keep constant for backwards-compatibility)
        nu_key = None
        for k in ("Poissons_Ratio", "Poisson_Ratio", "poissons_ratio", "poisson_ratio", "nu", "Nu"):
            if k in material_properties:
                nu_key = k
                break

        if use_constant_poissons_ratio or (nu_key is None):
            nu_vals = np.full_like(Youngs_Modulus, float(constant_poissons_ratio), dtype=float)
        else:
            nu_vals = material_properties[nu_key].detach().numpy().astype(float)

        # Build per-element materials (E and nu can vary per element)
        fe_solver_structural.mat_prop = []
        for i in range(len(Youngs_Modulus)):
            mp = mat_lib.create_material_with_defaults(name=f"Material_{i+1}", youngs_modulus=float(Youngs_Modulus[i]))
            # Set Poisson's ratio explicitly (avoid relying on library defaults)
            if hasattr(mp, "poissons_ratio"):
                mp.poissons_ratio = float(nu_vals[i])
            elif hasattr(mp, "nu"):
                mp.nu = float(nu_vals[i])
            fe_solver_structural.mat_prop.append(mp)
        fe_solver_structural.set_material(fe_solver_structural.mat_prop)
        # ----------------------------
        # DEBUG: check Poisson ratio spread actually used by FEA
        # ----------------------------
        try:
            nus = []
            for mp in fe_solver_structural.mat_prop:
                if hasattr(mp, "poissons_ratio"):
                    nus.append(float(mp.poissons_ratio))
                elif hasattr(mp, "nu"):
                    nus.append(float(mp.nu))
            if len(nus) > 0:
                print(f"[FEA nu check] min={min(nus):.6g}, max={max(nus):.6g}, "
                    f"use_constant={use_constant_poissons_ratio}, const_nu={constant_poissons_ratio}")
            else:
                print("[FEA nu check] Could not read nu from mat_prop objects.")
        except Exception as e:
            print(f"[FEA nu check] ERROR: {e}")

        sol = fe_solver_structural.solve(xDesign.detach().cpu().numpy(), MaterialModel.SIMP)
        fe_solver_structural.mesh.setPseudoDensity(xDesign.detach().cpu().numpy())
        fe_solver_structural.postprocess()

        if (plot_progress):
           fe_solver_structural.plot_pseudo_density_realtime(
                   title=f"Iter {mmaIterations + 1}",
                   external_plotter=plotter  # Pass GUI plotter if available
               )


        obj, grad_obj = compute_mmto_objective_and_gradient(
            to_params, sol, zeta, fe_solver_structural, KETemplate, matEncoder,
            use_constant_poissons_ratio=use_constant_poissons_ratio,
            constant_poissons_ratio=constant_poissons_ratio,
        )

        cons, grad_cons = compute_mmto_constraint_and_gradient(
            to_params, sol, zeta, fe_solver_structural, KETemplate, matEncoder,
            use_constant_poissons_ratio=use_constant_poissons_ratio,
            constant_poissons_ratio=constant_poissons_ratio,
        )


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

        mmaIterations += 1
        if (use_continuation) and (mmaIterations % 10 == 0):
            increment_SIMP_THERMAL_PENALTY(0.25)
            increment_SIMP_STRUCTURAL_PENALTY(0.25)
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
    # ---- Evaluate final metrics on continuous design (no binarization, no snapping)
    eval_stress_compliance_metrics(xOptimal, zOptimal, label="Continuous (pre-binarize, pre-snap)", active_thresh=0.5)

    if (binarize_topology):
        x_sorted = np.sort(xOptimal)
        threshold = x_sorted[int((1-np.mean(xOptimal))*len(xOptimal))]
        xOptimal = np.where(xOptimal < threshold, 0.0, 1.0)
    
        # IMPORTANT: keep zetaOptimal consistent with binarized xOptimal
        zetaOptimal[0:num_elems] = xOptimal

        # ---- Evaluate final metrics on binarized topology
        eval_stress_compliance_metrics(xOptimal, zOptimal, label="Binarized topology (post-binarize, pre-snap)", active_thresh=0.5)
    if (snap_to_real_material):
        zSnappedPts = torch.tensor(matEncoder.getClosestRealMaterialZValues(zOptimalPts))
        zetaOptimal[num_elems:] = zSnappedPts.T.flatten().numpy()

        # ---- Evaluate final metrics after snapping materials
        eval_stress_compliance_metrics(
            xOptimal,
            zetaOptimal[num_elems:],   # snapped z
            label="Binarized + snapped materials (post-snap)",
            active_thresh=0.5
        )

        print(50 * "-")
        print("After snapping:")
        print(50 * "-")
        MMTO_optimization_function(zetaOptimal)
##
        plot_stress_ff_violations(
            xOptimal,
            zetaOptimal[num_elems:],  # snapped z
            label="Binarized + snapped (final)",
            active_thresh=0.5
        )
##

        zOptimalPts = zSnappedPts
    # ----------------- Per-material volume fractions (final design) -----------------
    def _stable_softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
        m = np.max(logits, axis=axis, keepdims=True)
        ex = np.exp(logits - m)
        return ex / np.sum(ex, axis=axis, keepdims=True)

    def print_material_volfracs_final(x_vec, zPts_torch, to_params, matEncoder):
        """
        Computes Vk = (1/N) * sum_e x_e * p_{e,k},
        where p_{e,k} = softmax( -||z_e - z_k||^2 / tau ) over real-material latent points z_k.
        Matches your constraint definition in SensitivitiesPureStructural.py.
        """
        x = np.asarray(x_vec, dtype=float).reshape(-1)
        zPts = zPts_torch.detach().cpu().numpy()  # (N, latentDim)
        N = x.size

        # Real-material latent points zRealPoints: (M, latentDim)
        with torch.no_grad():
            zRealPoints = matEncoder.vaeNet.encoder(matEncoder.scaledMaterialData).detach().cpu().numpy()

        # Gather material-specific VF constraints from to_params.Constraints
        mat_constraints = []  # (material_id, tau, upper)
        for (qoi, params, upper) in to_params.Constraints:
            if qoi == TO_QOI.VOLUME_FRACTION and isinstance(params, dict) and ("material_id" in params):
                material_id = int(params["material_id"])
                tau = float(params.get("tau", 1.0))
                mat_constraints.append((material_id, tau, float(upper)))

        if len(mat_constraints) == 0:
            print("\n[Per-material VF] No material-specific volume fraction constraints found.\n")
            return

        # Compute once per distinct tau (so we don't redo softmax work unnecessarily)
        by_tau = {}
        for material_id, tau, upper in mat_constraints:
            by_tau.setdefault(tau, []).append((material_id, upper))

        print("\n" + "=" * 70)
        print("[FINAL] Per-material volume fractions Vk = mean(x * p_k)")
        print(f"Total mean(x) = {np.mean(x):.6g}")
        print("=" * 70)

        # Optional: names if available
        names = getattr(matEncoder, "materialNames", None)

        for tau, mats in by_tau.items():
            # logits_{e,k} = -||z_e - z_k||^2 / tau
            diff = zPts[:, None, :] - zRealPoints[None, :, :]          # (N, M, latentDim)
            dist2 = np.einsum("nmd,nmd->nm", diff, diff)               # (N, M)
            logits = -dist2 / tau
            p = _stable_softmax(logits, axis=1)                        # (N, M)

            for material_id, upper in mats:
                pk = p[:, material_id]
                Vk = float(np.sum(x * pk) / N)
                c = Vk / upper - 1.0

                mat_name = f"Material_{material_id}"
                if names is not None and 0 <= material_id < len(names):
                    mat_name = f"{names[material_id]} (id={material_id})"

                print(f"tau={tau:g} | {mat_name:>24s} | Vk={Vk:.6g} | upper={upper:.6g} | c=Vk/upper-1={c:.6g}")

        print("=" * 70 + "\n")

    # Call it on the FINAL xOptimal / zOptimalPts (post-binarize, post-snap if enabled)
    print_material_volfracs_final(xOptimal, zOptimalPts, to_params, matEncoder)
  
    decoded = matEncoder.vaeNet.decoder(zOptimalPts)
    material_properties = matEncoder.getMaterialProperties(decoded)
    Youngs_Modulus = material_properties['Youngs_Modulus'].detach().cpu().numpy()
    
    material_indices = matEncoder.getClosestRealMaterialIndex(zOptimalPts)  # shape: (num_elems,)
    colors = [material_colors[int(idx.item()) if hasattr(idx, "item") else int(idx)] for idx in material_indices]
    
    rgb_colors = np.array([mcolors.to_rgb(c) for c in colors])
    material_names = [matEncoder.materialNames[i] for i in range(len(matEncoder.materialNames))]

    # ----------------- HARD per-material volume fractions (counting snapped material IDs) -----------------
    # Exact counting is meaningful when snap_to_real_material=True (each element corresponds to a real material).
    if snap_to_real_material:
        x = np.asarray(xOptimal, dtype=float).reshape(-1)
        N = x.size
        solid_mask = (x > 0.5)  # for binarized x: same as (x==1)

        # Convert material_indices to plain int array
        if hasattr(material_indices, "detach"):
            idx = material_indices.detach().cpu().numpy().astype(int).reshape(-1)
        else:
            idx = np.asarray(material_indices, dtype=int).reshape(-1)

        total_solid = int(np.sum(solid_mask))

        print("\n" + "=" * 70)
        print("[FINAL] HARD per-material volume fractions (snapped materials, counted)")
        print(f"Total mean(x)       = {float(np.mean(x)):.6g}")
        print(f"Solid elems (x>0.5) = {total_solid} / {N}")
        print("=" * 70)

        for k, name in enumerate(material_names):
            cnt_k = int(np.sum((idx == k) & solid_mask))
            vf_k = cnt_k / N  # = exact VF if x is binarized
            share_solids = (cnt_k / total_solid) if total_solid > 0 else float("nan")

            print(f"{name:>24s} | solid_elems={cnt_k:6d} | Vk_exact={vf_k:.6g} | share_of_solids={share_solids:.6g}")

        print("=" * 70 + "\n")

    # Plot material distribution with legend
    fe_solver_structural.plot_material_distribution(
        material_indices=material_indices.cpu().numpy() if hasattr(material_indices, 'cpu') else material_indices,
        material_names=material_names,
        material_colors=material_colors,
        title='Material Distribution',
        show_legend=True
    )


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
    # Create a legend mapping material names to colors
    
# if __name__ == "__main__":

#     to_problem = MMTOExamplesPureStructural.LBracketTopLoad_Stress_VolumeFraction_Mass


#     run_topopt(
#         to_problem=to_problem,
#         use_penalization=True,
#         use_pretrained_vae=True,
#         snap_to_real_material=False,
#     )

if __name__ == "__main__":
    to_problem = MMTOExamplesPureStructural.LBracketTopLoad_Mass_StressFF_BM2

    run_topopt(
        to_problem=to_problem,
        use_penalization=True,
        use_pretrained_vae=True,
        snap_to_real_material=True,
        use_constant_poissons_ratio=True,
        constant_poissons_ratio=0.3,
        use_continuation=True,
    )
