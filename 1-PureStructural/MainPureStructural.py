"""
Structural MMTO driver: density topology + VAE-decoded materials, MMA optimizer.

PIPELINE:
    zeta = [x ; z]                         design vector (MMA variables)
      x  -> x_filt = (H_density @ x)/Hs    density filter
      x_filt -> x_phys = Heaviside(x_filt) optional projection (density only)
      z  -> decode (VAE) -> E, Y, mass_density  per-element material properties
    FEA solve(x_phys) -> von Mises stress (relaxed) -> objective/constraints
    SensitivitiesPureStructural computes obj/cons + gradients
    optional penalties (latent z-smoothing, distance-to-real-material) added here
    chain rule back through Heaviside + density filter -> gradient w.r.t. x
    MMA takes one step.

Latents z are raw MMA variables (no forward filter); their *gradients* are
optionally smoothed by _smooth_latent_gradient_inplace, gated by
APPLY_LATENT_SENS_FILTER (note: plain (H@g)/Hs smoothing, not Sigmund's
density-weighted form).

The only post-processing "extra" beyond plotting is the red/green local-stress
map (plot_stress_violation_distribution): green = FF<=1, red = FF>1 on active
elements, where FF = vonMises / decoded yield strength.
"""
import numpy as np
import torch
import time
import os
import sys
import matplotlib.pyplot as plt

from enum import Enum

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../0-Common")))

from materialColors import material_colors  # type: ignore[reportMissingImports]
from PyTOImports import (  # type: ignore
    deflation,
    linear_solvers,
    hex_structural_fea,
    hex_element_stiffness,
    createFilters,
    mat_lib,
    MaterialModel,
    TO_QOI,
    runMMA,
    initialize_SIMP_STRUCTURAL_PENALTY,
    initialize_SIMP_THERMAL_PENALTY,
    increment_SIMP_STRUCTURAL_PENALTY,
    increment_SIMP_THERMAL_PENALTY,
)
from ExamplesPureStructural import (
    MMTOExamplesPureStructural,
    getMMTOProblemPureStructural,
    material_colors,
)
from materialEncoder import MaterialEncoder  # type: ignore[reportMissingImports]
from SensitivitiesPureStructural import (
    compute_mmto_objective_and_gradient,
    compute_mmto_constraint_and_gradient,
)
from mmto_utility_functions import (
    run_basic_latent_space_diagnostics,
    create_density_and_material_filters,
    compute_domain_volume,
    init_run_history,
    start_run,
    log_design_state,
    record_fea_diagnostics,
    log_objective_and_constraints,
    run_postprocessing,
)
class Z0InitMethod(Enum):
    LIGHTEST = "lightest"
    HEAVIEST = "heaviest"
    ORIGIN = "origin"
    UNIFORM = "uniform"
class BetaScheduleScheme(Enum):
    PAPER = "paper"
def get_beta_schedule(scheme: BetaScheduleScheme):
    """Return the Heaviside beta-continuation parameters (PAPER scheme).

    Keys: beta_init/beta_max (range), beta_factor (growth per update),
    beta_update_every (iters between updates), beta_start_iter (first update).
    """
    if scheme == BetaScheduleScheme.PAPER:
        return {
            "beta_init": 1.0,
            "beta_max": 30.0,
            "beta_factor": 1.5,
            "beta_update_every": 5,
            "beta_start_iter": 50,
        }

    raise ValueError(f"Unknown beta schedule scheme: {scheme}")
    
def update_beta(beta_proj, mmaIterations, beta_schedule):
    """Decide whether to increase the Heaviside sharpness beta this iteration.

    Multiplies beta by beta_factor every beta_update_every iterations after
    beta_start_iter, capped at beta_max.

    Returns (beta_new, updated_flag).
    """
    if mmaIterations <= beta_schedule["beta_start_iter"]:
        return beta_proj, False

    if (mmaIterations % beta_schedule["beta_update_every"]) != 0:
        return beta_proj, False

    if beta_proj >= beta_schedule["beta_max"]:
        return beta_proj, False

    return min(beta_proj * beta_schedule["beta_factor"], beta_schedule["beta_max"]), True

def run_topopt(
    to_problem,
    timeLimit=10 * 60 * 60,
    saveNet=None,
    plot_progress=True,
    verbose=True,
    save_dir=None,
    # -------------------------------
    # VAE parameters
    # -------------------------------
    use_pretrained_vae=False,
    # -------------------------------
    # Misc. parameters
    # -------------------------------
    use_penalization=False,
    use_continuation=True,
    use_constant_poissons_ratio: bool = True,
    constant_poissons_ratio: float = 0.3,
    snap_to_real_material=True,
    # -------------------------------
    # Optimization parameters
    # -------------------------------    
    rel_conv_tol=1e-7,
    maxIterations=200,
    binarize_topology=True,
    z0_init_method=Z0InitMethod.ORIGIN,
    # -------------------------------
    # Plotting/diagnostic parameters
    # -------------------------------
    latent_space_diagnostics=True,
    # -------------------------------
    # Distance penalty parameters
    # -------------------------------
    gamma_init=1e-3,
    gamma_max=10,
    gamma_factor=1.05,
    penalty_start_iter=10,
    distance_penalty_disable_after_iter=100,
    plotter=None,
    # -------------------------------
    # Heaviside projection parameters
    # -------------------------------
    use_heaviside_projection=True,
    eta_proj=0.5,
    beta_schedule_scheme=BetaScheduleScheme.PAPER,
    # -------------------------------
    # z - H z / Hs smoothing penalty
    # -------------------------------
    use_z_smoothing=False,
    z_smoothing_weight=1e-3,
    z_smoothing_disable_after_iter=50,
):
    """
    DENSITY FILTER + HEAVISIDE PROJECTION (density ONLY):
      x (MMA var) --density filter--> x_filt = (H@x)/Hs --projection--> x_phys = H_{beta,eta}(x_filt)

    Latents:
      z are raw MMA variables; no direct filtering of z is performed.

    Optional latent smoothing penalty (density-weighted):
      Pz = lambda * sum_ell sum_e x_e * ( z_e^(ell) - ((H z^(ell))/Hs)_e )^2
    """

    history = init_run_history()
    mesh_structural, mat_prop_struct, bc_struct, elem_body_force, to_params, vae_params = (
        getMMTOProblemPureStructural(to_problem)
    )

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
        matEncoder.trainAutoencoder(
            vae_params.numEpochs,
            vae_params.klFactor,
            saveNet,
            vae_params.learningRate,
            vae_params.maxAttributeErrorPercent,
        )
        time_end = time.time()
        print(f"Autoencoder training time: {time_end - time_start:.2f} seconds")

    with torch.no_grad():
        matEncoder.training_latents = matEncoder.vaeNet.encoder(
            matEncoder.scaledMaterialData
        ).cpu()
    matEncoder.printEncodingErrors()

    zRealPoints = matEncoder.training_latents
    zRealPoints_np = zRealPoints.detach().cpu().numpy().astype(float)

    if latent_space_diagnostics:
        run_basic_latent_space_diagnostics(
            matEncoder=matEncoder,
            zRealPoints=zRealPoints,
            property_names=("Density", "Youngs_Modulus", "Yield_Strength"),
            n_interp=21,
            local_eps=0.05,
            random_seed=0,
            show_plots=True,
        )

    solver = linear_solvers.Solvers.PARDISO
    dsolver = deflation.DeflationSolver()

    fe_solver_structural = hex_structural_fea.HexStructuralFEA(
        mesh=mesh_structural,
        mat_prop=mat_prop_struct,
        bc=bc_struct,
        solver=solver,
        dsolver=dsolver,
        rtol=1e-8,
        elem_body_force=elem_body_force,
    )

    nu_template = (
        float(constant_poissons_ratio)
        if use_constant_poissons_ratio
        else float(mat_prop_struct.poissons_ratio)
    )
    KETemplate = hex_element_stiffness.hex8_stiffness_matrix_structural(
        mat_prop_struct.youngs_modulus,
        nu_template,
        mesh_structural.elem_size,
    )

    num_dof = fe_solver_structural.bc.num_dofs
    print(f"Number of DOF: {num_dof}")

    num_elems = mesh_structural.num_elems
    latentDim = matEncoder.vae_params.latentDim
    num_design_var = num_elems + num_elems * latentDim
    print(f"Using latent dimension: {latentDim}")
    elem_volume = float(np.prod(fe_solver_structural.mesh.elem_size))
    design_domain_volume = compute_domain_volume(fe_solver_structural)
    print("Creating filter...")
    H_density, Hs_density, H_material, Hs_material = create_density_and_material_filters(
        fe_solver_structural, to_params, createFilters
    )

    APPLY_DENSITY_FILTER = True
    APPLY_LATENT_SENS_FILTER = False  # heuristic (H_material@g)/Hs smoothing of latent
                                      # gradient blocks; set False to ablate.

    mmaIterations = 0
    obj0 = None
    gamma = float(gamma_init)
    beta_schedule = get_beta_schedule(beta_schedule_scheme)
    beta_proj = float(beta_schedule["beta_init"])

    if use_continuation:
        initialize_SIMP_STRUCTURAL_PENALTY(1.5)
        initialize_SIMP_THERMAL_PENALTY(1)
    else:
        initialize_SIMP_STRUCTURAL_PENALTY(3)
        initialize_SIMP_THERMAL_PENALTY(1)

    def heaviside_projection(x_filt: np.ndarray, beta: float, eta: float):
        """Smoothed Heaviside projection x_phys(x_filt) and derivative dx_phys/dx_filt.

        x_phys = (tanh(b*eta) + tanh(b*(x_filt-eta))) / (tanh(b*eta) + tanh(b*(1-eta))).
        Larger beta -> sharper (more 0/1) projection.
        """
        x_filt = np.asarray(x_filt, dtype=float).reshape(-1)
        denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        x = (np.tanh(beta * eta) + np.tanh(beta * (x_filt - eta))) / denom
        dxphys_dxfilt = (beta * (1.0 - np.tanh(beta * (x_filt - eta)) ** 2)) / denom
        return x, dxphys_dxfilt

    def _distance_to_real_material_penalty(
        density_for_penalty: np.ndarray,
        z_elemmajor: np.ndarray,
        gamma_val: float,
        eps: float = 1e-12,
    ):
        """Distance-to-nearest-real-material penalty and gradients.

        P = gamma_val * sum_e dens_e * dist(z_e, nearest real material) / nActive.
        Pulls each element latent toward the closest real-material latent point.
        Returns (penalty_value, dP/d(density), dP/d(z))  [z grad shape (N, latentDim)].
        """
        dens = np.asarray(density_for_penalty, dtype=float).reshape(-1)
        z = np.asarray(z_elemmajor, dtype=float)
        N = dens.size

        diff = z[:, None, :] - zRealPoints_np[None, :, :]
        dist2 = np.einsum("nml,nml->nm", diff, diff)
        dist = np.sqrt(np.maximum(dist2, eps))
        k_star = np.argmin(dist, axis=1)
        dmin = dist[np.arange(N), k_star]
        nActive = int(np.sum(dens > 0.1))
        nActive = max(nActive, 1)

        penalty_value = float(gamma_val * np.sum(dens * dmin) / nActive)

        dpen_d_dens = (gamma_val / nActive) * dmin

        z_k = zRealPoints_np[k_star, :]
        dpen_d_z = (gamma_val / nActive) * (
            dens[:, None] * (z - z_k) / np.maximum(dmin[:, None], eps)
        )

        return penalty_value, dpen_d_dens, dpen_d_z

    def _z_smoothing_penalty_and_grads(
        density_for_penalty: np.ndarray,
        z_elemmajor: np.ndarray,
        smooth_weight: float,
    ):
        """Density-weighted latent smoothing penalty
        Pz = w * sum_ell sum_e x_e * ( z - filtered(z) )^2.

        filtered = (H_material @ z)/Hs_material. Each element term is multiplied
        by its density so void regions are not smoothed.
        Returns (Pz, dPz/d(density), dPz/d(z)).
        """
        dens = np.asarray(density_for_penalty, dtype=float).reshape(-1)
        z = np.asarray(z_elemmajor, dtype=float)

        zbar = np.column_stack([(H_material @ z[:, ell]) / Hs_material for ell in range(latentDim)])
        r = z - zbar

        Pz = float(smooth_weight * np.sum(dens[:, None] * (r ** 2)))
        dP_dx = smooth_weight * np.sum(r ** 2, axis=1)

        dP_dz = np.zeros_like(z)
        for ell in range(latentDim):
            wr = dens * r[:, ell]
            dP_dz[:, ell] = 2.0 * smooth_weight * (
                wr - (H_material.T @ (wr / Hs_material))
            )

        return Pz, dP_dx, dP_dz

    def _smooth_latent_gradient_inplace(vec: np.ndarray):
        """Sensitivity-filter (in place) each latent block of an objective gradient.

        Applies (H_material @ g)/Hs_material to each per-latent-dim slice. This is a
        Sigmund-style regularizer on the latent gradient (z itself is not filtered).
        """
        for ell in range(latentDim):
            sl = slice(
                num_elems + ell * num_elems,
                num_elems + (ell + 1) * num_elems,
            )
            vec[sl] = (H_material @ vec[sl]) / Hs_material

    def _smooth_latent_constraint_gradient_inplace(mat: np.ndarray):
        """Same latent-block sensitivity filter as above, for the constraint Jacobian."""
        nCons = mat.shape[0]
        for i in range(nCons):
            for ell in range(latentDim):
                sl = slice(
                    num_elems + ell * num_elems,
                    num_elems + (ell + 1) * num_elems,
                )
                mat[i, sl] = (H_material @ mat[i, sl]) / Hs_material

    def MMTO_optimization_function(zeta):
        """MMA callback: returns (obj, grad_obj, cons, grad_cons) for design zeta.

        Steps:
          1. split zeta into density x and latents z
          2. density filter x -> x_filt, Heaviside x_filt -> x_phys (+ dx_phys/dx_filt)
          3. decode z -> E, nu, material props; assign per-element materials
          4. FEA solve(x_phys); record stress-violation diagnostics into history
          5. base obj/grad and cons/grad from SensitivitiesPureStructural
          6. normalize obj by obj0; add optional z-smoothing and distance penalties
          7. chain rule: latent-block sens filter, then Heaviside + density filter
          8. log, update beta continuation / SIMP continuation
        """
        nonlocal mmaIterations, obj0, gamma, beta_proj, use_z_smoothing, use_penalization
        if (
            use_z_smoothing
            and z_smoothing_disable_after_iter is not None
            and mmaIterations >= z_smoothing_disable_after_iter
        ):
            use_z_smoothing = False
            if verbose:
                print(f"[INFO] z-smoothing penalty disabled at iteration {mmaIterations}")
        # Disable distance-based penalty after specified iteration
        if (
            use_penalization
            and distance_penalty_disable_after_iter is not None
            and mmaIterations >= distance_penalty_disable_after_iter
        ):
            use_penalization = False
            if verbose:
                print(f"[INFO] Distance-based penalty disabled at iteration {mmaIterations}")
        zeta = np.asarray(zeta).flatten()

        # ===================== SPLIT DESIGN VECTOR =====================
        x = zeta[:num_elems].copy()
        z = zeta[num_elems:].reshape(latentDim, -1).T   # numpy, element-major (N, latentDim)

        # ============== DENSITY FILTER + HEAVISIDE PROJECTION ==========
        x_filt = (H_density @ x) / Hs_density if APPLY_DENSITY_FILTER else x.copy()

        if use_heaviside_projection:
            x_phys, dxphys_dxfilt = heaviside_projection(x_filt, beta_proj, eta_proj)
        else:
            x_phys = x_filt
            dxphys_dxfilt = np.ones_like(x_filt)

        # ===================== DECODE MATERIALS =======================
        zTorch = torch.tensor(z, dtype=torch.float32)   # torch twin of z, for the decoder

        with torch.no_grad():
            decoded = matEncoder.vaeNet.decoder(zTorch)
            material_properties = matEncoder.getMaterialProperties(decoded)
            Youngs_Modulus = material_properties["Youngs_Modulus"].detach().cpu().numpy()

        log_design_state(history, mmaIterations, x_phys, z, material_properties,
                         zTorch, matEncoder, num_elems, beta_proj, eta_proj, verbose=verbose)
        nu_key = None
        for k in (
            "Poissons_Ratio",
            "Poisson_Ratio",
            "poissons_ratio",
            "poisson_ratio",
            "nu",
            "Nu",
        ):
            if k in material_properties:
                nu_key = k
                break

        if use_constant_poissons_ratio or (nu_key is None):
            nu_vals = np.full_like(
                Youngs_Modulus,
                float(constant_poissons_ratio),
                dtype=float,
            )
        else:
            nu_vals = material_properties[nu_key].detach().cpu().numpy().astype(float)

        if use_z_smoothing:
            z_weight_eff = float(z_smoothing_weight) * float(obj0 if obj0 is not None else 1.0)
            Pz, dPz_dx, dPz_dz = _z_smoothing_penalty_and_grads(
                density_for_penalty=x_phys,
                z_elemmajor=z,
                smooth_weight=z_weight_eff,
            )
        else:
            Pz = 0.0
            dPz_dx = np.zeros(num_elems, dtype=float)
            dPz_dz = np.zeros((num_elems, latentDim), dtype=float)

        # ========================= MAIN FEA ===========================
        fe_solver_structural.mat_prop = []
        for i in range(len(Youngs_Modulus)):
            mp = mat_lib.create_material_with_defaults(
                name=f"Material_{i+1}",
                youngs_modulus=float(Youngs_Modulus[i]),
            )
            if hasattr(mp, "poissons_ratio"):
                mp.poissons_ratio = float(nu_vals[i])
            elif hasattr(mp, "nu"):
                mp.nu = float(nu_vals[i])
            fe_solver_structural.mat_prop.append(mp)

        fe_solver_structural.set_material(fe_solver_structural.mat_prop)

        sol = fe_solver_structural.solve(x_phys, MaterialModel.SIMP)
        fe_solver_structural.mesh.setPseudoDensity(x_phys)
        fe_solver_structural.postprocess()

        record_fea_diagnostics(history, mmaIterations, x_phys, z, material_properties,
                               fe_solver_structural, elem_volume, design_domain_volume,
                               plot_progress=plot_progress, plotter=plotter, verbose=verbose)

        # ============ OBJECTIVE / CONSTRAINTS / SENSITIVITIES =========
        zeta_phys = zeta.copy()
        zeta_phys[0:num_elems] = x_phys
        zeta_phys[num_elems:] = z.T.reshape(-1)
        to_params.current_major_iter = int(mmaIterations)
        obj, grad_obj = compute_mmto_objective_and_gradient(
            to_params,
            sol,
            zeta_phys,
            fe_solver_structural,
            KETemplate,
            matEncoder,
            use_constant_poissons_ratio=use_constant_poissons_ratio,
            constant_poissons_ratio=constant_poissons_ratio,
        )

        cons, grad_cons = compute_mmto_constraint_and_gradient(
            to_params,
            sol,
            zeta_phys,
            fe_solver_structural,
            KETemplate,
            matEncoder,
            use_constant_poissons_ratio=use_constant_poissons_ratio,
            constant_poissons_ratio=constant_poissons_ratio,
        )

        # ================= PENALTIES + NORMALIZATION ==================
        J_phys = float(obj)
        if obj0 is None:
            # For AL local-stress objective, normalize by mass alone
            # so that the frozen normalization constant does not
            # conflate the non-stationary AL penalty with the
            # physical mass objective.
            if to_params.Objective[0] == TO_QOI.MASS_AL_LOCAL_STRESS:
                s_al = to_params.AL_state
                elemVolume_tmp = float(np.prod(fe_solver_structural.mesh.elem_size))
                with torch.no_grad():
                    dec_tmp = matEncoder.vaeNet.decoder(zTorch)
                    props_tmp = matEncoder.getMaterialProperties(dec_tmp)
                    mass_density_tmp = props_tmp['Density'].detach().cpu().numpy()
                obj0 = float(np.sum(mass_density_tmp * x_phys) * elemVolume_tmp)
                obj0 = max(obj0, 1e-12)  # safety
            else:
                obj0 = J_phys

        Jn = J_phys / float(obj0)
        grad_obj = grad_obj / float(obj0)
        obj = Jn

        if to_params.ElemsToKeep is not None:
            grad_obj[to_params.ElemsToKeep] = np.min(grad_obj)

        grad_obj_phys = grad_obj.copy()
        if APPLY_LATENT_SENS_FILTER:
                    _smooth_latent_gradient_inplace(grad_obj_phys)
                    _smooth_latent_constraint_gradient_inplace(grad_cons)

        if use_z_smoothing:
            Pzn = Pz / float(obj0 if obj0 is not None else 1.0)
            obj = obj + Pzn
            grad_obj_phys[0:num_elems] += dPz_dx / float(obj0 if obj0 is not None else 1.0)
            grad_obj_phys[num_elems:] += dPz_dz.T.reshape(-1) / float(obj0 if obj0 is not None else 1.0)
        else:
            Pzn = 0.0

        P = 0.0
        Pn = 0.0

        if use_penalization and (mmaIterations >= int(penalty_start_iter)):
            gamma_eff = float(gamma) * float(obj0)

            penalty_value, dpen_d_x, dpen_d_z = _distance_to_real_material_penalty(
                x_phys,
                z,
                gamma_eff,
            )

            P = float(penalty_value)
            Pn = P / float(obj0)
            obj = obj + Pn
            grad_obj_phys[0:num_elems] += dpen_d_x / float(obj0)
            grad_obj_phys[num_elems:] += dpen_d_z.T.reshape(-1) / float(obj0)   # existing line

            gamma = min(gamma * gamma_factor, gamma_max)

        grad_obj = grad_obj_phys

        # ============== CHAIN RULE -> DESIGN VARIABLES ================
        if APPLY_DENSITY_FILTER:
            g_x = grad_obj[0:num_elems].copy()
            g_xfilt = g_x * dxphys_dxfilt
            grad_obj[0:num_elems] = H_density.T @ (g_xfilt / Hs_density)

            for i in range(grad_cons.shape[0]):
                g_xc = grad_cons[i, 0:num_elems].copy()
                g_xfilt_c = g_xc * dxphys_dxfilt
                grad_cons[i, 0:num_elems] = H_density.T @ (g_xfilt_c / Hs_density)

        grad_obj = np.array([grad_obj]).reshape((num_design_var, 1))
        cons = np.array(cons).reshape((-1, 1))
        grad_cons = np.array(grad_cons).reshape((len(cons), num_design_var))

        log_objective_and_constraints(history, obj, J_phys, Jn, P, Pn, Pzn, cons,
                                      to_params, verbose=verbose)

        # ================ CONTINUATION + BETA UPDATE ==================
        mmaIterations += 1

        if use_continuation and (mmaIterations % 10 == 0):
            increment_SIMP_THERMAL_PENALTY(0.25)
            increment_SIMP_STRUCTURAL_PENALTY(0.25)
        if use_heaviside_projection:
            beta_new, beta_updated = update_beta(
                beta_proj=beta_proj,
                mmaIterations=mmaIterations,
                beta_schedule=beta_schedule,
            )
            if beta_updated:
                beta_proj = beta_new
                if verbose:
                    print(f"[PROJ] Updated beta -> {beta_proj:.4g} (eta={eta_proj}, scheme={beta_schedule_scheme.value})")
        return np.array([[float(obj)]]), grad_obj, cons, grad_cons

    initialDensity = 0.5
    for constraint in to_params.Constraints:
        if constraint[0] == TO_QOI.VOLUME_FRACTION and not (
            isinstance(constraint[1], dict) and ("material_id" in constraint[1])
        ):
            initialDensity = constraint[2]
            break

    d0 = initialDensity * np.ones(num_elems, dtype=float)
    z0 = np.zeros(latentDim * num_elems, dtype=float)

    if z0_init_method == Z0InitMethod.LIGHTEST:
        zLightest = matEncoder.getLightestMaterial()
        for i in range(latentDim):
            z0[i * num_elems : (i + 1) * num_elems] = zLightest[i]

    elif z0_init_method == Z0InitMethod.HEAVIEST:
        zHeaviest = matEncoder.getHeaviestMaterial()
        for i in range(latentDim):
            z0[i * num_elems : (i + 1) * num_elems] = zHeaviest[i]

    elif z0_init_method == Z0InitMethod.ORIGIN:
        z0[:] = 0.0

    elif z0_init_method == Z0InitMethod.UNIFORM:
        for i in range(latentDim):
            z0[i * num_elems : (i + 1) * num_elems] = np.random.uniform(
                -0.5, 0.5, size=num_elems
            )
    else:
        raise ValueError(f"Unknown z0_init_method: {z0_init_method}")

    zeta0 = np.concatenate((d0, z0), axis=0).reshape(-1, 1)

    lowerBound = np.zeros(num_design_var, dtype=float).reshape(-1, 1)
    upperBound = np.ones(num_design_var, dtype=float).reshape(-1, 1)

    lowerBound[num_elems:num_design_var] = np.min(zRealPoints.detach().cpu().numpy())
    upperBound[num_elems:num_design_var] = np.max(zRealPoints.detach().cpu().numpy())

    # ============================ RUN MMA ============================
    nVariables = num_design_var
    nConstraints = len(to_params.Constraints)

    tStart = time.time()
    optResults = runMMA(
        nVariables,
        nConstraints,
        MMTO_optimization_function,
        zeta0.reshape(-1, 1),
        lowerBound,
        upperBound,
        maxIterations=maxIterations,
        timeLimitSecs=timeLimit,
        move_limit=0.05,
        kktTol=1e-6,
        fTolerance=rel_conv_tol,
        gTolerance=rel_conv_tol,
        verbose=False,
    )
    zetaOptimal = np.asarray(optResults[0]).flatten()
    tEnd = time.time()
    print(f"Total optimization time: {tEnd - tStart:.2f} seconds")
    run_postprocessing(
        history=history,
        zetaOptimal=zetaOptimal,
        fe_solver=fe_solver_structural,
        matEncoder=matEncoder,
        mat_lib=mat_lib,
        MaterialModel=MaterialModel,
        heaviside_projection=heaviside_projection,
        H_density=H_density,
        Hs_density=Hs_density,
        zRealPoints=zRealPoints,
        material_colors=material_colors,
        num_elems=num_elems,
        latentDim=latentDim,
        beta_proj=beta_proj,
        eta_proj=eta_proj,
        elem_volume=elem_volume,
        design_domain_volume=design_domain_volume,
        apply_density_filter=APPLY_DENSITY_FILTER,
        use_heaviside_projection=use_heaviside_projection,
        binarize_topology=binarize_topology,
        snap_to_real_material=snap_to_real_material,
        save_dir=save_dir,
    )

if __name__ == "__main__":
    save_dir = start_run(name="CorbelMidLoad_Mass_StressFF_BM2")
    to_problem = MMTOExamplesPureStructural.CorbelMidLoad_Mass_StressFF_BM2

    run_topopt(
        to_problem=to_problem,
        use_penalization=True,
        use_pretrained_vae=True,
        snap_to_real_material=False,
        use_constant_poissons_ratio=True,
        constant_poissons_ratio=0.3,
        use_continuation=False,
        gamma_factor=1.1,
        penalty_start_iter=0,
        use_heaviside_projection=True,
        eta_proj=0.5,
        beta_schedule_scheme=BetaScheduleScheme.PAPER,  # implement your own on the fly
        use_z_smoothing=False,
        z_smoothing_weight=1e-4,
        distance_penalty_disable_after_iter=250,
        z_smoothing_disable_after_iter=250,
        maxIterations=300,
        latent_space_diagnostics=False,
        save_dir=save_dir,
        verbose=True,   # set False for one-line-per-iteration output
    )