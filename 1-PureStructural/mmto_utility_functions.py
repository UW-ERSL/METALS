import numpy as np
import torch
import matplotlib.pyplot as plt
from dataclasses import replace


def run_basic_latent_space_diagnostics(
    matEncoder,
    zRealPoints,
    property_names=("Density", "Youngs_Modulus", "Yield_Strength"),
    n_interp=21,
    local_eps=0.05,
    random_seed=0,
    show_plots=True,
):
    """
    Minimal VAE latent-space diagnostics.

    Checks:
      1) reconstruction error on real material latent points
      2) interpolation behavior between all real-material pairs
      3) local finite-difference sensitivity near real material points
    """
    rng = np.random.default_rng(random_seed)

    def decode_props(z_np):
        z_t = torch.tensor(z_np, dtype=torch.float32)
        with torch.no_grad():
            decoded = matEncoder.vaeNet.decoder(z_t)
            props = matEncoder.getMaterialProperties(decoded)
        out = {}
        for k in property_names:
            if k in props:
                out[k] = props[k].detach().cpu().numpy().astype(float).reshape(-1)
        return out

    def get_true_props():
        out = {}
        for k in property_names:
            if k in matEncoder.materialAttributes:
                idx = matEncoder.materialAttributes[k]["idx"]
                out[k] = np.asarray(matEncoder.rawData[:, idx], dtype=float)
        return out

    z_real = zRealPoints.detach().cpu().numpy().astype(float)
    n_materials, latentDim = z_real.shape

    true_props = get_true_props()
    pred_props = decode_props(z_real)

    print("\n" + "=" * 80)
    print("LATENT SPACE DIAGNOSTICS")
    print("=" * 80)

    print("\n[1] Reconstruction error at real material latent points")
    for k in property_names:
        if k not in true_props or k not in pred_props:
            continue
        true_k = true_props[k]
        pred_k = pred_props[k]
        denom = np.maximum(np.abs(true_k), 1e-12)
        pct_err = 100.0 * np.abs(pred_k - true_k) / denom
        print(
            f"{k:>16s} | mean % err = {np.mean(pct_err):8.4f} | "
            f"max % err = {np.max(pct_err):8.4f}"
        )

    print("\n[2] Interpolation sanity between random real-material pairs")
    if n_materials >= 2:
        pairs = [(i, j) for i in range(n_materials) for j in range(i + 1, n_materials)]

        for pidx, (i, j) in enumerate(pairs, start=1):
            t = np.linspace(0.0, 1.0, n_interp)
            z_line = (1.0 - t)[:, None] * z_real[i][None, :] + t[:, None] * z_real[j][None, :]
            line_props = decode_props(z_line)

            print(f"  Pair {pidx}: material {i} -> {j}")
            for k in property_names:
                if k not in line_props:
                    continue
                vals = line_props[k]
                vmin = np.min(vals)
                vmax = np.max(vals)
                end_min = min(vals[0], vals[-1])
                end_max = max(vals[0], vals[-1])
                overshoot = (vmin < end_min - 1e-12) or (vmax > end_max + 1e-12)
                print(
                    f"    {k:>16s} | line min={vmin:.6g} max={vmax:.6g} "
                    f"| endpoint range=[{end_min:.6g}, {end_max:.6g}] "
                    f"| overshoot={overshoot}"
                )

            if show_plots:
                fig, axes = plt.subplots(
                    len(property_names),
                    1,
                    figsize=(6, 2.5 * len(property_names)),
                    squeeze=False,
                )
                axes = axes.flatten()
                for ax, k in zip(axes, property_names):
                    if k not in line_props:
                        ax.set_visible(False)
                        continue
                    vals = line_props[k]
                    ax.plot(t, vals, marker="o", markersize=3)
                    ax.set_title(f"Interp {i}->{j}: {k}")
                    ax.set_xlabel("t")
                    ax.set_ylabel(k)
                    ax.grid(True)
                plt.tight_layout()
                plt.show()

    print("\n[3] Local finite-difference sensitivity near real material points")
    directions = np.eye(latentDim)
    for i in range(n_materials):
        z0 = z_real[i].copy()
        base = decode_props(z0[None, :])
        print(f"  Material {i}:")
        for k in property_names:
            if k not in base:
                continue
            max_fd = 0.0
            for d in directions:
                zp = z0 + local_eps * d
                zm = z0 - local_eps * d
                fp = decode_props(zp[None, :])[k][0]
                fm = decode_props(zm[None, :])[k][0]
                fd = abs(fp - fm) / (2.0 * local_eps)
                max_fd = max(max_fd, fd)
            print(f"    {k:>16s} | max local FD magnitude = {max_fd:.6g}")

    print("=" * 80 + "\n")


def create_density_and_material_filters(fe_solver_structural, to_params, createFilters):
    """
    Build two filters:
      - density filter: uses to_params.RelativeFilterRadius
      - material filter: uses to_params.MaterialFilterRadius if provided,
                         otherwise falls back to density filter
    """
    H_density, Hs_density = createFilters(fe_solver_structural, to_params)

    mat_radius = (
        to_params.RelativeFilterRadius
        if to_params.MaterialFilterRadius is None
        else to_params.MaterialFilterRadius
    )

    if abs(mat_radius - to_params.RelativeFilterRadius) < 1e-15:
        return H_density, Hs_density, H_density, Hs_density

    to_params_mat = replace(to_params, RelativeFilterRadius=mat_radius)
    H_material, Hs_material = createFilters(fe_solver_structural, to_params_mat)

    return H_density, Hs_density, H_material, Hs_material


def compute_domain_volume(fe_solver_structural) -> float:
    """
    Original design-domain volume before optimization.
    """
    elem_volume = float(np.prod(fe_solver_structural.mesh.elem_size))
    return float(fe_solver_structural.mesh.num_elems * elem_volume)

def compute_max_failure_factor(vm, material_properties):
    """
    Compute max failure factor = max(vonMises / Yield_Strength).

    Returns None if Yield_Strength is not present.
    """
    if "Yield_Strength" not in material_properties:
        return None

    vm = np.asarray(vm, dtype=float).reshape(-1)
    yield_strength = material_properties["Yield_Strength"].detach().cpu().numpy().astype(float).reshape(-1)

    return float(np.max(vm / np.maximum(yield_strength, 1e-12)))


def _property_to_numpy(material_properties, key):
    if key not in material_properties:
        return None
    val = material_properties[key]
    if hasattr(val, "detach"):
        return val.detach().cpu().numpy().astype(float).reshape(-1)
    return np.asarray(val, dtype=float).reshape(-1)


def compute_stress_violation_diagnostics(
    vm,
    material_properties,
    x_vec=None,
    active_thresh=0.5,
    mesh=None,
    bc=None,
    violation_tol=0.0,
):
    """
    Compute elementwise stress-constraint diagnostics.

    Local constraint:
        FF_e = vonMises_e / YieldStrength_e
        Q_e  = FF_e - 1

    An element violates the local stress constraint when:
        Q_e > violation_tol

    The active-topology count uses x_vec > active_thresh.
    """
    yield_strength = _property_to_numpy(material_properties, "Yield_Strength")
    if yield_strength is None:
        return None

    vm = np.asarray(vm, dtype=float).reshape(-1)
    ff = vm / np.maximum(yield_strength, 1e-12)
    q_local = ff - 1.0

    n_elems = vm.size
    all_mask = np.ones(n_elems, dtype=bool)

    if x_vec is None:
        x_arr = None
        active_mask = all_mask.copy()
    else:
        x_arr = np.asarray(x_vec, dtype=float).reshape(-1)
        active_mask = x_arr > active_thresh

    violation_mask_all = q_local > violation_tol
    violation_mask_active = violation_mask_all & active_mask
    satisfied_mask_active = (~violation_mask_all) & active_mask

    active_count = int(np.sum(active_mask))
    n_viol_all = int(np.sum(violation_mask_all))
    n_viol_active = int(np.sum(violation_mask_active))
    n_sat_active = int(np.sum(satisfied_mask_active))

    active_ff = ff[active_mask] if active_count > 0 else np.array([], dtype=float)
    active_q = q_local[active_mask] if active_count > 0 else np.array([], dtype=float)

    viol_indices_active = np.where(violation_mask_active)[0]
    viol_indices_all = np.where(violation_mask_all)[0]

    out = {
        "ff": ff,
        "q_local": q_local,
        "active_mask": active_mask,
        "violation_mask_all": violation_mask_all,
        "violation_mask_active": violation_mask_active,
        "satisfied_mask_active": satisfied_mask_active,
        "violating_indices_active": viol_indices_active,
        "violating_indices_all": viol_indices_all,
        "num_elements": n_elems,
        "num_active": active_count,
        "num_violating_all": n_viol_all,
        "num_violating_active": n_viol_active,
        "num_satisfied_active": n_sat_active,
        "frac_violating_all": n_viol_all / max(n_elems, 1),
        "frac_violating_active": n_viol_active / max(active_count, 1),
        "max_ff_all": float(np.max(ff)) if ff.size else float("nan"),
        "max_q_all": float(np.max(q_local)) if q_local.size else float("nan"),
        "max_ff_active": float(np.max(active_ff)) if active_ff.size else float("nan"),
        "max_q_active": float(np.max(active_q)) if active_q.size else float("nan"),
    }

    if mesh is not None and hasattr(mesh, "elem_centers") and viol_indices_active.size > 0:
        centers = np.asarray(mesh.elem_centers, dtype=float)
        viol_centers = centers[viol_indices_active]

        out["violating_centroid"] = np.mean(viol_centers, axis=0)
        out["violating_bbox_min"] = np.min(viol_centers, axis=0)
        out["violating_bbox_max"] = np.max(viol_centers, axis=0)

        if bc is not None and hasattr(bc, "force") and hasattr(mesh, "node_xyz"):
            force = np.asarray(bc.force, dtype=float).reshape(-1)
            loaded_dofs = np.where(np.abs(force) > 1e-14)[0]
            loaded_nodes = np.unique(loaded_dofs // 3)

            if loaded_nodes.size > 0:
                load_xyz = np.asarray(mesh.node_xyz, dtype=float)[loaded_nodes]
                d = np.linalg.norm(
                    viol_centers[:, None, :] - load_xyz[None, :, :],
                    axis=2,
                )
                nearest = np.min(d, axis=1)

                out["num_loaded_nodes"] = int(loaded_nodes.size)
                out["nearest_load_distance_min"] = float(np.min(nearest))
                out["nearest_load_distance_mean"] = float(np.mean(nearest))
                out["nearest_load_distance_max"] = float(np.max(nearest))

    return out


def print_stress_violation_diagnostics(diag, iteration=None, prefix="[STRESS DIAG]"):
    """
    Print compact stress violation diagnostics.
    """
    if diag is None:
        print(f"{prefix} Yield_Strength not present; stress violation count unavailable.")
        return

    iter_txt = "" if iteration is None else f" Iter {int(iteration):4d} |"

    print(
        f"{prefix}{iter_txt} "
        f"active violations = {diag['num_violating_active']}/{diag['num_active']} "
        f"({100.0 * diag['frac_violating_active']:.3f}%) | "
        f"all violations = {diag['num_violating_all']}/{diag['num_elements']} "
        f"({100.0 * diag['frac_violating_all']:.3f}%) | "
        f"max FF active = {diag['max_ff_active']:.6g} | "
        f"max Q active = {diag['max_q_active']:.6g}"
    )

    if diag["num_violating_active"] > 0:
        sample_ids = diag["violating_indices_active"][:20]
        print(f"{prefix} violating active element IDs, first 20: {sample_ids}")

        if "violating_centroid" in diag:
            c = diag["violating_centroid"]
            bmin = diag["violating_bbox_min"]
            bmax = diag["violating_bbox_max"]
            print(
                f"{prefix} violating centroid xyz = "
                f"[{c[0]:.6g}, {c[1]:.6g}, {c[2]:.6g}]"
            )
            print(
                f"{prefix} violating bbox min xyz = "
                f"[{bmin[0]:.6g}, {bmin[1]:.6g}, {bmin[2]:.6g}]"
            )
            print(
                f"{prefix} violating bbox max xyz = "
                f"[{bmax[0]:.6g}, {bmax[1]:.6g}, {bmax[2]:.6g}]"
            )

        if "nearest_load_distance_min" in diag:
            print(
                f"{prefix} distance from violating active elements to nearest loaded node: "
                f"min={diag['nearest_load_distance_min']:.6g}, "
                f"mean={diag['nearest_load_distance_mean']:.6g}, "
                f"max={diag['nearest_load_distance_max']:.6g} "
                f"over {diag['num_loaded_nodes']} loaded nodes"
            )


def plot_stress_violation_distribution(
    fe_solver_structural,
    violation_mask_active,
    x_vec,
    title="Stress Constraint Satisfaction",
    active_density_plot_thresh=0.1,
    save_path=None,
):
    """
    Plot optimized topology with:
        green = active element satisfies FF <= 1
        red   = active element violates FF > 1

    Void / very-low-density elements are hidden by the existing mesh pseudo-density mask.
    """
    x_vec = np.asarray(x_vec, dtype=float).reshape(-1)
    violation_mask_active = np.asarray(violation_mask_active, dtype=bool).reshape(-1)

    fe_solver_structural.mesh.setPseudoDensity(x_vec)

    # 0 = satisfied, 1 = violating
    status_indices = np.zeros_like(x_vec, dtype=int)
    status_indices[violation_mask_active] = 1

    old_pseudo = fe_solver_structural.mesh.elemPseudoDensity.copy()

    try:
        # The inherited plotting method masks elemPseudoDensity <= 0.1.
        # This keeps the plot on the optimized topology rather than showing void elements.
        fe_solver_structural.mesh.setPseudoDensity(x_vec)

        fe_solver_structural.plot_material_distribution(
            material_indices=status_indices,
            material_names=["Satisfied: FF <= 1", "Violating: FF > 1"],
            material_colors=["green", "red"],
            title=title,
            mask_low_pseudodensity=True,
            auto_close=True,
            save_path=save_path,
            fontsize=10,
            show_legend=True,
        )
    finally:
        fe_solver_structural.mesh.setPseudoDensity(old_pseudo)


def compute_mass_from_x_and_z(
    x_density: np.ndarray,
    z_flat: np.ndarray,
    matEncoder,
    latentDim: int,
    elem_volume: float,
):
    """
    Compute total mass = sum_e x_e * rho_e * V_e using decoded material density.

    Returns
    -------
    mass : float | None
        None if 'Density' is not present in the decoded material properties.
    """
    x_density = np.asarray(x_density, dtype=float).reshape(-1)

    zPts = torch.tensor(
        np.asarray(z_flat, dtype=float).reshape(latentDim, -1).T,
        dtype=torch.float32,
    )

    with torch.no_grad():
        decoded = matEncoder.vaeNet.decoder(zPts)
        material_properties = matEncoder.getMaterialProperties(decoded)

    if "Density" not in material_properties:
        return None

    rho_e = material_properties["Density"].detach().cpu().numpy().astype(float)
    mass = float(np.sum(x_density * rho_e) * elem_volume)
    return mass


def eval_stress_compliance_metrics(
    x_vec,
    z_vec,
    label,
    active_thresh,
    latentDim,
    matEncoder,
    fe_solver_structural,
    mat_lib,
    MaterialModel,
    elem_volume,
    design_domain_volume,
):
    x_vec = np.asarray(x_vec, dtype=float).flatten()
    z_vec = np.asarray(z_vec, dtype=float).flatten()

    zPts = torch.tensor(np.asarray(z_vec, dtype=float).reshape(latentDim, -1).T).float()

    with torch.no_grad():
        decoded = matEncoder.vaeNet.decoder(zPts)
        material_properties = matEncoder.getMaterialProperties(decoded)
        Youngs_Modulus = material_properties["Youngs_Modulus"].detach().cpu().numpy()

    fe_solver_structural.mat_prop = [
        mat_lib.create_material_with_defaults(
            name=f"Material_{i+1}",
            youngs_modulus=float(Youngs_Modulus[i]),
        )
        for i in range(len(Youngs_Modulus))
    ]
    fe_solver_structural.set_material(fe_solver_structural.mat_prop)

    sol_local = fe_solver_structural.solve(x_vec, MaterialModel.SIMP)
    fe_solver_structural.mesh.setPseudoDensity(x_vec)
    fe_solver_structural.postprocess()

    vm = np.asarray(fe_solver_structural.vonMisesStress, dtype=float)
    max_failure_factor = compute_max_failure_factor(vm, material_properties)
    pnorm_vm = float(fe_solver_structural.pNormStress)
    compliance = float(np.einsum("i,i->", fe_solver_structural.total_force, sol_local))

    active = x_vec > active_thresh
    vm_min_active = float(np.min(vm[active])) if np.any(active) else float("nan")
    vm_max_active = float(np.max(vm[active])) if np.any(active) else float("nan")

    final_mass = compute_mass_from_x_and_z(
        x_density=x_vec,
        z_flat=z_vec,
        matEncoder=matEncoder,
        latentDim=latentDim,
        elem_volume=elem_volume,
    )

    if final_mass is not None:
        mass_ratio = final_mass / max(design_domain_volume, 1e-12)
    else:
        mass_ratio = None

    print("\n" + "=" * 70)
    print(f"[FINAL METRICS] {label}")

    if final_mass is not None:
        print(f" Final mass       : {final_mass:.6g}")
        print(
            f" Mass ratio       : {mass_ratio:.6g} "
            f"(final mass / volume of original design domain before optimization)"
        )
    else:
        print(" Final mass       : N/A (Density not present in material database)")
        print(" Mass ratio       : N/A (Density not present in material database)")
    if max_failure_factor is not None:
        print(f" Max failure factor : {max_failure_factor:.6g}")
    else:
        print(" Max failure factor : N/A (Yield_Strength not present in material database)")

    print(f" Compliance       : {compliance:.6g}")
    print(f" p-norm(vonMises) : {pnorm_vm:.6g}")
    print(f" max(vonMises)    : {vm_max_active:.6g} (x > {active_thresh})")
    print(f" min(vonMises)    : {vm_min_active:.6g} (x > {active_thresh})")
    print("=" * 70 + "\n")

    return {
        "final_mass": final_mass,
        "mass_ratio": mass_ratio,
        "compliance": compliance,
        "pnorm_vm": pnorm_vm,
        "vm_max_active": vm_max_active,
        "vm_min_active": vm_min_active,
        "max_failure_factor": max_failure_factor,
    }


def plot_history(history: dict):
    plt.figure(figsize=(12, 6))
    plt.plot(
        range(len(history["objective"])),
        history["objective"],
        label="MMA objective f (Jn + Pn + Pz_n)",
        linewidth=2,
        marker="o",
        markevery=5,
    )

    if len(history["constraints"]) > 0:
        markers = ["s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "+", "x", "|", "_"]
        colors = plt.cm.tab10.colors

        for i in range(len(history["constraints"][0])):
            constraint_values = [
                history["constraints"][j][i] for j in range(len(history["constraints"]))
            ]
            plt.plot(
                range(len(constraint_values)),
                constraint_values,
                label=f"Constraint {i+1}",
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                markevery=5,
            )

    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Objective and Constraints vs. Iterations")
    plt.legend()
    plt.grid()
    plt.show()

def update_paper_history(
    history,
    x_phys,
    z_phys_elemmajor,
    material_properties,
    von_mises,
    elem_volume,
    design_domain_volume,
):
    """
    Update paper-style histories:
      - mass ratio
      - maximum failure factor

    If Density or Yield_Strength are unavailable, store np.nan.
    """
    density_vals = None
    yield_vals = None

    if "Density" in material_properties:
        density_vals = material_properties["Density"].detach().cpu().numpy().astype(float)

    if "Yield_Strength" in material_properties:
        yield_vals = material_properties["Yield_Strength"].detach().cpu().numpy().astype(float)

    # Mass ratio
    if density_vals is not None:
        final_mass = float(np.sum(np.asarray(x_phys, dtype=float).reshape(-1) * density_vals) * elem_volume)
        mass_ratio = final_mass / max(float(design_domain_volume), 1e-12)
    else:
        mass_ratio = np.nan

    max_failure_factor = compute_max_failure_factor(von_mises, material_properties)
    if max_failure_factor is None:
        max_failure_factor = np.nan

    history["mass_ratio"].append(mass_ratio)
    history["max_failure_factor"].append(max_failure_factor)

def plot_reported_histories(
    history,
    y1_getter=None,
    y2_getter=None,
    y1_label=None,
    y2_label=None,
):
    """
    Plot 1: always J_phys + MMA constraints
    Plot 2: by default mass ratio + max failure factor if available

    Optional customization for Plot 2:
      y1_getter(history) -> 1D array-like
      y2_getter(history) -> 1D array-like
    """

    # --------------------------------------------------
    # Plot 1: J_phys + MMA constraints
    # --------------------------------------------------
    plt.figure(figsize=(12, 6))

    plt.plot(
        range(len(history["J_phys"])),
        history["J_phys"],
        label=r"$J_{\mathrm{phys}}$",
        linewidth=2.2,
        marker="o",
        markevery=max(1, len(history["J_phys"]) // 20),
    )

    if len(history["constraints"]) > 0:
        markers = ["s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "+", "x"]
        colors = plt.cm.tab10.colors

        first_constraint = np.asarray(history["constraints"][0]).flatten()
        n_constraints = len(first_constraint)

        for i in range(n_constraints):
            cvals = [
                np.asarray(history["constraints"][j]).flatten()[i]
                for j in range(len(history["constraints"]))
            ]
            plt.plot(
                range(len(cvals)),
                cvals,
                label=f"Constraint {i+1}",
                linewidth=1.8,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                markevery=max(1, len(cvals) // 20),
            )

    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title(r"Iteration History: $J_{\mathrm{phys}}$ and MMA Constraints")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # Plot 2: default or user-defined, with dual y-axes
    # --------------------------------------------------
    if y1_getter is None:
        y1 = np.asarray(history.get("mass_ratio", []), dtype=float)
        y1_name = "Mass ratio"
    else:
        y1 = np.asarray(y1_getter(history), dtype=float)
        y1_name = y1_label if y1_label is not None else "Series 1"

    if y2_getter is None:
        y2 = np.asarray(history.get("max_failure_factor", []), dtype=float)
        y2_name = "Maximum failure factor"
    else:
        y2 = np.asarray(y2_getter(history), dtype=float)
        y2_name = y2_label if y2_label is not None else "Series 2"

    if len(y1) == 0 or len(y2) == 0:
        return

    if np.all(np.isnan(y1)) or np.all(np.isnan(y2)):
        return

    it1 = np.arange(len(y1))
    it2 = np.arange(len(y2))

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    line1 = ax1.plot(
        it1,
        y1,
        label=y1_name,
        linewidth=2.2,
        marker="o",
        markevery=max(1, len(y1) // 20),
    )[0]

    line2 = ax2.plot(
        it2,
        y2,
        label=y2_name,
        linewidth=2.2,
        marker="s",
        markevery=max(1, len(y2) // 20),
        linestyle="--",
    )[0]

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(y1_name)
    ax2.set_ylabel(y2_name)

    ax1.grid(True, alpha=0.3)
    ax1.set_title("Iteration History: Reported Design Metrics")

    # combined legend
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    fig.tight_layout()
    plt.show()


# ============================================================================
# RUN I/O + PER-ITERATION DIAGNOSTICS + POST-PROCESSING
# Extracted from MainPureStructural so the driver stays optimization-only.
# History is recorded in all modes; `verbose` gates console output only.
# ============================================================================

def init_run_history():
    """Return a fresh history dict for a topopt run."""
    return {
        "objective": [],
        "constraints": [],
        "J_phys": [],
        "P": [],
        "Pn": [],
        "Pz_n": [],
        "grey": [],
        "mass_ratio": [],
        "max_failure_factor": [],
        "num_stress_violating_active": [],
        "frac_stress_violating_active": [],
        "num_stress_violating_all": [],
        "frac_stress_violating_all": [],
        "max_q_active": [],
        "max_q_all": [],
    }


def start_run(base="runs", name=None):
    """Create a timestamped run folder and tee stdout/stderr into run.log.

    Returns the run-folder path (str). The matplotlib autosave hook is enabled
    later, inside run_postprocessing, so intermediate/live plots are skipped.
    """
    import os, sys, datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, "run_" + stamp + (("_" + name) if name else ""))
    os.makedirs(run_dir, exist_ok=True)
    print(f"[OUTPUT] folder: {os.path.abspath(run_dir)}")

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, d):
            for s in self.streams:
                s.write(d); s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    logf = open(os.path.join(run_dir, "run.log"), "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, logf)
    sys.stderr = _Tee(sys.__stderr__, logf)
    return run_dir


def enable_plot_autosave(outdir):
    """Hook matplotlib so every plt.show() also saves the figure (uncropped) to outdir.

    Enable AFTER the optimization loop so live/intermediate topologies are skipped.
    PyVista plots save themselves via their own save_path= argument.
    """
    import os
    os.makedirs(outdir, exist_ok=True)
    n = {"i": 0}

    _orig_mpl_show = plt.show
    _saved_ids = set()   # dedup by figure OBJECT id, not fignum (numbers get reused
                         # when earlier figures close -> history plots were skipped)
    _keep = []           # strong refs so saved figures aren't GC'd (prevents id reuse)

    def _mpl_show(*a, **k):
        for fnum in plt.get_fignums():
            fig = plt.figure(fnum)
            if id(fig) in _saved_ids:
                continue
            n["i"] += 1
            fig.savefig(
                os.path.join(outdir, f"plot_{n['i']:02d}_mpl.png"),
                bbox_inches="tight", dpi=150,
            )
            _saved_ids.add(id(fig))
            _keep.append(fig)
        return _orig_mpl_show(*a, **k)

    plt.show = _mpl_show
    print(f"[OUTPUT] matplotlib autosave ON -> {os.path.abspath(outdir)}")


def log_design_state(history, iteration, x_phys, z, material_properties, zTorch,
                     matEncoder, num_elems, beta_proj, eta_proj, verbose=True):
    """Record the grey fraction; in verbose mode also print the iteration header,
    grey %, and the [MAT] active-material statistics."""
    grey_elements = np.sum((x_phys > 0.1) & (x_phys < 0.9))
    fraction_grey = grey_elements / num_elems
    history["grey"].append(float(fraction_grey))

    if not verbose:
        return

    print("-------------- Iteration", iteration, "-----------------")
    print(
        f"Percentage grey elements (x_phys): {fraction_grey*100:.2f}% | "
        f"beta={beta_proj:.3g}, eta={eta_proj}"
    )
    act = x_phys > 0.5
    Y = material_properties["Yield_Strength"].detach().cpu().numpy()
    mass_density = material_properties["Density"].detach().cpu().numpy()
    midx = matEncoder.getClosestRealMaterialIndex(zTorch).cpu().numpy()
    counts = np.bincount(midx[act], minlength=len(matEncoder.materialNames))
    if act.any():
        print(f"[MAT] Y_active min/mean/max="
              f"{Y[act].min():.3g}/{Y[act].mean():.3g}/{Y[act].max():.3g} | "
              f"rho_active min/mean/max="
              f"{mass_density[act].min():.3g}/{mass_density[act].mean():.3g}/{mass_density[act].max():.3g} | "
              f"counts={counts.tolist()}")


def record_fea_diagnostics(history, iteration, x_phys, z, material_properties,
                           fe_solver, elem_volume, design_domain_volume,
                           plot_progress=False, plotter=None, verbose=True):
    """After the FEA solve: update paper history (mass ratio, max FF), record
    stress-violation diagnostics into history, optionally print them (verbose),
    and optionally draw the realtime topology (plot_progress)."""
    update_paper_history(
        history=history,
        x_phys=x_phys,
        z_phys_elemmajor=z,
        material_properties=material_properties,
        von_mises=fe_solver.vonMisesStress,
        elem_volume=elem_volume,
        design_domain_volume=design_domain_volume,
    )
    stress_diag = compute_stress_violation_diagnostics(
        vm=fe_solver.vonMisesStress,
        material_properties=material_properties,
        x_vec=x_phys,
        active_thresh=0.5,
        mesh=fe_solver.mesh,
        bc=fe_solver.bc,
        violation_tol=0.0,
    )

    if stress_diag is not None:
        history["num_stress_violating_active"].append(stress_diag["num_violating_active"])
        history["frac_stress_violating_active"].append(stress_diag["frac_violating_active"])
        history["num_stress_violating_all"].append(stress_diag["num_violating_all"])
        history["frac_stress_violating_all"].append(stress_diag["frac_violating_all"])
        history["max_q_active"].append(stress_diag["max_q_active"])
        history["max_q_all"].append(stress_diag["max_q_all"])

    if verbose:
        print_stress_violation_diagnostics(
            stress_diag, iteration=iteration + 1, prefix="[STRESS DIAG]"
        )
    if plot_progress:
        fe_solver.plot_pseudo_density_realtime(
            title=f"Iter {iteration + 1}", external_plotter=plotter
        )


def log_objective_and_constraints(history, obj, J_phys, Jn, P, Pn, Pzn, cons,
                                  to_params, verbose=True):
    """Record objective/constraint history. In verbose mode print the full
    Obj line + per-constraint lines (exactly as before); in simple mode print
    one compact per-iteration line."""
    cons_flat = np.asarray(cons).flatten()
    constraint_names = [getattr(c[0], "name", str(c[0])) for c in to_params.Constraints]

    history["objective"].append(float(obj))
    history["constraints"].append(cons_flat.copy())
    history["J_phys"].append(J_phys)
    history["P"].append(P)
    history["Pn"].append(Pn)
    history["Pz_n"].append(Pzn)

    if verbose:
        print(
            f"Obj: J={J_phys:.4g} | Jn={Jn:.4g} | P={P:.3g} | "
            f"Pn={Pn:.3g} | Pz_n={Pzn:.3g} | f={float(obj):.4g}"
        )
        for idx, val in enumerate(cons_flat):
            inequality = "<="
            if constraint_names[idx] in ("STRESS_SAFETY_FACTOR", "TEMPERATURE_SAFETY_FACTOR"):
                inequality = ">="
            print(
                f"Constraint {idx+1} ({constraint_names[idx]}): "
                f"{(val + 1) * to_params.Constraints[idx][2]:.3g} "
                f"{inequality} {to_params.Constraints[idx][2]:.3g}?"
            )
    else:
        mass = history["mass_ratio"][-1] if history["mass_ratio"] else float("nan")
        maxff = history["max_failure_factor"][-1] if history["max_failure_factor"] else float("nan")
        cons_str = " ".join(
            f"{constraint_names[idx]}={(val + 1) * to_params.Constraints[idx][2]:.3g}"
            for idx, val in enumerate(cons_flat)
        )
        print(f"[iter] f={float(obj):.4g} J={J_phys:.4g} mass={mass:.3g} "
              f"maxFF={maxff:.3g} | {cons_str}")


def run_postprocessing(history, zetaOptimal, fe_solver, matEncoder, mat_lib,
                       MaterialModel, heaviside_projection, H_density, Hs_density,
                       zRealPoints, material_colors, num_elems, latentDim,
                       beta_proj, eta_proj, elem_volume, design_domain_volume,
                       apply_density_filter, use_heaviside_projection,
                       binarize_topology, snap_to_real_material, save_dir=None):
    """Reconstruct the optimal fields, evaluate stress/compliance metrics at the
    continuous / binarized / snapped designs, and draw the final plots."""
    import os
    if save_dir is not None:
        enable_plot_autosave(save_dir)

    def _p(name):
        return os.path.join(save_dir, name) if save_dir else None

    # Reconstruct physical fields
    xOptimal_raw = zetaOptimal[0:num_elems].copy()
    zOptimal = zetaOptimal[num_elems:].reshape(latentDim, -1).T   # numpy, element-major

    xOptimal_filt = (H_density @ xOptimal_raw) / Hs_density if apply_density_filter else xOptimal_raw.copy()
    if use_heaviside_projection:
        xOptimal_cont, _ = heaviside_projection(xOptimal_filt, beta_proj, eta_proj)
    else:
        xOptimal_cont = xOptimal_filt.copy()

    eval_stress_compliance_metrics(
        x_vec=xOptimal_cont,
        z_vec=zOptimal.T.reshape(-1),
        label="Continuous (x_phys projected, raw z)",
        active_thresh=0.5,
        latentDim=latentDim,
        matEncoder=matEncoder,
        fe_solver_structural=fe_solver,
        mat_lib=mat_lib,
        MaterialModel=MaterialModel,
        elem_volume=elem_volume,
        design_domain_volume=design_domain_volume,
    )

    xOptimal = xOptimal_cont.copy()
    if binarize_topology:
        x_sorted = np.sort(xOptimal)
        threshold = x_sorted[int((1 - np.mean(xOptimal)) * len(xOptimal))]
        xOptimal = np.where(xOptimal < threshold, 0.0, 1.0)

    eval_stress_compliance_metrics(
        x_vec=xOptimal,
        z_vec=zOptimal.T.reshape(-1),
        label="Binarized topology (x projected -> binarized, raw z)",
        active_thresh=0.5,
        latentDim=latentDim,
        matEncoder=matEncoder,
        fe_solver_structural=fe_solver,
        mat_lib=mat_lib,
        MaterialModel=MaterialModel,
        elem_volume=elem_volume,
        design_domain_volume=design_domain_volume,
    )

    zOptimalPts = torch.tensor(zOptimal, dtype=torch.float32)
    if snap_to_real_material:
        zSnappedPts = torch.tensor(matEncoder.getClosestRealMaterialZValues(zOptimalPts))
        zOptimalPts = zSnappedPts
    eval_stress_compliance_metrics(
        x_vec=xOptimal,
        z_vec=zOptimalPts.T.flatten().numpy(),
        label="Binarized + snapped materials (x binarized, snapped z)",
        active_thresh=0.5,
        latentDim=latentDim,
        matEncoder=matEncoder,
        fe_solver_structural=fe_solver,
        mat_lib=mat_lib,
        MaterialModel=MaterialModel,
        elem_volume=elem_volume,
        design_domain_volume=design_domain_volume,
    )

    decoded = matEncoder.vaeNet.decoder(zOptimalPts)
    material_properties = matEncoder.getMaterialProperties(decoded)
    Youngs_Modulus = material_properties["Youngs_Modulus"].detach().cpu().numpy()

    final_stress_diag = compute_stress_violation_diagnostics(
        vm=fe_solver.vonMisesStress,
        material_properties=material_properties,
        x_vec=xOptimal,
        active_thresh=0.5,
        mesh=fe_solver.mesh,
        bc=fe_solver.bc,
        violation_tol=0.0,
    )

    print_stress_violation_diagnostics(
        final_stress_diag,
        iteration=None,
        prefix="[FINAL STRESS DIAG]",
    )

    # Red/green local stress map: green = FF<=1, red = FF>1 (active elements only).
    fe_solver.plotter.camera_position = "xy"
    if final_stress_diag is not None:
        plot_stress_violation_distribution(
            fe_solver_structural=fe_solver,
            violation_mask_active=final_stress_diag["violation_mask_active"],
            x_vec=xOptimal,
            title="Final Local Stress Constraint Map: Green = Satisfied, Red = Violating",
            save_path=_p("01_stress_map.png"),
        )

    material_indices = matEncoder.getClosestRealMaterialIndex(zOptimalPts)
    material_names = [matEncoder.materialNames[i] for i in range(len(matEncoder.materialNames))]

    fe_solver.plotter.camera_position = "xy"
    fe_solver.plot_material_distribution(
        material_indices=material_indices.cpu().numpy()
        if hasattr(material_indices, "cpu")
        else material_indices,
        material_names=material_names,
        material_colors=material_colors,
        title="Material Distribution",
        show_legend=True,
        save_path=_p("02_material_distribution.png"),
    )

    fe_solver.mesh.setPseudoDensity(xOptimal)
    fe_solver.plotter.camera_position = "xy"
    fe_solver.plot_elem_field(
        Youngs_Modulus,
        title="YoungModulus",
        colormap="viridis",
        save_path=_p("03_youngs_modulus.png"),
    )

    # Final NON-binarized (continuous, projected) topology, grayscale, top-down.
    fe_solver.mesh.setPseudoDensity(xOptimal_cont)
    fe_solver.plotter.camera_position = "xy"
    fe_solver.plot_pseudo_density(
        save_path=_p("04_topology.png"),
        title="Final Topology (continuous density)",
    )

    matEncoder.plotLSR(
        zRealPoints.detach().cpu().numpy(),
        zOptimalPts,
        xDesign=xOptimal,
    )

    plot_reported_histories(history)
