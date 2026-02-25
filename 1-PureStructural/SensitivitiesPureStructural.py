import numpy as np
import sys
import os
common_path = os.path.join(os.path.dirname(__file__), '..', '0-Common')
sys.path.insert(0, os.path.abspath(common_path))

from PyTOImports import (get_pNorm_exponent, hex_element_stiffness, get_stress_relaxation_factor_sensitivity, 
        get_stress_relaxation_correction, linear_solvers, get_structural_material_model_sensitivity, TO_QOI,
        MaterialModel, get_structural_material_model_scaling) 


def _stable_softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    # logits shape: (N, M)
    m = np.max(logits, axis=axis, keepdims=True)
    ex = np.exp(logits - m)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def compute_material_volfrac_constraint_and_gradient_logits(
    x: np.ndarray,
    zPts: np.ndarray,
    zRealPoints: np.ndarray,
    material_id: int,
    volfracUpper: float,
    tau: float = 1.0
) -> tuple:
    """
    Material-specific volume fraction constraint using logits-softmax over distances.

    x:            (N,) density-like design variables
    zPts:         (N, latentDim) element latent points
    zRealPoints:  (M, latentDim) real material latent points (one per material)
    material_id:  int in [0, M-1]
    volfracUpper: scalar upper bound for this material volume fraction
    tau:          softmax temperature (>0). Smaller => harder assignment.
    """
    x = np.asarray(x, dtype=float)
    zPts = np.asarray(zPts, dtype=float)
    zRealPoints = np.asarray(zRealPoints, dtype=float)

    N = x.size
    M = zRealPoints.shape[0]
    if material_id < 0 or material_id >= M:
        raise ValueError(f"material_id={material_id} out of range (M={M}).")
    if tau <= 0:
        raise ValueError("tau must be > 0.")
    if volfracUpper <= 0:
        raise ValueError("volfracUpper must be > 0.")

    # logits_{e,k} = -||z_e - z_k||^2 / tau
    diff = zPts[:, None, :] - zRealPoints[None, :, :]          # (N, M, latentDim)
    dist2 = np.einsum("nmd,nmd->nm", diff, diff)               # (N, M)
    logits = -dist2 / tau                                      # (N, M)

    p = _stable_softmax(logits, axis=1)                        # (N, M)
    pk = p[:, material_id]                                     # (N,)

    # V_k = (1/N) * sum_e x_e * p_{e,k}
    Vk = np.sum(x * pk) / N
    c = Vk / volfracUpper - 1.0                                # MMA form

    # Gradient wrt x: dVk/dx_e = pk/N
    dc_dx = (pk / N) / volfracUpper                            # (N,)

    # Gradient wrt z:
    # d p_k / d z = (2/tau) * p_k * ( z_k - sum_j p_j z_j )
    z_bar = p @ zRealPoints                                    # (N, latentDim)
    z_k = zRealPoints[material_id][None, :]                    # (1, latentDim)
    dpdz = (2.0 / tau) * pk[:, None] * (z_k - z_bar)            # (N, latentDim)

    # dVk/dz_e = (x_e/N) * dpdz_e
    dVk_dz = (x[:, None] / N) * dpdz                            # (N, latentDim)
    dc_dz = dVk_dz / volfracUpper                               # (N, latentDim)

    dc_dz_flat = dc_dz.T.reshape(-1)                           

    return c, dc_dx, dc_dz_flat

# Poisson-ratio sensitivity helpers
def _find_poisson_key(material_properties: dict):
    """Return the most likely key for Poisson's ratio in decoded material properties."""
    for k in ("Poissons_Ratio", "Poisson_Ratio", "poissons_ratio", "poisson_ratio", "nu", "Nu"):
        if k in material_properties:
            return k
    return None


def isotropic_constitutive_matrix_dnu(E: float, nu: float) -> np.ndarray:
    """Analytical derivative dD/dnu for 3D isotropic linear elasticity (Voigt 6x6)."""
    nu = float(nu)

    # D = (E/((1+nu)(1-2nu))) * A(nu)
    denom = (1.0 + nu) * (1.0 - 2.0 * nu)
    c = E / denom

    # c' = d/dnu [E/((1+nu)(1-2nu))] = E*(1+4nu)/((1+nu)^2(1-2nu)^2)
    cprime = E * (1.0 + 4.0 * nu) / ((1.0 + nu) ** 2 * (1.0 - 2.0 * nu) ** 2)

    A = np.array([
        [1.0 - nu, nu,       nu,       0.0, 0.0, 0.0],
        [nu,       1.0 - nu, nu,       0.0, 0.0, 0.0],
        [nu,       nu,       1.0 - nu, 0.0, 0.0, 0.0],
        [0.0,      0.0,      0.0,      (1.0 - 2.0 * nu) / 2.0, 0.0, 0.0],
        [0.0,      0.0,      0.0,      0.0, (1.0 - 2.0 * nu) / 2.0, 0.0],
        [0.0,      0.0,      0.0,      0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
    ])

    Aprime = np.array([
        [-1.0,  1.0,  1.0,  0.0, 0.0, 0.0],
        [ 1.0, -1.0,  1.0,  0.0, 0.0, 0.0],
        [ 1.0,  1.0, -1.0,  0.0, 0.0, 0.0],
        [ 0.0,  0.0,  0.0, -1.0, 0.0, 0.0],
        [ 0.0,  0.0,  0.0,  0.0, -1.0, 0.0],
        [ 0.0,  0.0,  0.0,  0.0, 0.0, -1.0],
    ])

    return cprime * A + c * Aprime


def hex8_stiffness_matrix_structural_dnu(E: float, nu: float, elem_size, gauss_order: int = 2) -> np.ndarray:
    """Compute dKE/dnu by reusing the same Gauss integration as hex8_stiffness_matrix_structural."""
    dx, dy, dz = elem_size
    nodes = np.array([[0, dx, dx, 0, 0, dx, dx, 0],
                      [0, 0, dy, dy, 0, 0, dy, dy],
                      [0, 0, 0, 0, dz, dz, dz, dz]]).T
    dc = isotropic_constitutive_matrix_dnu(E, nu)

    dke = np.zeros((24, 24))
    gauss_pts, gauss_wts = hex_element_stiffness.get_gauss_integ_points_weights(gauss_order, 3)
    jacobians, _ = hex_element_stiffness.compute_jacobian_and_determinant(gauss_pts, nodes)
    dN_dxyz = hex_element_stiffness.get_gradient_shape_function_physical(gauss_pts, nodes)

    for ctr in range(gauss_wts.shape[0]):
        jac = jacobians[ctr, :, :]
        dn_dx, dn_dy, dn_dz = dN_dxyz[ctr, :, 0], dN_dxyz[ctr, :, 1], dN_dxyz[ctr, :, 2]
        b = np.zeros((6, 24))
        for l in range(8):
            b[0, 3 * l]     = dn_dx[l]
            b[1, 3 * l + 1] = dn_dy[l]
            b[2, 3 * l + 2] = dn_dz[l]
            b[3, 3 * l]     = dn_dy[l]
            b[3, 3 * l + 1] = dn_dx[l]
            b[4, 3 * l + 1] = dn_dz[l]
            b[4, 3 * l + 2] = dn_dy[l]
            b[5, 3 * l]     = dn_dz[l]
            b[5, 3 * l + 2] = dn_dx[l]

        dke += b.T @ dc @ b * np.linalg.det(jac) * gauss_wts[ctr]

    return dke
# --- Support Functions ---
def compute_pnorm_stress_and_sensitivities(sol: np.ndarray, x, fe_solver, EDesign, dE_dz, KETemplate, material_model,
                                           dnu_dz=None,
                                           use_constant_poissons_ratio: bool = True,
                                           constant_poissons_ratio: float = 0.3,
                                           nu_round_decimals: int = 6):
    """
    Compute p-norm von Mises stress and sensitivities w.r.t.
      (i) density-like variables x (length nelems) and
      (ii) latent variables z via decoded Young's modulus E(z).

    Returns:
        vm_pnorm: float
        d_vm_pnorm_dx: (nelems,)
        d_vm_pnorm_dz_flat: (latentDim*nelems,) in blocks of length nelems per latent dim
        max_vm: float
    """
    mesh = fe_solver.mesh
    nelems = mesh.num_elems
    pNormExponent = get_pNorm_exponent()

    # Ensure element-wise Poisson's ratio array
    if isinstance(fe_solver.mat_prop, list):
        nu = np.array([fe_solver.mat_prop[i].poissons_ratio for i in range(nelems)])
    else:
        nu = np.full(nelems, fe_solver.mat_prop.poissons_ratio)

    # Build element-wise constitutive matrices D(E,nu) and templates D0(nu) built with E=1
    D_list = []
    D0_list = []
    for Ei, nui in zip(EDesign, nu):
        D0i = hex_element_stiffness.isotropic_constitutive_matrix(1.0, float(nui))
        D0_list.append(D0i)
        D_list.append(float(Ei) * D0i)
    D = np.stack(D_list)
    D0 = np.stack(D0_list)

    # B matrix setup (match HexStructuralFEA.postprocess)
    gradN = (1 / 8) * np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1]
    ])
    for i in range(3):
        gradN[i, :] = 2 * gradN[i, :] / fe_solver.mesh.elem_size[i]
    B = np.zeros((6, 24))
    Bi = np.zeros((6, 3, 8))
    Bi[0, 0, :] = gradN[0, :]
    Bi[1, 1, :] = gradN[1, :]
    Bi[2, 2, :] = gradN[2, :]
    Bi[3, 0, :] = gradN[1, :]
    Bi[3, 1, :] = gradN[0, :]
    Bi[4, 0, :] = gradN[2, :]
    Bi[4, 2, :] = gradN[0, :]
    Bi[5, 1, :] = gradN[2, :]
    Bi[5, 2, :] = gradN[1, :]
    idx = np.arange(8)
    B[:, (3 * idx)[:, None] + np.arange(3)] = Bi.transpose(0, 2, 1)

    vm_elems = fe_solver.vonMisesStress
    vm_pnorm = fe_solver.pNormStress
    dpn_dvms = (np.sum(vm_elems ** pNormExponent)) ** (1/pNormExponent - 1)

    # DvmDs for all elements (safe division)
    DvmDs_all = np.zeros((nelems, 6))
    for e in range(nelems):
        stress_elem = fe_solver.stressComponents[e]
        sigma11, sigma22, sigma33, sigma12, sigma13, sigma23 = stress_elem
        vm_safe = max(vm_elems[e], 1e-12)
        DvmDs_all[e, 0] = 1/(2*vm_safe) * (2*sigma11 - sigma22 - sigma33)
        DvmDs_all[e, 1] = 1/(2*vm_safe) * (2*sigma22 - sigma11 - sigma33)
        DvmDs_all[e, 2] = 1/(2*vm_safe) * (2*sigma33 - sigma11 - sigma22)
        DvmDs_all[e, 3] = 3/vm_safe * sigma12
        DvmDs_all[e, 4] = 3/vm_safe * sigma13
        DvmDs_all[e, 5] = 3/vm_safe * sigma23

    x_safe = np.maximum(x, 1e-12)

    # --- Adjoint RHS (same regardless of which design variable we differentiate w.r.t.) ---
    g = np.zeros(fe_solver.bc.num_dofs)
    for e in range(nelems):
        edof = mesh.edofMatStructural[e]
        g_e = (get_stress_relaxation_correction(x_safe[e]) *
               dpn_dvms *
               (B.T @ D[e].T @ DvmDs_all[e]) *
               (vm_elems[e] ** (pNormExponent - 1)))
        g[edof] += np.asarray(g_e).reshape(-1)

    adjointSol = linear_solvers.solve(
        fe_solver.stiff_mtrx,
        g,
        fe_solver.solver,
        fe_solver.bc,
        dsolver=fe_solver.dsolver,
        **fe_solver.kwargs
    )

    # Common ce_base = adj_e^T * KE_template * u_e (per element)
    # --- Element-wise adjoint energy terms ---
    dofMat = mesh.edofMatStructural

    # Optionally force a constant Poisson's ratio for backwards-compatibility
    if use_constant_poissons_ratio:
        nu = np.full(nelems, float(constant_poissons_ratio))

    if use_constant_poissons_ratio:
        # Backwards-compatible path: assumes KE = E * KETemplate (KETemplate built with E=1 and constant nu)
        nRows = KETemplate.shape[0]
        ce_base = (np.dot(adjointSol[dofMat].reshape(nelems, nRows), KETemplate) *
                   sol[dofMat].reshape(nelems, nRows)).sum(1)  # lambda_e^T * K0 * u_e
        ce_K = EDesign * ce_base                               # lambda_e^T * KE * u_e
        ce_K0 = ce_base                                        # lambda_e^T * K0 * u_e
        ce_dK_dnu = np.zeros(nelems)                           # no nu sensitivity in constant-nu mode
    else:
        # KE = KE(E_e, nu_e) varies per element
        u_elems = sol[dofMat].reshape(nelems, 24)
        lam_elems = adjointSol[dofMat].reshape(nelems, 24)

        KE_elems = fe_solver.elem_stiff  # expected shape (nelems, 24, 24)
        if KE_elems.ndim == 2:
            # Fallback: single KE for all elements
            KE_elems = np.repeat(KE_elems[None, :, :], nelems, axis=0)

        ce_K = np.einsum('ei,eij,ej->e', lam_elems, KE_elems, u_elems)  # lambda^T * KE * u
        E_safe = np.maximum(EDesign, 1e-12)
        ce_K0 = ce_K / E_safe                                          # lambda^T * K0 * u

        # nu-indirect term needs lambda^T * (dKE/dnu) * u
        ce_dK_dnu = np.zeros(nelems)
        elem_size = (float(mesh.elem_size[0]), float(mesh.elem_size[1]), float(mesh.elem_size[2]))

        # Cache dKE0/dnu by (rounded) nu
        nu_keys = np.round(nu.astype(float), int(nu_round_decimals))
        unique_nu = np.unique(nu_keys)

        dKE0_cache = {}
        for nu_k in unique_nu:
            nu_val = float(nu_k)
            if nu_val not in dKE0_cache:
                dKE0_cache[nu_val] = hex8_stiffness_matrix_structural_dnu(1.0, nu_val, elem_size)
            dKE0_dnu = dKE0_cache[nu_val]

            idxs = np.where(nu_keys == nu_k)[0]
            lam_g = lam_elems[idxs]
            u_g = u_elems[idxs]
            ce0 = np.einsum('ei,ij,ej->e', lam_g, dKE0_dnu, u_g)  # lambda^T * dKE0/dnu * u
            ce_dK_dnu[idxs] = EDesign[idxs] * ce0                 # dKE/dnu = E * dKE0/dnu

    # --- Sensitivity w.r.t. x (density-like) ---
    beta_x = np.zeros(nelems)
    for e in range(nelems):
        edof = mesh.edofMatStructural[e]
        u_e = sol[edof]
        beta_x[e] = (get_stress_relaxation_factor_sensitivity(x_safe[e]) *
                     (vm_elems[e] ** (pNormExponent - 1)) *
                     (DvmDs_all[e] @ D[e] @ B @ u_e))
    T1_x = dpn_dvms * beta_x

    # Indirect term: dK/dx = d(scaling)/dx * KE
    T2_x = -get_structural_material_model_sensitivity(x_safe, material_model) * ce_K
    d_vm_pnorm_dx = T1_x + T2_x

    # --- Sensitivity w.r.t. E ---
    beta_E = np.zeros(nelems)
    for e in range(nelems):
        edof = mesh.edofMatStructural[e]
        u_e = sol[edof]
        beta_E[e] = (get_stress_relaxation_correction(x_safe[e]) *
                     (vm_elems[e] ** (pNormExponent - 1)) *
                     (DvmDs_all[e] @ D0[e] @ B @ u_e))
    T1_E = dpn_dvms * beta_E

    # Indirect term: dK/dE = scaling(x) * K0
    T2_E = -get_structural_material_model_scaling(x_safe, material_model) * ce_K0
    d_vm_pnorm_dE = T1_E + T2_E

    # --- Sensitivity w.r.t. nu ---
    if use_constant_poissons_ratio:
        d_vm_pnorm_dnu = np.zeros(nelems)
    else:
        beta_nu = np.zeros(nelems)
        for e in range(nelems):
            edof = mesh.edofMatStructural[e]
            u_e = sol[edof]
            dD0_dnu = isotropic_constitutive_matrix_dnu(1.0, float(nu[e]))
            beta_nu[e] = (get_stress_relaxation_correction(x_safe[e]) *
                          (vm_elems[e] ** (pNormExponent - 1)) *
                          (DvmDs_all[e] @ (EDesign[e] * dD0_dnu) @ B @ u_e))
        T1_nu = dpn_dvms * beta_nu
        T2_nu = -get_structural_material_model_scaling(x_safe, material_model) * ce_dK_dnu
        d_vm_pnorm_dnu = T1_nu + T2_nu

    # Chain to latent (dE_dz and optionally dnu_dz)
    if (dE_dz is None) and (dnu_dz is None):
        d_vm_pnorm_dz_flat = np.zeros(0)
    else:
        latentDim = (dE_dz.shape[0] if dE_dz is not None else dnu_dz.shape[0])
        termE = (dE_dz * d_vm_pnorm_dE[None, :]) if dE_dz is not None else np.zeros((latentDim, nelems))
        termNu = (dnu_dz * d_vm_pnorm_dnu[None, :]) if dnu_dz is not None else np.zeros((latentDim, nelems))
        d_vm_pnorm_dz_flat = (termE + termNu).reshape(-1)

    max_vm = float(np.max(vm_elems))
    return vm_pnorm, d_vm_pnorm_dx, d_vm_pnorm_dz_flat, max_vm



def compute_pnorm_failure_factor_and_sensitivities(sol: np.ndarray, x, fe_solver, EDesign, YDesign, dE_dz, dY_dz, KETemplate, material_model,
                                                   dnu_dz=None,
                                                   use_constant_poissons_ratio: bool = True,
                                                   constant_poissons_ratio: float = 0.3,
                                                   nu_round_decimals: int = 6):

    """
    Compute p-norm failure factor (inverse safety factor) and sensitivities w.r.t.
      (i) x (density-like) and
      (ii) latent z via E(z) and Y(z).

    Returns:
        ff_pnorm: float
        d_ff_pnorm_dx: (nelems,)
        d_ff_pnorm_dz_flat: (latentDim*nelems,)
        max_ff: float
    """
    mesh = fe_solver.mesh
    nelems = mesh.num_elems
    pNormExponent = get_pNorm_exponent()

    # Ensure element-wise Poisson's ratio array
    if isinstance(fe_solver.mat_prop, list):
        nu = np.array([fe_solver.mat_prop[i].poissons_ratio for i in range(nelems)])
    else:
        nu = np.full(nelems, fe_solver.mat_prop.poissons_ratio)

    # Build D(E,nu) and D0(nu)
    D_list = []
    D0_list = []
    for Ei, nui in zip(EDesign, nu):
        D0i = hex_element_stiffness.isotropic_constitutive_matrix(1.0, float(nui))
        D0_list.append(D0i)
        D_list.append(float(Ei) * D0i)
    D = np.stack(D_list)
    D0 = np.stack(D0_list)

    # B matrix setup (match HexStructuralFEA.postprocess)
    gradN = (1 / 8) * np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1]
    ])
    for i in range(3):
        gradN[i, :] = 2 * gradN[i, :] / fe_solver.mesh.elem_size[i]
    B = np.zeros((6, 24))
    Bi = np.zeros((6, 3, 8))
    Bi[0, 0, :] = gradN[0, :]
    Bi[1, 1, :] = gradN[1, :]
    Bi[2, 2, :] = gradN[2, :]
    Bi[3, 0, :] = gradN[1, :]
    Bi[3, 1, :] = gradN[0, :]
    Bi[4, 0, :] = gradN[2, :]
    Bi[4, 2, :] = gradN[0, :]
    Bi[5, 1, :] = gradN[2, :]
    Bi[5, 2, :] = gradN[1, :]
    idx = np.arange(8)
    B[:, (3 * idx)[:, None] + np.arange(3)] = Bi.transpose(0, 2, 1)

    # Inverse safety factor per element (sigma_vm already includes stress relaxation correction)
    sigma_vm = np.maximum(fe_solver.vonMisesStress, 1e-12)
    Y_safe = np.maximum(YDesign, 1e-12)
    inv_sf_elems = sigma_vm / Y_safe

    ff_pnorm = (np.sum(inv_sf_elems ** pNormExponent) / nelems) ** (1.0 / pNormExponent)
    dpn_dinv = ((np.sum(inv_sf_elems ** pNormExponent) / nelems) ** (1.0 / pNormExponent - 1)) / nelems

    # D(inv_sf)/D(stress) = (1/Y) * D(vm)/D(stress)
    DinvSfDs_all = np.zeros((nelems, 6))
    for e in range(nelems):
        stress_elem = fe_solver.stressComponents[e]
        sigma11, sigma22, sigma33, sigma12, sigma13, sigma23 = stress_elem
        denom = max(fe_solver.vonMisesStress[e], 1e-12) * Y_safe[e]
        DinvSfDs_all[e, 0] = 1/(2*denom) * (2*sigma11 - sigma22 - sigma33)
        DinvSfDs_all[e, 1] = 1/(2*denom) * (2*sigma22 - sigma11 - sigma33)
        DinvSfDs_all[e, 2] = 1/(2*denom) * (2*sigma33 - sigma11 - sigma22)
        DinvSfDs_all[e, 3] = 3/denom * sigma12
        DinvSfDs_all[e, 4] = 3/denom * sigma13
        DinvSfDs_all[e, 5] = 3/denom * sigma23

    # --- Adjoint RHS ---
    x_safe = np.maximum(x, 1e-12)
    g = np.zeros(fe_solver.bc.num_dofs)
    for e in range(nelems):
        edof = mesh.edofMatStructural[e]
        g_e = (get_stress_relaxation_correction(x_safe[e]) *
               dpn_dinv *
               (B.T @ D[e].T @ DinvSfDs_all[e]) *
               (inv_sf_elems[e] ** (pNormExponent - 1)))
        g[edof] += np.asarray(g_e).reshape(-1)

    adjointSol = linear_solvers.solve(
        fe_solver.stiff_mtrx,
        g,
        fe_solver.solver,
        fe_solver.bc,
        dsolver=fe_solver.dsolver,
        **fe_solver.kwargs
    )

    # Common ce_base = adj_e^T * KE_template * u_e
    # --- Element-wise adjoint energy terms ---
    dofMat = mesh.edofMatStructural

    # Optionally force a constant Poisson's ratio for backwards-compatibility
    if use_constant_poissons_ratio:
        nu = np.full(nelems, float(constant_poissons_ratio))

    if use_constant_poissons_ratio:
        nRows = KETemplate.shape[0]
        ce_base = (np.dot(adjointSol[dofMat].reshape(nelems, nRows), KETemplate) *
                   sol[dofMat].reshape(nelems, nRows)).sum(1)  # lambda^T * K0 * u
        ce_K = EDesign * ce_base
        ce_K0 = ce_base
        ce_dK_dnu = np.zeros(nelems)
    else:
        u_elems = sol[dofMat].reshape(nelems, 24)
        lam_elems = adjointSol[dofMat].reshape(nelems, 24)

        KE_elems = fe_solver.elem_stiff
        if KE_elems.ndim == 2:
            KE_elems = np.repeat(KE_elems[None, :, :], nelems, axis=0)

        ce_K = np.einsum('ei,eij,ej->e', lam_elems, KE_elems, u_elems)
        E_safe = np.maximum(EDesign, 1e-12)
        ce_K0 = ce_K / E_safe

        ce_dK_dnu = np.zeros(nelems)
        elem_size = (float(mesh.elem_size[0]), float(mesh.elem_size[1]), float(mesh.elem_size[2]))
        nu_keys = np.round(nu.astype(float), int(nu_round_decimals))
        unique_nu = np.unique(nu_keys)
        dKE0_cache = {}
        for nu_k in unique_nu:
            nu_val = float(nu_k)
            if nu_val not in dKE0_cache:
                dKE0_cache[nu_val] = hex8_stiffness_matrix_structural_dnu(1.0, nu_val, elem_size)
            dKE0_dnu = dKE0_cache[nu_val]
            idxs = np.where(nu_keys == nu_k)[0]
            lam_g = lam_elems[idxs]
            u_g = u_elems[idxs]
            ce0 = np.einsum('ei,ij,ej->e', lam_g, dKE0_dnu, u_g)
            ce_dK_dnu[idxs] = EDesign[idxs] * ce0

    # --- Sensitivity w.r.t. x ---
    beta_x = np.zeros(nelems)
    for e in range(nelems):
        edof = mesh.edofMatStructural[e]
        u_e = sol[edof]
        beta_x[e] = (get_stress_relaxation_factor_sensitivity(x_safe[e]) *
                     (inv_sf_elems[e] ** (pNormExponent - 1)) *
                     (DinvSfDs_all[e] @ D[e] @ B @ u_e))
    T1_x = dpn_dinv * beta_x
    T2_x = -get_structural_material_model_sensitivity(x_safe, material_model) * ce_K
    d_ff_pnorm_dx = T1_x + T2_x

    # --- Sensitivity w.r.t. E ---
    beta_E = np.zeros(nelems)
    for e in range(nelems):
        edof = mesh.edofMatStructural[e]
        u_e = sol[edof]
        beta_E[e] = (get_stress_relaxation_correction(x_safe[e]) *
                     (inv_sf_elems[e] ** (pNormExponent - 1)) *
                     (DinvSfDs_all[e] @ D0[e] @ B @ u_e))
    T1_E = dpn_dinv * beta_E
    T2_E = -get_structural_material_model_scaling(x_safe, material_model) * ce_K0
    d_ff_pnorm_dE = T1_E + T2_E

    # --- Sensitivity w.r.t. nu ---
    if use_constant_poissons_ratio:
        d_ff_pnorm_dnu = np.zeros(nelems)
    else:
        beta_nu = np.zeros(nelems)
        for e in range(nelems):
            edof = mesh.edofMatStructural[e]
            u_e = sol[edof]
            dD0_dnu = isotropic_constitutive_matrix_dnu(1.0, float(nu[e]))
            beta_nu[e] = (get_stress_relaxation_correction(x_safe[e]) *
                          (inv_sf_elems[e] ** (pNormExponent - 1)) *
                          (DinvSfDs_all[e] @ (EDesign[e] * dD0_dnu) @ B @ u_e))
        T1_nu = dpn_dinv * beta_nu
        T2_nu = -get_structural_material_model_scaling(x_safe, material_model) * ce_dK_dnu
        d_ff_pnorm_dnu = T1_nu + T2_nu


    # --- Sensitivity w.r.t. Y (no adjoint term; K does not depend on Y) ---
    d_inv_dY = -sigma_vm / (Y_safe ** 2)
    d_ff_pnorm_dY = dpn_dinv * (inv_sf_elems ** (pNormExponent - 1)) * d_inv_dY

 
    # Chain to latent
    if (dE_dz is None) and (dY_dz is None) and (dnu_dz is None):
        d_ff_pnorm_dz_flat = np.zeros(0)
    else:
        latentDim = (dE_dz.shape[0] if dE_dz is not None else (dY_dz.shape[0] if dY_dz is not None else dnu_dz.shape[0]))
        termE  = (dE_dz  * d_ff_pnorm_dE[None, :])  if dE_dz  is not None else np.zeros((latentDim, nelems))
        termY  = (dY_dz  * d_ff_pnorm_dY[None, :])  if dY_dz  is not None else np.zeros((latentDim, nelems))
        termNu = (dnu_dz * d_ff_pnorm_dnu[None, :]) if dnu_dz is not None else np.zeros((latentDim, nelems))
        d_ff_pnorm_dz_flat = (termE + termY + termNu).reshape(-1)


    max_ff = float(np.max(inv_sf_elems))
    return ff_pnorm, d_ff_pnorm_dx, d_ff_pnorm_dz_flat, max_ff


def compute_volumefraction_constraint_and_gradient(x: np.ndarray, volfracUpper: float) -> tuple:
    volFracConstraint = ((np.mean(x)/volfracUpper) - 1.0)
    volFracConstraint_gradient = np.ones_like(x) / volfracUpper/ x.size
    return volFracConstraint, volFracConstraint_gradient


# --- Main Objective/Constraint Functions ---
def compute_mmto_objective_and_gradient(to_params, sol, zeta, fe_solver, KETemplate, matEncoder,
                                        use_constant_poissons_ratio: bool = True,
                                        constant_poissons_ratio: float = 0.3):

    """
    Compute objective value and its gradient for METALS LSR.
    Handles:
      1. Compliance objective (subject to mass)
      2. Compliance objective (subject to mass and cost)
      3. Mass objective (subject to yield strength and compliance)
    """
    objectiveType = to_params.Objective[0]
    optionalParam = to_params.Objective[1]
    num_elems = fe_solver.mesh.num_elems
    x = zeta[0:fe_solver.mesh.num_elems]

    material_model = to_params.materialModel
    latentDim = matEncoder.vae_params.latentDim
    zPts = zeta[num_elems:].reshape((latentDim, -1)).T
    material_properties, gradients = matEncoder.getMaterialPropertiesAtLatentPoints(zPts, compute_gradients=True)

    if 'Youngs_Modulus' in material_properties:
        EDesign = material_properties['Youngs_Modulus'].detach().numpy()
        dE_dz = gradients['Youngs_Modulus'].detach().numpy().T
    else:
        EDesign = None
        dE_dz = None
    # --- Poisson's ratio (nu) handling ---
    use_const_nu = bool(use_constant_poissons_ratio)
    const_nu = float(constant_poissons_ratio)
    print(f"[SENS nu mode] use_constant={use_const_nu}, const_nu={const_nu}")

    nu_key = _find_poisson_key(material_properties)
    if use_const_nu or (nu_key is None):
        nuDesign = np.full(num_elems, float(const_nu))
        dnu_dz = None
    else:
        nuDesign = material_properties[nu_key].detach().numpy()
        dnu_dz = gradients[nu_key].detach().numpy().T

    if 'Density' in material_properties:
        mass_density = material_properties['Density'].detach().numpy()
        dMassDensity_dz = gradients['Density'].detach().numpy().T
    else:
        mass_density = None
        dMassDensity_dz = None

    pNormExponent = get_pNorm_exponent()
    if objectiveType == TO_QOI.COMPLIANCE:
        compliance = np.einsum('i, i -> ', fe_solver.total_force, sol)
        ce = (np.dot(sol[fe_solver.mesh.edofMatStructural].reshape(num_elems, 24), KETemplate) * sol[fe_solver.mesh.edofMatStructural].reshape(num_elems, 24)).sum(1)
        dJ_dxDesign = (-get_structural_material_model_sensitivity(x, material_model)) * EDesign * ce
        dJ_dEDesign = np.asarray((get_structural_material_model_scaling(x, material_model)) * ce)
        dJ_dzeta = (dJ_dEDesign * dE_dz).flatten()
        grad_compliance = np.concatenate((dJ_dxDesign, -dJ_dzeta))
        return compliance, grad_compliance
    elif objectiveType == TO_QOI.PNORM_STRESS:
        # P-norm von Mises stress objective
        pNormExponent = get_pNorm_exponent()

        # Density + latent sensitivities in a single call (one adjoint solve)
        vm_pnorm, grad_vm_density, grad_vm_z, max_vm = compute_pnorm_stress_and_sensitivities(
            sol, x, fe_solver, EDesign, dE_dz, KETemplate, material_model,
            dnu_dz=dnu_dz,
            use_constant_poissons_ratio=use_const_nu,
            constant_poissons_ratio=const_nu,
)


        # Objective gradient w.r.t zeta = [x; z]
        grad_pnorm_stress = np.zeros_like(zeta)
        grad_pnorm_stress[:num_elems] = grad_vm_density
        grad_pnorm_stress[num_elems:] = grad_vm_z

        return vm_pnorm, grad_pnorm_stress
    elif objectiveType == TO_QOI.MASS:
        elemVolume =  fe_solver.mesh.elem_size[0] * fe_solver.mesh.elem_size[1] * fe_solver.mesh.elem_size[2]
        totalMass = np.einsum('m,m->m', mass_density, x).sum()*elemVolume
        grad_mass = np.concatenate((mass_density*elemVolume, (dMassDensity_dz*x*elemVolume).flatten()))
        return totalMass, grad_mass
    else:
        raise NotImplementedError(f"Objective {objectiveType} is not implemented yet.")


def compute_mmto_constraint_and_gradient(to_params, sol, zeta, fe_solver, KETemplate, matEncoder,
                                         use_constant_poissons_ratio: bool = True,
                                         constant_poissons_ratio: float = 0.3):

    """
    Compute constraint values and their gradients for METALS LSR.
    Handles:
      1. Compliance constraint
      2. Mass constraint
      3. Cost constraint
      4. Yield strength/safety factor constraint
    """
    nConstraints = len(to_params.Constraints)
    num_elems = fe_solver.mesh.num_elems
    x = zeta[0:num_elems]
    latentDim = matEncoder.vae_params.latentDim

    zPts = zeta[num_elems:].reshape((latentDim, -1)).T
    material_properties, gradients = matEncoder.getMaterialPropertiesAtLatentPoints(zPts, compute_gradients=True)
    material_model = to_params.materialModel

    if 'Youngs_Modulus' in material_properties:
        EDesign = material_properties['Youngs_Modulus'].detach().numpy()
        dE_dz = gradients['Youngs_Modulus'].detach().numpy().T
    else:
        EDesign = None
        dE_dz = None
    # --- Poisson's ratio (nu) handling ---
    use_const_nu = bool(use_constant_poissons_ratio)
    const_nu = float(constant_poissons_ratio)
    print(f"[SENS nu mode] use_constant={use_const_nu}, const_nu={const_nu}")

    nu_key = _find_poisson_key(material_properties)
    if use_const_nu or (nu_key is None):
        nuDesign = np.full(num_elems, float(const_nu))
        dnu_dz = None
    else:
        nuDesign = material_properties[nu_key].detach().numpy()
        dnu_dz = gradients[nu_key].detach().numpy().T

    if 'Density' in material_properties:
        mass_density = material_properties['Density'].detach().numpy()
        dMassDensity_dz = gradients['Density'].detach().numpy().T
    else:
        mass_density = None
        dMassDensity_dz = None

    if 'Cost' in material_properties:
        cost_per_unitmass = material_properties['Cost'].detach().numpy()
        dCost_dz = gradients['Cost'].detach().numpy().T
    else:
        cost_per_unitmass = None
        dCost_dz = None

    if 'Criticality' in material_properties:
        criticality = material_properties['Criticality'].detach().numpy()
        dCriticality_dz = gradients['Criticality'].detach().numpy().T
    else:
        criticality = None
        dCriticality_dz = None

    if 'Yield_Strength' in material_properties:
        YDesign = material_properties['Yield_Strength'].detach().numpy()
        dY_dz = gradients['Yield_Strength'].detach().numpy().T
    else:
        YDesign = None
        dY_dz = None

    pNormExponent = get_pNorm_exponent()

    c = np.zeros((nConstraints, 1))
    num_design_var = zeta.size
    dc = np.zeros((nConstraints, num_design_var))
    for m in range(nConstraints):
        constraintType = to_params.Constraints[m][0]
        optionalParam = to_params.Constraints[m][1]
        constraintLimit = to_params.Constraints[m][2]
        if constraintType == TO_QOI.COMPLIANCE:
            compliance = np.einsum('i, i -> ', fe_solver.total_force, sol)
            ce = (np.dot(sol[fe_solver.mesh.edofMatStructural].reshape(num_elems, 24), KETemplate) * sol[fe_solver.mesh.edofMatStructural].reshape(num_elems, 24)).sum(1)
            dJ_dxDesign = (-get_structural_material_model_sensitivity(x, material_model)) * EDesign * ce
            dJ_dEDesign = np.asarray((get_structural_material_model_scaling(x, material_model)) * ce)
            dJ_dz = (dJ_dEDesign * dE_dz).flatten()
            grad_compliance = np.concatenate((dJ_dxDesign, -dJ_dz))
            c[m, 0] = compliance/constraintLimit-1
            dc[m, :] = grad_compliance / constraintLimit
        elif constraintType == TO_QOI.VOLUME_FRACTION:
            # Two cases:
            # (A) global volume fraction: optionalParam is None (original behavior)
            # (B) material-specific volume fraction via logits-softmax:
            #     optionalParam = {"material_id":k, "tau":...}

            if isinstance(optionalParam, dict) and ("material_id" in optionalParam):
                # get real material latent points (one per material)
                if not hasattr(matEncoder, "training_latents") or (matEncoder.training_latents is None):
                    raise ValueError("matEncoder.training_latents must be set to (nMaterials, latentDim) latent points.")

                training_latents = matEncoder.training_latents
                if hasattr(training_latents, "detach"):
                    zRealPoints = training_latents.detach().cpu().numpy()
                else:
                    zRealPoints = np.asarray(training_latents, dtype=float)

                # Specify it like this in ExamplesPureStructural.py:
                # to_params.Constraints = [
                #     (TO_QOI.VOLUME_FRACTION, {"material_id":0, "tau":0.5}, 0.12),
                #     (TO_QOI.VOLUME_FRACTION, {"material_id":1, "tau":0.5}, 0.10),
                #     (TO_QOI.VOLUME_FRACTION, {"material_id":2, "tau":0.5}, 0.08),
                # ]

                material_id = int(optionalParam["material_id"])
                tau = float(optionalParam.get("tau", 1.0))

                c_val, dc_dx, dc_dz_flat = compute_material_volfrac_constraint_and_gradient_logits(
                    x=x,
                    zPts=zPts,
                    zRealPoints=zRealPoints,
                    material_id=material_id,
                    volfracUpper=float(constraintLimit),
                    tau=tau
                )

                grad_volfrac = np.zeros_like(zeta)
                grad_volfrac[0:num_elems] = dc_dx
                grad_volfrac[num_elems:] = dc_dz_flat

                c[m, 0] = c_val
                dc[m, :] = grad_volfrac
            else:
                # Original global volume fraction constraint
                volfracConstraint, volfracConstraint_gradient = compute_volumefraction_constraint_and_gradient(
                    x, constraintLimit
                )
                grad_volfracConstraint = np.zeros_like(zeta)
                grad_volfracConstraint[0:num_elems] = volfracConstraint_gradient
                c[m, 0] = volfracConstraint
                dc[m, :] = grad_volfracConstraint

        elif constraintType == TO_QOI.MASS:
            elemVolume =  fe_solver.mesh.elem_size[0] * fe_solver.mesh.elem_size[1] * fe_solver.mesh.elem_size[2]
            totalMass =  (mass_density * x).sum()*elemVolume
            grad_mass = np.concatenate((mass_density*elemVolume, (dMassDensity_dz*x*elemVolume).flatten()))
            c[m, 0] = totalMass/constraintLimit - 1.0
            dc[m, :] = grad_mass / constraintLimit
        elif constraintType == TO_QOI.COST:
            elemVolume =  fe_solver.mesh.elem_size[0] * fe_solver.mesh.elem_size[1] * fe_solver.mesh.elem_size[2]
            totalCost =  (mass_density* cost_per_unitmass * x).sum()*elemVolume
            dTotalCost_dz = (dMassDensity_dz * cost_per_unitmass + mass_density * dCost_dz)*x*elemVolume
            dTotalCost_dx = mass_density * cost_per_unitmass * elemVolume
            grad_cost = np.concatenate((dTotalCost_dx, (dTotalCost_dz).flatten()))
            c[m, 0] = totalCost/constraintLimit - 1.0
            dc[m, :] = grad_cost / constraintLimit
        elif constraintType == TO_QOI.MAX_CRITICALITY:
            pnorm_criticality = np.sum(criticality ** pNormExponent) ** (1.0 / pNormExponent)
            maxCriticality = pnorm_criticality
            dpnorm_dcriticality = (criticality ** (pNormExponent - 1)) / (pnorm_criticality ** (pNormExponent - 1))
            dpnorm_dz = dCriticality_dz * dpnorm_dcriticality[np.newaxis, :]  # Broadcasting
            grad_crit_z = dpnorm_dz.flatten()
            grad_cons_criticality = np.zeros_like(zeta)
            grad_cons_criticality[num_elems:] = grad_crit_z / constraintLimit
            c[m, 0] = ((maxCriticality / constraintLimit) - 1.0)
            dc[m, :] = grad_cons_criticality
        elif constraintType == TO_QOI.MEAN_CRITICALITY:
            # Compute mean criticality
            mean_criticality = np.mean(criticality)
            grad_cons_mean_criticality = np.zeros_like(zeta)
            grad_cons_mean_criticality[num_elems:] = dCriticality_dz.flatten() / constraintLimit/len(criticality)
            c[m, 0] = ((mean_criticality / constraintLimit) - 1.0)
            dc[m, :] = grad_cons_mean_criticality
        elif constraintType == TO_QOI.STRESS_FAILURE_FACTOR:
            # Agglomerated (p-norm) failure factor = inverse safety factor
            ff_pNorm, grad_ff_density, grad_ff_z, max_inv_sf = compute_pnorm_failure_factor_and_sensitivities(
                sol, x, fe_solver, EDesign, YDesign, dE_dz, dY_dz, KETemplate, material_model,
                dnu_dz=dnu_dz,
                use_constant_poissons_ratio=use_const_nu,
                constant_poissons_ratio=const_nu,
            )


            # Standard MMA inequality form: g(zeta) = FF_PN/FF_limit - 1 <= 0
            safety_constraint = ff_pNorm / constraintLimit - 1.0

            grad_stress_ff = np.zeros_like(zeta)
            grad_stress_ff[:num_elems] = grad_ff_density / constraintLimit
            grad_stress_ff[num_elems:] = grad_ff_z / constraintLimit

            c[m] = safety_constraint
            dc[m, :] = grad_stress_ff
        else:
            raise NotImplementedError(f"Constraint {constraintType} is not implemented yet.")

    return c, dc
