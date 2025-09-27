import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from LSRImports import *
import hex_element_stiffness
import math
def plot_patches(mesh, nPatchesDesired=8, title_prefix="Patchwork Coloring"):
    patchwork_colors = patchwork(mesh, nPatchesDesired=nPatchesDesired)
    elem_centers = mesh.elem_centers
    num_patches = len(np.unique(patchwork_colors))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    if num_patches > 50:
        np.random.seed(42)
        colors = []
        for i in range(num_patches):
            hue = (i * 137.508) % 360
            sat = 0.6 + 0.4 * (i % 3) / 2
            val = 0.7 + 0.3 * ((i // 3) % 3) / 2
            rgb = matplotlib.colors.hsv_to_rgb([hue/360, sat, val])
            colors.append(rgb)
        colors = np.array(colors)
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)
    else:
        cmap = cm.get_cmap('nipy_spectral', num_patches)
    sc = ax.scatter(
        elem_centers[:, 0], elem_centers[:, 1], elem_centers[:, 2],
        c=patchwork_colors, cmap=cmap, s=40
    )
    plt.title(f"{title_prefix} ({num_patches} patches)", fontsize=18)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.colorbar(sc, label="Patch ID")
    ax.set_box_aspect([
        np.ptp(elem_centers[:, 0]),
        np.ptp(elem_centers[:, 1]),
        np.ptp(elem_centers[:, 2])
    ])
    plt.tight_layout()
    plt.show()
    return patchwork_colors

def patchwork(mesh, nPatchesDesired=8):
    if (nPatchesDesired is None or 
        nPatchesDesired < 1 or 
        nPatchesDesired >= mesh.num_elems):
        print(f"nPatchesDesired ({nPatchesDesired}) is None, < 1, or >= num_elems ({mesh.num_elems}). Each element will be its own patch (no patching).")
        return np.arange(mesh.num_elems, dtype=np.int32)
    xyz = mesh.elem_centers
    xMin = np.min(xyz[:,0])
    yMin = np.min(xyz[:,1])
    zMin = np.min(xyz[:,2])
    xLength = np.max(xyz[:,0]) - xMin
    yLength = np.max(xyz[:,1]) - yMin
    zLength = np.max(xyz[:,2]) - zMin

    if (zLength < 1e-12):
        print("2D problem detected (zLength is negligible). Using 2D patching.")
        zLength = 1.0
        temp = xLength * yLength 
    else:
        temp = xLength * yLength * zLength
    alpha = (nPatchesDesired / temp) ** (1.0 / 3)
    print(f"Calculated alpha={alpha} for nPatchesDesired={nPatchesDesired}.")
    nX = max(round(alpha*xLength), 1)
    nY = max(round(alpha*yLength), 1)
    nZ = max(round(alpha*zLength), 1)
    print(f"Dividing domain into nX={nX}, nY={nY}, nZ={nZ} patches.")
    nPatchesTentative = nX * nY * nZ
    sizeX = xLength / nX
    sizeY = yLength / nY
    sizeZ = zLength / nZ
    rel_pos = xyz - np.array([xMin, yMin, zMin])
    indices = np.floor(rel_pos / np.array([sizeX, sizeY, sizeZ])).astype(np.int32)
    indices = np.minimum(indices, np.array([nX - 1, nY - 1, nZ - 1]))
    elemPatchNumber = (indices[:, 0] + nX * indices[:, 1] + nX * nY * indices[:, 2]).astype(np.int32)
    # Count number of elements in each patch
    unique, counts = np.unique(elemPatchNumber, return_counts=True)
    num_elems_per_patch = np.zeros(np.max(elemPatchNumber) + 1, dtype=int)
    num_elems_per_patch[unique] = counts
  
    # Remove empty patches and renumber so patch numbers are contiguous
    unique_patches, inverse_indices = np.unique(elemPatchNumber, return_inverse=True)
    elemPatchNumber = inverse_indices.astype(np.int32)

    print(elemPatchNumber)
    return elemPatchNumber

def compute_pnorm_safety_factor_and_sensitivity(sol: np.ndarray, x, fe_solver, KE, material_model, p=6):
    """
    Compute p-norm of (von Mises stress / yield strength) and its sensitivity for multi-material case.
    """
    mesh = fe_solver.mesh
    nelems = mesh.num_elems
    q = 1  # STRESS_RELAXATION factor

    # Handle multi-material: get yield strength for each element
    if isinstance(fe_solver.mat_prop, list):
        # Use elemComponentId if it exists, otherwise default to zeros
        if hasattr(mesh, "elemComponentId"):
            elem_ids = mesh.elemComponentId
        else:
            elem_ids = np.zeros(mesh.num_elems, dtype=int)
        yield_strengths = np.array([fe_solver.mat_prop[i].yield_strength for i in elem_ids])
        E = np.array([fe_solver.mat_prop[i].youngs_modulus for i in elem_ids])
        nu = np.array([fe_solver.mat_prop[i].poissons_ratio for i in elem_ids])
        D_list = []
        for Ei, nui in zip(E, nu):
            D = hex_element_stiffness.isotropic_constitutive_matrix ( Ei, nui)
            D_list.append(D)
        D_stack = np.stack(D_list)
    else:
        yield_strengths = np.full(nelems, fe_solver.mat_prop.yield_strength)
        E = fe_solver.mat_prop.youngs_modulus
        nu = fe_solver.mat_prop.poissons_ratio
        D =  hex_element_stiffness.isotropic_constitutive_matrix ( E, nu)


    gradN = (1 / 8) * np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1]
    ])
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
    # F can be per-element for multi-material
    if isinstance(E, np.ndarray):
        F_stack = np.array([D_stack[e] @ B for e in range(nelems)])
    else:
        F = D @ B

    g_elem = np.zeros((nelems, 24))
    inv_sf_elems = np.zeros(nelems)
    T1 = np.zeros(nelems)
    T2 = np.zeros(nelems)

    for e in range(nelems):
        # Stress for T1 (no relaxation)
        stress_elem = fe_solver.stressComponents[e]
        sigma11, sigma22, sigma33, sigma12, sigma13, sigma23 = stress_elem
        vm = np.sqrt(
            0.5 * ((sigma11 - sigma22) ** 2 + (sigma22 - sigma33) ** 2 + (sigma33 - sigma11) ** 2)
            + 3 * (sigma12 ** 2 + sigma13 ** 2 + sigma23 ** 2)
        )
        inv_sf = vm / yield_strengths[e]
        T1[e] = p * q * (x[e] ** (p * q - 1)) * inv_sf

        # Stress for T2 (with relaxation)
        stress_elem_relaxed = (x[e] ** q) * fe_solver.stressComponents[e]
        sigma11, sigma22, sigma33, sigma12, sigma13, sigma23 = stress_elem_relaxed
        vm_relaxed = np.sqrt(
            0.5 * ((sigma11 - sigma22) ** 2 + (sigma22 - sigma33) ** 2 + (sigma33 - sigma11) ** 2)
            + 3 * (sigma12 ** 2 + sigma13 ** 2 + sigma23 ** 2)
        )
        inv_sf_elems[e] = vm_relaxed / yield_strengths[e]

        if isinstance(E, np.ndarray):
            F = F_stack[e]
        # Sensitivity of von Mises stress w.r.t. displacement
        g_e = (
            (sigma11 - sigma22) * (F[0] - F[1])
            + (sigma11 - sigma33) * (F[0] - F[2])
            + (sigma22 - sigma33) * (F[1] - F[2])
            + 6 * sigma12 * F[3]
            + 6 * sigma13 * F[4]
            + 6 * sigma23 * F[5]
        ) / np.sqrt(2)
        g_elem[e] = p * (inv_sf_elems[e] ** (p - 2)) * g_e / yield_strengths[e]

    inv_sf_pnorm = np.sum(inv_sf_elems ** p) ** (1 / p)
    T1 *= (1 / p) * (np.sum(inv_sf_elems ** p) ** (1 / p - 1))

    # Assemble adjoint RHS
    g = np.zeros(fe_solver.bc.num_dofs)
    for e in range(nelems):
        edof = mesh.edofMat[e]
        g[edof] += g_elem[e]
    g *= -(1 / p) * (np.sum(inv_sf_elems ** p) ** (1 / p - 1))

    adjointSol = linear_solvers.solve(
        fe_solver.stiff_mtrx,
        g,
        fe_solver.solver,
        fe_solver.bc,
        dsolver=fe_solver.dsolver,
        **fe_solver.kwargs
    )

    dofMat = fe_solver.mesh.edofMat
    num_elems = fe_solver.mesh.num_elems
    nRows = KE.shape[0]
    ce = (
        np.dot(adjointSol[dofMat].reshape(num_elems, nRows), KE)
        * sol[dofMat].reshape(num_elems, nRows)
    ).sum(1)

    T2 = get_structural_material_model_sensitivity(x, material_model) * ce
    inv_sf_pnorm_sensitivity = T1 + T2

    return inv_sf_pnorm, inv_sf_pnorm_sensitivity

def d_relaxed_von_mises_dE(stress, x, q=1):
    """
    Compute derivative of relaxed von Mises stress with respect to Young's modulus E for a single element.
    stress: (6,) array-like [sxx, syy, szz, syz, sxz, sxy] (unrelaxed)
    x: density variable for the element
    q: stress relaxation exponent (default 1)
    Returns: scalar d(sigma_vm_relaxed)/dE
    """
    sxx, syy, szz, syz, sxz, sxy = stress
    # Relaxed stress
    factor = x**q
    sxx_r = factor * sxx
    syy_r = factor * syy
    szz_r = factor * szz
    syz_r = factor * syz
    sxz_r = factor * sxz
    sxy_r = factor * sxy
    sigma_vm_relaxed = np.sqrt(
        0.5 * ((sxx_r - syy_r) ** 2 + (syy_r - szz_r) ** 2 + (szz_r - sxx_r) ** 2) +
        3 * (syz_r ** 2 + sxz_r ** 2 + sxy_r ** 2)
    )
    if sigma_vm_relaxed == 0:
        return 0.0
    # Partial derivatives w.r.t. each relaxed stress component
    d_vm_dsxx = (2 * sxx_r - syy_r - szz_r) / (2 * sigma_vm_relaxed)
    d_vm_dsyy = (2 * syy_r - sxx_r - szz_r) / (2 * sigma_vm_relaxed)
    d_vm_dszz = (2 * szz_r - sxx_r - syy_r) / (2 * sigma_vm_relaxed)
    d_vm_dsyz = 3 * syz_r / sigma_vm_relaxed
    d_vm_dsxz = 3 * sxz_r / sigma_vm_relaxed
    d_vm_dsxy = 3 * sxy_r / sigma_vm_relaxed
    # Chain rule: d(sigma_vm_relaxed)/dE = sum_i d(sigma_vm_relaxed)/d(sigma_i) * d(sigma_i)/dE
    # For linear elasticity, stress is proportional to E, so d(sigma_i)/dE = stress_i / E
    d_vm_dE = (
        d_vm_dsxx * sxx +
        d_vm_dsyy * syy +
        d_vm_dszz * szz +
        d_vm_dsyz * syz +
        d_vm_dsxz * sxz +
        d_vm_dsxy * sxy
    ) * factor
    return d_vm_dE
# --- Pure Structural Optimization Function ---
def optimizationFunction_structural(
    x, fe_solver, to_params, vae_info, patchwork_colors, num_patches, num_elems, num_design_var, H, Hs, KE, materialEncoder, shared_vars, gamma=100, debug=False, apply_filter_to_materials=True, use_penalization=True
):
    if 'J0' not in shared_vars or shared_vars['J0'] is None:
        shared_vars['J0'] = None
    if use_penalization:
        x = np.asarray(x).flatten()
        x = vae_info.unnormalize_last_n(arr=x, n=2*num_patches)
    else:
        x = np.asarray(x).flatten()
        x = vae_info.map_to_ellipse_torch_patch(x, num_material_vars=2*num_patches)
    xTensor = torch.tensor(x).float()
    xTensor.requires_grad = True
    xDesign = x[0:num_elems]
  
    #fe_solver.mesh.setPseudoDensity(xDesign)
    #fe_solver.plot_pseudo_density()
    zD = xTensor[num_elems:]
    zDesign = zD.view(2, -1).T
    decoded = materialEncoder.vaeNet.decoder(zDesign)
    youngsModulus, _ = materialEncoder.getMaterialProperties(decoded)
    ym = youngsModulus.detach().numpy()
    EDesign = np.zeros_like(patchwork_colors, dtype=float)
    for patch_id in range(num_patches):
        EDesign[patchwork_colors == patch_id] = ym[patch_id]
    fe_solver.mat_prop = [
        mat_lib.create_material_with_defaults(name=f"Material_{i+1}", youngs_modulus=EDesign[i])
        for i in range(EDesign.shape[0])
    ]
    fe_solver.set_structural_material(fe_solver.mat_prop)
    sol = fe_solver.solve(xDesign, MaterialModel.SIMP)
    obj = np.einsum('i, i -> ', fe_solver.total_force, sol)
    if shared_vars['J0'] is None:
        shared_vars['J0'] = obj
        print(f"J0: {obj}")
    J0 = shared_vars['J0']
    print(f"J: {obj}")
    obj_norm = obj / J0
    ce = (np.dot(sol[fe_solver.mesh.edofMat].reshape(num_elems, 24), KE) * sol[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
    penal = 3.0
    dJ_dxDesign = (-penal * xDesign ** (penal - 1)) * EDesign * ce
    dJ_dEDesign = np.asarray((xDesign ** penal) * ce)
    reduced_dJ_dEDesign = np.zeros(num_patches)
    for patch_id in range(num_patches):
        reduced_dJ_dEDesign[patch_id] = np.mean(dJ_dEDesign[patchwork_colors == patch_id])
    dJ_dEDesign_tensor = torch.tensor(reduced_dJ_dEDesign)
    youngsModulus.backward(dJ_dEDesign_tensor)
    dJ_dzDesign = xTensor.grad.detach().numpy()
    grad_obj = np.concatenate((dJ_dxDesign, -dJ_dzDesign[num_elems:].flatten()))
    grad_obj = grad_obj / J0
    vf = np.mean(xDesign)
    xConstraint_tensor = torch.tensor(x).float()
    xConstraint_tensor.requires_grad = True
    pseudoDensity = xConstraint_tensor[0:num_elems]
    zcTensor = xConstraint_tensor[num_elems:]
    zc = zcTensor.view(2, -1).T
    decoded = materialEncoder.vaeNet.decoder(zc)
    _, massDensity = materialEncoder.getMaterialProperties(decoded)
    md = torch.zeros(patchwork_colors.size, dtype=torch.float32)
    for patch_id in range(num_patches):
        md[patchwork_colors == patch_id] = massDensity[patch_id]
    totalMass = torch.einsum('m,m->m', md, pseudoDensity).sum() * fe_solver.mesh.elem_size[0] ** 3
    # Store current mass for summary and history
    shared_vars['current_mass'] = float(totalMass.item())
    if 'history' not in shared_vars:
        shared_vars['history'] = {'compliance': [], 'volfrac': [], 'mass': []}
    shared_vars['history']['mass'].append(float(totalMass.item()))
    massConstraint = ((totalMass / to_params.Constraints[0][2]) - 1.0)
    massConstraint.backward()
    cons = massConstraint.detach().numpy()
    grad_cons = xConstraint_tensor.grad.detach().numpy()
    # Print target and current mass
    target_mass = to_params.Constraints[0][2]
   
    shared_vars['EDesign'] = EDesign.copy()
    shared_vars['zDesign'] = zDesign.clone()
    if 'history' not in shared_vars:
        shared_vars['history'] = {'compliance': [], 'volfrac': []}
    shared_vars['history']['compliance'].append(float(obj))
    shared_vars['history']['volfrac'].append(float(vf))
    # Apply filter to density and (optionally) latent variables
    grad_obj[0:num_elems] = (H * grad_obj[0:num_elems]) / Hs
    grad_cons[0:num_elems] = (H * grad_cons[0:num_elems]) / Hs
    if apply_filter_to_materials:
        grad_obj[num_elems:num_elems + num_patches] = (H * grad_obj[num_elems:num_elems + num_patches]) / Hs
        grad_obj[num_elems + num_patches:num_elems + 2*num_patches] = (H * grad_obj[num_elems + num_patches:num_elems + 2*num_patches]) / Hs  
        grad_cons[num_elems:num_elems + num_patches] = (H * grad_cons[num_elems:num_elems + num_patches]) / Hs
        grad_cons[num_elems + num_patches:num_elems + 2*num_patches] = (H * grad_cons[num_elems + num_patches:num_elems + 2*num_patches]) / Hs  
    elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force,3)
    if (elemsWithForces.size > 0):
        grad_obj[elemsWithForces] = min(grad_obj)


    # print(to_params.ElemsToKeep)

    if (to_params.ElemsToKeep is not None):
        grad_obj[to_params.ElemsToKeep] = min(grad_obj)
    grad_obj= np.array([grad_obj]).reshape((num_design_var, 1))
    cons = np.array([cons]).reshape((1, 1))
    grad_cons = grad_cons.reshape((1, num_design_var))
   # --- Latent space penalization ---
    Z_data = vae_info.training_latents.to(zDesign.device)  # shape (N_train, latentDim)
    p_softmin = -1
    # gamma = 1

    d_ij = torch.cdist(zDesign, Z_data, p=2)  # shape (num_patches, N_train)
    soft_i = torch.sum(d_ij ** p_softmin, dim=1).pow(1.0/p_softmin)
    # print(f"soft_i: {soft_i}")
    # input()
    penalty = gamma * torch.sum(soft_i)/num_patches

    # Add penalty to objective
    obj = obj_norm + penalty.item()

    # Backprop for penalty gradient
    xTensor.grad = None
    penalty.backward(retain_graph=True)
    dpen = xTensor.grad[num_elems:].detach().numpy().reshape(-1, 2)  # shape (num_patches, latentDim)
    # Add penalty gradient to grad_obj (for latent variables only)
    grad_obj[num_elems:,0] += dpen.flatten()      
    

    # Print order of magnitude for scaled compliance and penalty
    # if obj_norm > 0:
    #     print(f"Iteration: Scaled compliance (obj/J0) = {obj_norm:.3e} (10^{int(math.log10(obj_norm))})")
    # else:
    #     print(f"Iteration: Scaled compliance (obj/J0) = {obj_norm:.3e} (zero or negative)")

    # if penalty.item() > 0:
    #     print(f"Iteration: Distance penalty = {penalty.item():.3e} (10^{int(math.log10(penalty.item()))})")
    # else:
    #     print(f"Iteration: Distance penalty = {penalty.item():.3e} (zero or negative)") 


    return obj, grad_obj, cons, grad_cons

# --- Temp-Dependent Optimization Function ---
def optimizationFunction_tempdependent(
    x, fe_solver_structural, fe_solver_thermal, to_params, vae_info, patchwork_colors, num_patches, num_elems, num_design_var, H, Hs, KE, shared_vars, gamma=100, debug=False, apply_filter_to_materials=True, use_penalization=True
):
    if 'J0' not in shared_vars or shared_vars['J0'] is None:
        shared_vars['J0'] = None
    if use_penalization:
        x = np.asarray(x).flatten()
        x = vae_info.unnormalize_last_n(arr=x, n=2*num_patches)
    else:
        x = np.asarray(x).flatten()
        x = vae_info.map_to_ellipse_torch_patch(x, num_material_vars=2*num_patches)
    xTensor = torch.tensor(x).float()
    xTensor.requires_grad = True
    xDesign = x[0:num_elems]
    zD = xTensor[num_elems:]
    zDesign = zD.view(2, -1).T
    decoded = vae_info.vaeNet.decoder(zDesign)
    Ea, Eb, Ec, Ed, _, thermalConductivity = vae_info.getMaterialProperties_tempdependent(decoded)
    # Assign patch properties to elements (preserve grad)
    patchwork_colors_torch = torch.tensor(patchwork_colors, dtype=torch.long, device=Ea.device)
    Ea_elem = Ea[patchwork_colors_torch]
    Eb_elem = Eb[patchwork_colors_torch]
    Ec_elem = Ec[patchwork_colors_torch]
    Ed_elem = Ed[patchwork_colors_torch]
    # --- THERMAL ANALYSIS ---
    thermalConductivity_elem = thermalConductivity[patchwork_colors_torch]
    fe_solver_thermal.mat_prop = [
        mat_lib.create_material_with_defaults(
            name=f"ThermalMaterial_{i+1}",
            thermal_conductivity=thermalConductivity_elem[i].item()
        )
        for i in range(num_elems)
    ]
    fe_solver_thermal.set_thermal_material(fe_solver_thermal.mat_prop)
    T_full = fe_solver_thermal.solve(xDesign)
    edofMat = fe_solver_thermal.mesh.edofMat
    T = np.mean(T_full[edofMat], axis=1)
    # --- STRUCTURAL ANALYSIS WITH TEMPERATURE-DEPENDENT MATERIALS ---
    T_torch = torch.tensor(T, dtype=Ea.dtype, device=Ea.device)
    E0 = 100
    T0 = 500
    EDesign = (
        Ea_elem * T_torch**3 * E0 / T0**3 +
        Eb_elem * T_torch**2 * E0 / T0**2 +
        Ec_elem * T_torch * E0 / T0 +
        Ed_elem * E0
    )
    fe_solver_structural.mat_prop = [
        mat_lib.create_material_with_defaults(
            name=f"Material_{i+1}", 
            youngs_modulus=EDesign[i].item(), 
            poissons_ratio=0.3
        )
        for i in range(num_elems)
    ]
    fe_solver_structural.set_structural_material(fe_solver_structural.mat_prop)
    sol = fe_solver_structural.solve(xDesign, MaterialModel.SIMP)
    obj = np.einsum('i, i -> ', fe_solver_structural.total_force, sol)
    if shared_vars['J0'] is None:
        shared_vars['J0'] = obj
    J0 = shared_vars['J0']
    obj_norm = obj / J0
    ce = (np.dot(sol[fe_solver_structural.mesh.edofMat].reshape(num_elems, 24), KE) * sol[fe_solver_structural.mesh.edofMat].reshape(num_elems, 24)).sum(1)
    penal = 3.0
    dJ_dxDesign = (-penal * xDesign ** (penal - 1)) * EDesign.detach().cpu().numpy() * ce
    dJ_dE = torch.tensor((xDesign ** penal) * ce, dtype=Ea.dtype, device=Ea.device)
    E = EDesign
    xTensor.grad = None
    E.backward(dJ_dE, retain_graph=True)
    dJ_dzDesign = xTensor.grad[num_elems:].detach().cpu().numpy()
    grad_obj = np.concatenate((dJ_dxDesign, -dJ_dzDesign.flatten()))
    grad_obj = grad_obj / J0
    vf = np.mean(xDesign)
    xConstraint_tensor = torch.tensor(x).float()
    xConstraint_tensor.requires_grad = True
    pseudoDensity = xConstraint_tensor[0:num_elems]
    zcTensor = xConstraint_tensor[num_elems:]
    zc = zcTensor.view(2, -1).T
    decoded = vae_info.vaeNet.decoder(zc)
    Ea_c, Eb_c, Ec_c, Ed_c, massDensity_c, _ = vae_info.getMaterialProperties_tempdependent(decoded)
    md_elem = massDensity_c[patchwork_colors_torch].detach().cpu().numpy()
    md = torch.tensor(md_elem, dtype=pseudoDensity.dtype, device=pseudoDensity.device)
    elem_volume = fe_solver_structural.mesh.elem_size[0] * fe_solver_structural.mesh.elem_size[1] * fe_solver_structural.mesh.elem_size[2]
    totalMass = torch.einsum('m,m->m', md, pseudoDensity).sum() * elem_volume
    # Store current mass for summary and history
    shared_vars['current_mass'] = float(totalMass.item())
    if 'history' not in shared_vars:
        shared_vars['history'] = {'compliance': [], 'volfrac': [], 'mass': []}
    shared_vars['history']['mass'].append(float(totalMass.item()))
    massConstraint = ((totalMass / to_params.Constraints[0][2]) - 1.0)
    massConstraint.backward()
    cons = massConstraint.detach().cpu().numpy()
    grad_cons = xConstraint_tensor.grad.detach().cpu().numpy()
    # Print target and current mass
    target_mass = to_params.Constraints[0][2]
    shared_vars['EDesign'] = EDesign.detach().cpu().numpy().copy()
    shared_vars['zDesign'] = zDesign.clone()
    shared_vars['thermalConductivity'] = thermalConductivity_elem.detach().cpu().numpy().copy()
    shared_vars['massDensity'] = massDensity_c.detach().cpu().numpy().copy()
    if 'history' not in shared_vars:
        shared_vars['history'] = {'compliance': [], 'volfrac': []}
    shared_vars['history']['compliance'].append(float(obj))
    shared_vars['history']['volfrac'].append(float(vf))
    # Apply filter to density and (optionally) latent variables
    grad_obj[0:num_elems] = (H * grad_obj[0:num_elems]) / Hs
    grad_cons[0:num_elems] = (H * grad_cons[0:num_elems]) / Hs
    if apply_filter_to_materials:
        grad_obj[num_elems:num_elems + num_patches] = (H * grad_obj[num_elems:num_elems + num_patches]) / Hs
        grad_obj[num_elems + num_patches:num_elems + 2*num_patches] = (H * grad_obj[num_elems + num_patches:num_elems + 2*num_patches]) / Hs  
        grad_cons[num_elems:num_elems + num_patches] = (H * grad_cons[num_elems:num_elems + num_patches]) / Hs
        grad_cons[num_elems + num_patches:num_elems + 2*num_patches] = (H * grad_cons[num_elems + num_patches:num_elems + 2*num_patches]) / Hs  
    elemsWithForces = find_elements_with_forces(fe_solver_structural.mesh, fe_solver_structural.bc.force,3)
    if (elemsWithForces.size > 0):
        grad_obj[elemsWithForces] = min(grad_obj)


    # print(to_params.ElemsToKeep)

    if (to_params.ElemsToKeep is not None):
        grad_obj[to_params.ElemsToKeep] = min(grad_obj)
    grad_obj= np.array([grad_obj]).reshape((num_design_var, 1))
    cons = np.array([cons]).reshape((1, 1))
    grad_cons = grad_cons.reshape((1, num_design_var))
   # --- Latent space penalization ---
    Z_data = vae_info.training_latents.to(zDesign.device)  # shape (N_train, latentDim)
    p_softmin = -1
    # gamma = 1

    d_ij = torch.cdist(zDesign, Z_data, p=2)  # shape (num_patches, N_train)
    soft_i = torch.sum(d_ij ** p_softmin, dim=1).pow(1.0/p_softmin)
    # print(f"soft_i: {soft_i}")
    # input()
    penalty = gamma * torch.sum(soft_i)/num_patches

    # Add penalty to objective
    obj = obj_norm + penalty.item()

    # Backprop for penalty gradient
    xTensor.grad = None
    penalty.backward(retain_graph=True)
    dpen = xTensor.grad[num_elems:].detach().numpy().reshape(-1, 2)  # shape (num_patches, latentDim)
    # Add penalty gradient to grad_obj (for latent variables only)
    grad_obj[num_elems:,0] += dpen.flatten()  


    # Print order of magnitude for scaled compliance and penalty
    if obj_norm > 0:
        print(f"Iteration: Scaled compliance (obj/J0) = {obj_norm:.3e} (10^{int(math.log10(obj_norm))})")
    else:
        print(f"Iteration: Scaled compliance (obj/J0) = {obj_norm:.3e} (zero or negative)")

    if penalty.item() > 0:
        print(f"Iteration: Distance penalty = {penalty.item():.3e} (10^{int(math.log10(penalty.item()))})")
    else:
        print(f"Iteration: Distance penalty = {penalty.item():.3e} (zero or negative)") 
    return obj, grad_obj, cons, grad_cons

# --- Structural Cost Optimization Function ---
def optimizationFunction_structuralcost(
    x, fe_solver, to_params, vae_info, num_elems, num_design_var, H, Hs, KE, materialEncoder, shared_vars, gamma=100, debug=False, apply_filter_to_materials=True, use_penalization=True
):
    if 'J0' not in shared_vars or shared_vars['J0'] is None:
        shared_vars['J0'] = None
    if use_penalization:
        x = np.asarray(x).flatten()
        x = vae_info.unnormalize_last_n(arr=x, n=2*num_elems)
    else:
        x = np.asarray(x).flatten()
        x = vae_info.map_to_ellipse_torch(x, num_material_vars=2*num_elems)
    xTensor = torch.tensor(x).float()
    xTensor.requires_grad = True
    xDesign = x[0:num_elems]
    zD = xTensor[num_elems:]
    zDesign = zD.view(2, -1).T
    decoded = materialEncoder.vaeNet.decoder(zDesign)
    youngsModulus, massDensity, cost = materialEncoder.getMaterialProperties_structuralcost(decoded)
    EDesign = youngsModulus.detach().numpy()

    fe_solver.mat_prop = [
        mat_lib.create_material_with_defaults(name=f"Material_{i+1}", youngs_modulus=EDesign[i])
        for i in range(EDesign.shape[0])
    ]
    fe_solver.set_structural_material(fe_solver.mat_prop)
    sol = fe_solver.solve(xDesign, MaterialModel.SIMP)
    obj = np.einsum('i, i -> ', fe_solver.total_force, sol)
    if shared_vars['J0'] is None:
        shared_vars['J0'] = obj
        print(f"J0: {obj}")
    J0 = shared_vars['J0']
    print(f"J: {obj}")
    obj_norm = obj / J0
    ce = (np.dot(sol[fe_solver.mesh.edofMat].reshape(num_elems, 24), KE) * sol[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
    penal = 3.0
    dJ_dxDesign = (-penal * xDesign ** (penal - 1)) * EDesign * ce
    dJ_dEDesign = np.asarray((xDesign ** penal) * ce)
    dJ_dEDesign_tensor = torch.tensor(dJ_dEDesign)
    youngsModulus.backward(dJ_dEDesign_tensor)
    dJ_dzDesign = xTensor.grad.detach().numpy()
    grad_obj = np.concatenate((dJ_dxDesign, -dJ_dzDesign[num_elems:].flatten()))
    grad_obj = grad_obj / J0
    vf = np.mean(xDesign)
    # --- Mass and Cost Constraints ---
    xConstraint_tensor = torch.tensor(x).float()
    xConstraint_tensor.requires_grad = True
    pseudoDensity = xConstraint_tensor[0:num_elems]
    zcTensor = xConstraint_tensor[num_elems:]
    zc = zcTensor.view(2, -1).T
    decoded = materialEncoder.vaeNet.decoder(zc)
    _, massDensity, cost = materialEncoder.getMaterialProperties_structuralcost(decoded)
    totalMass = torch.einsum('m,m->m', massDensity, pseudoDensity).sum() * fe_solver.mesh.elem_size[0] ** 3
    shared_vars['current_mass'] = float(totalMass.item())
    if 'history' not in shared_vars:
        shared_vars['history'] = {'compliance': [], 'volfrac': [], 'mass': [], 'cost': []}
    shared_vars['history']['mass'].append(float(totalMass.item()))
    massConstraint = ((totalMass / to_params.Constraints[0][2]) - 1.0)
    massConstraint.backward(retain_graph=True)
    cons_mass = massConstraint.detach().numpy()
    grad_cons_mass = xConstraint_tensor.grad.detach().numpy()
    xConstraint_tensor.grad = None
    totalCost = torch.einsum('m,m,m->m', cost, massDensity, pseudoDensity).sum() * fe_solver.mesh.elem_size[0] ** 3
    shared_vars['current_cost'] = float(totalCost.item())
    shared_vars['history']['cost'].append(float(totalCost.item()))
    costConstraint = ((totalCost / to_params.Constraints[1][2]) - 1.0)
    costConstraint.backward()
    cons_cost = costConstraint.detach().numpy()
    grad_cons_cost = xConstraint_tensor.grad.detach().numpy()
    # Print target and current mass/cost
    target_mass = to_params.Constraints[0][2]
    target_cost = to_params.Constraints[1][2]
    shared_vars['EDesign'] = EDesign.copy()
    shared_vars['zDesign'] = zDesign.clone()
    if 'history' not in shared_vars:
        shared_vars['history'] = {'compliance': [], 'volfrac': []}
    shared_vars['history']['compliance'].append(float(obj))
    shared_vars['history']['volfrac'].append(float(vf))
    # Apply filter to density and (optionally) latent variables
    grad_obj[0:num_elems] = (H * grad_obj[0:num_elems]) / Hs
    grad_cons_mass[0:num_elems] = (H * grad_cons_mass[0:num_elems]) / Hs
    grad_cons_cost[0:num_elems] = (H * grad_cons_cost[0:num_elems]) / Hs
    if apply_filter_to_materials:
        grad_obj[num_elems:2*num_elems] = (H * grad_obj[num_elems:2*num_elems]) / Hs
        grad_obj[2*num_elems:3*num_elems] = (H * grad_obj[2*num_elems:3*num_elems]) / Hs  
        grad_cons_mass[num_elems:2*num_elems] = (H * grad_cons_mass[num_elems:2*num_elems]) / Hs
        grad_cons_mass[2*num_elems:3*num_elems] = (H * grad_cons_mass[2*num_elems:3*num_elems]) / Hs  
        grad_cons_cost[num_elems:2*num_elems] = (H * grad_cons_cost[num_elems:2*num_elems]) / Hs
        grad_cons_cost[2*num_elems:3*num_elems] = (H * grad_cons_cost[2*num_elems:3*num_elems]) / Hs  
    elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force,3)
    if (elemsWithForces.size > 0):
        grad_obj[elemsWithForces] = min(grad_obj)
    if (to_params.ElemsToKeep is not None):
        grad_obj[to_params.ElemsToKeep] = min(grad_obj)
    grad_obj= np.array([grad_obj]).reshape((num_design_var, 1))
    cons = np.array([cons_mass, cons_cost]).reshape((2, 1))
    grad_cons = np.vstack([grad_cons_mass.reshape((1, num_design_var)), grad_cons_cost.reshape((1, num_design_var))])
    # --- Latent space penalization ---
    Z_data = vae_info.training_latents.to(zDesign.device)  # shape (N_train, latentDim)
    p_softmin = -6
    d_ij = torch.cdist(zDesign, Z_data, p=2) + 1e-6  # shape (num_patches, N_train)
    soft_i = torch.sum(d_ij ** p_softmin, dim=1).pow(1.0/p_softmin)
    penalty = gamma * torch.sum(soft_i)/num_elems
    # Add penalty to objective
    obj = obj_norm + penalty.item()
    # Backprop for penalty gradient
    xTensor.grad = None
    penalty.backward(retain_graph=True)
    dpen = xTensor.grad[num_elems:].detach().numpy().reshape(-1, 2)  # shape (num_patches, latentDim)
    grad_obj[num_elems:,0] += dpen.flatten()      
    return obj, grad_obj, cons, grad_cons

# --- Structural Yield Optimization Function ---
def optimizationFunction_structuralyield(
    x, fe_solver, to_params, vae_info, num_elems, num_design_var, H, Hs, KE, 
    materialEncoder, shared_vars, gamma=100, debug=False, 
    apply_filter_to_materials=True, use_penalization=True):

   
    # --- Mass Objective ---
    if 'M0' not in shared_vars or shared_vars['M0'] is None:
        shared_vars['M0'] = None
    if use_penalization:
        x = np.asarray(x).flatten()
        x = vae_info.unnormalize_last_n(arr=x, n=2*num_elems)
    else:
        x = np.asarray(x).flatten()
        x = vae_info.map_to_ellipse_torch(x, num_material_vars=2*num_elems)
    xTensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    xDesign = xTensor[0:num_elems]
    zD = xTensor[num_elems:]
    zDesign = zD.view(2, -1).T
    
    decoded = materialEncoder.vaeNet.decoder(zDesign)
    youngsModulus, massDensity, yieldStrength = materialEncoder.getMaterialProperties_structuralyield(decoded)
  
    # print(f"zDesign min/max: {zDesign.min().item():.2f} / {zDesign.max().item():.2f}")
    # print(f"Young's modulus range: {youngsModulus.min().item():.2e} to {youngsModulus.max().item():.2e}")
    # print(f"Mass density range: {massDensity.min().item():.2e} to {massDensity.max().item():.2e}")
    # print(f"Yield strength range: {yieldStrength.min().item():.2e} to {yieldStrength.max().item():.2e}")
    
    poissons_ratio = 0.3
    fe_solver.mat_prop = [
        mat_lib.create_material_with_defaults(
            name=f"Material_{i+1}",
            youngs_modulus=youngsModulus[i].item(),
            yield_strength=yieldStrength[i].item(),
            mass_density=massDensity[i].item(),
            poissons_ratio=poissons_ratio
        )
        for i in range(num_elems)
    ]
    fe_solver.set_structural_material(fe_solver.mat_prop)
    sol = fe_solver.solve(xDesign.detach().numpy(), MaterialModel.SIMP)
   
    pseudoDensity = xDesign
    totalMass = torch.einsum('m,m->m', massDensity, pseudoDensity).sum() * fe_solver.mesh.elem_size[0] ** 3

    # Set and use M0 for normalization
    if shared_vars['M0'] is None:
        shared_vars['M0'] = float(totalMass.item())
        print(f"M0: {shared_vars['M0']}")
    M0 = shared_vars['M0']
    obj_norm = totalMass / M0

    shared_vars['current_mass'] = float(totalMass.item())
    if 'history' not in shared_vars:
        shared_vars['history'] = {'compliance': [], 'volfrac': [], 'mass': [], 'max safety factor': []}
    shared_vars['history']['mass'].append(float(totalMass.item()))
    totalMass.backward(retain_graph=True)
    grad_obj = xTensor.grad.detach().numpy()
    vf = torch.mean(xDesign).item()

    # --- Compliance Constraint ---
    xConstraint_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    xDesign_c = xConstraint_tensor[0:num_elems]
    zD_c = xConstraint_tensor[num_elems:]
    zDesign_c = zD_c.view(2, -1).T
    decoded_c = materialEncoder.vaeNet.decoder(zDesign_c)
    youngsModulus_c, _, _ = materialEncoder.getMaterialProperties_structuralyield(decoded_c)
    shared_vars['EDesign'] = youngsModulus_c.detach().cpu().numpy().copy()
    shared_vars['zDesign'] = zDesign_c.clone()
   
    
    compliance = np.einsum('i,i->', fe_solver.total_force, sol)
    compliance_constraint = compliance / to_params.Constraints[1][2] - 1.0
    
    # Compute gradient of compliance constraint
    ce = (np.dot(sol[fe_solver.mesh.edofMat].reshape(num_elems, 24), KE) * sol[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
    penal = 3.0
    dC_dxDesign = (-penal * xDesign_c.detach().numpy() ** (penal - 1)) * youngsModulus_c.detach().numpy() * ce
    dC_dEDesign = (xDesign_c.detach().numpy() ** penal) * ce
    dC_dEDesign_tensor = torch.tensor(dC_dEDesign, dtype=youngsModulus_c.dtype)
    xConstraint_tensor.grad = None
    youngsModulus_c.backward(dC_dEDesign_tensor)
    dC_dzDesign = xConstraint_tensor.grad[num_elems:].detach().numpy()
    grad_compliance_cons = np.concatenate((dC_dxDesign, -dC_dzDesign.flatten()))
    grad_compliance_cons = grad_compliance_cons / to_params.Constraints[0][2]

    # --- Safety Factor Constraint (p-norm of relaxed von Mises / yield strength) ---
    # Get p-norm and its gradient wrt only density variables (not latent) using compute_pnorm_safety_factor_and_sensitivity function
    fe_solver.postprocess()  # Ensure stress is computed
    print(f"Mass: {shared_vars['current_mass']:.2f}; J: {compliance:.2f}; Max Stress (Pa): {np.max(fe_solver.stressComponents):.2e}")
 
    inv_sf_pnorm, grad_inv_sf_density = compute_pnorm_safety_factor_and_sensitivity(
        sol, xDesign.detach().numpy(), fe_solver, KE, MaterialModel.SIMP, 
        p=to_params.PNormExponent
    )
   
    safety_factor = to_params.Constraints[0][2]
    safety_constraint = inv_sf_pnorm - (1.0 / safety_factor)
    #print(f"Inverse Safety factor (p-norm): {inv_sf_pnorm:.4f}, Constraint (SF - 1/SF_target): {safety_constraint:.4f}")
   
    # 2. Compute latent variable part of gradient (chain rule)
    p = to_params.PNormExponent
    num_latent = zDesign.numel()
    d_sigma_vm_dE = np.zeros(num_elems)
    for e in range(num_elems):
        # Divide by decoded youngs modulus for that element
        d_sigma_vm_dE[e] = d_relaxed_von_mises_dE(
            fe_solver.stressComponents[e], xDesign[e].item(), q=1) / youngsModulus[e].item()
    # Get per-element von Mises and yield strength
    sigma_vm = np.zeros(num_elems)
    for e in range(num_elems):
        stress = fe_solver.stressComponents[e]
        sxx, syy, szz, syz, sxz, sxy = stress
        sigma_vm[e] = np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) +
            3 * (syz ** 2 + sxz ** 2 + sxy ** 2)
        ) * (xDesign[e].item() ** 1)
    Y = np.array([mat.yield_strength for mat in fe_solver.mat_prop])
    S = sigma_vm
    inv_sf_elem = S / Y
    # Track max safety factor for summary
    if 'history' in shared_vars:
        shared_vars['history'].setdefault('max safety factor', []).append(np.max(inv_sf_elem))
    sum_p = np.sum(inv_sf_elem ** p)
    outer = (sum_p) ** (1.0 / p - 1)
    grad_z = np.zeros(num_latent)
    # Backward for dE/dz and dY/dz
    xTensor.grad = None
    youngsModulus.backward(torch.ones_like(youngsModulus), retain_graph=True)
    dE_dz = xTensor.grad[num_elems:].detach().numpy().reshape(num_elems, -1)
    xTensor.grad = None
    yieldStrength.backward(torch.ones_like(yieldStrength), retain_graph=True)
    dY_dz = xTensor.grad[num_elems:].detach().numpy().reshape(num_elems, -1)
    xTensor.grad = None
    for e in range(num_elems):
        d_sigma_dz = d_sigma_vm_dE[e] * dE_dz[e]
        dYdz = dY_dz[e]
        bracket = (d_sigma_dz * Y[e] - dYdz * S[e]) / (Y[e] ** 2) 
        grad_z[0:num_elems] += p * (inv_sf_elem[e] ** (p - 1)) * bracket[0]
        grad_z[num_elems:] += p * (inv_sf_elem[e] ** (p - 1)) * bracket[1]
    grad_z = (1.0 / p) * outer * grad_z

    # 3. Assemble full gradient for constraint
    grad_safety = np.zeros_like(x)
    grad_safety[:num_elems] = grad_inv_sf_density
    grad_safety[num_elems:] = grad_z

    # --- Filtering ---
    grad_obj[0:num_elems] = (H * grad_obj[0:num_elems]) / Hs
    grad_compliance_cons[0:num_elems] = (H * grad_compliance_cons[0:num_elems]) / Hs
    grad_safety[0:num_elems] = (H * grad_safety[0:num_elems]) / Hs
    if apply_filter_to_materials:
        grad_obj[num_elems:2*num_elems] = (H * grad_obj[num_elems:2*num_elems]) / Hs
        grad_obj[2*num_elems:3*num_elems] = (H * grad_obj[2*num_elems:3*num_elems]) / Hs  
        grad_compliance_cons[num_elems:2*num_elems] = (H * grad_compliance_cons[num_elems:2*num_elems]) / Hs
        grad_compliance_cons[2*num_elems:3*num_elems] = (H * grad_compliance_cons[2*num_elems:3*num_elems]) / Hs  
        grad_safety[num_elems:2*num_elems] = (H * grad_safety[num_elems:2*num_elems]) / Hs
        grad_safety[2*num_elems:3*num_elems] = (H * grad_safety[2*num_elems:3*num_elems]) / Hs  
    elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force, 3)
    if (elemsWithForces.size > 0):
        grad_obj[elemsWithForces] = min(grad_obj)
    if (to_params.ElemsToKeep is not None):
        grad_obj[to_params.ElemsToKeep] = min(grad_obj)
    grad_obj = np.array([grad_obj]).reshape((num_design_var, 1))
    cons = np.array([compliance_constraint, safety_constraint]).reshape((2, 1))
    grad_cons = np.vstack([grad_compliance_cons.reshape((1, num_design_var)), grad_safety.reshape((1, num_design_var))])
    # --- Latent space penalization ---
    Z_data = vae_info.training_latents.to(zDesign.device)  # shape (N_train, latentDim)
    p_softmin = -6
    d_ij = torch.cdist(zDesign, Z_data, p=2) + 1e-6  # shape (num_patches, N_train)
    soft_i = torch.sum(d_ij ** p_softmin, dim=1).pow(1.0 / p_softmin)
    penalty = gamma * torch.sum(soft_i) / num_elems
    # Add penalty to objective
    obj = obj_norm + penalty.item()
    # Backprop for penalty gradient
    xTensor.grad = None
    penalty.backward(retain_graph=True)
    dpen = xTensor.grad[num_elems:].detach().numpy().reshape(-1, 2)  # shape (num_patches, latentDim)
    grad_obj[num_elems:, 0] += dpen.flatten()

    obj_norm = obj_norm.detach().numpy() 
    print("Constraints: ",cons.flatten())
    #input("Press Enter to continue...")
    return obj_norm, grad_obj, cons, grad_cons

def plot_loading_and_bc(mesh, bc, title="Loading and Boundary Conditions"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # Plot all nodes
    ax.scatter(mesh.node_xyz[:, 0], mesh.node_xyz[:, 1], mesh.node_xyz[:, 2], c='lightgray', s=10, label='Nodes')
    # Plot fixed nodes
    fixed_nodes = np.unique(bc.fixed_dofs // 3)
    ax.scatter(mesh.node_xyz[fixed_nodes, 0], mesh.node_xyz[fixed_nodes, 1], mesh.node_xyz[fixed_nodes, 2],
               c='blue', s=40, label='Fixed Nodes')
    # Plot loaded nodes (where force != 0)
    loaded_nodes = np.where(np.linalg.norm(bc.force.reshape(-1, 3), axis=1) > 0)[0]
    ax.scatter(mesh.node_xyz[loaded_nodes, 0], mesh.node_xyz[loaded_nodes, 1], mesh.node_xyz[loaded_nodes, 2],
               c='red', s=40, label='Loaded Nodes')
    # Optionally, plot force vectors
    for node in loaded_nodes:
        force_vec = bc.force[3*node:3*node+3]
        start = mesh.node_xyz[node]
        ax.quiver(start[0], start[1], start[2],
                  force_vec[0], force_vec[1], force_vec[2],
                  color='green', length=0.1, normalize=True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
