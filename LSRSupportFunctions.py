import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from LSRImports import *
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
    temp = xLength * yLength * zLength
    alpha = (nPatchesDesired / temp) ** (1.0 / 3)
    nX = max(round(alpha*xLength), 1)
    nY = max(round(alpha*yLength), 1)
    nZ = max(round(alpha*zLength), 1)
    nPatchesTentative = nX * nY * nZ
    sizeX = xLength / nX
    sizeY = yLength / nY
    sizeZ = zLength / nZ
    rel_pos = xyz - np.array([xMin, yMin, zMin])
    indices = np.floor(rel_pos / np.array([sizeX, sizeY, sizeZ])).astype(np.int32)
    indices = np.minimum(indices, np.array([nX - 1, nY - 1, nZ - 1]))
    elemPatchNumber = (indices[:, 0] + nX * indices[:, 1] + nX * nY * indices[:, 2]).astype(np.int32)
    return elemPatchNumber

# def patchwork(mesh, nPatchesDesired=8):
#     xyz = mesh.elem_centers
#     xMin, yMin, zMin = np.min(xyz, axis=0)
#     xMax, yMax, zMax = np.max(xyz, axis=0)
#     xLength, yLength, zLength = xMax - xMin, yMax - yMin, zMax - zMin

#     # Estimate grid splits based on aspect ratio and desired patch count
#     aspect = np.array([xLength, yLength, zLength])
#     aspect = aspect / np.max(aspect)
#     cube_root = nPatchesDesired ** (1/3)
#     splits = np.round(cube_root * aspect).astype(int)
#     splits[splits < 1] = 1

#     # Efficient local search for best splits
#     best_splits = splits.copy()
#     best_score = float('inf')
#     for dx in range(-2, 3):
#         for dy in range(-2, 3):
#             for dz in range(-2, 3):
#                 test = splits + np.array([dx, dy, dz])
#                 test[test < 1] = 1
#                 npatches = np.prod(test)
#                 # Patch sizes
#                 patch_sizes = np.array([xLength/test[0], yLength/test[1], zLength/test[2]])
#                 aspect_ratio = patch_sizes.max() / patch_sizes.min()
#                 # Score: weighted sum of patch count error and aspect ratio
#                 score = abs(npatches - nPatchesDesired) + 0.1 * (aspect_ratio - 1)
#                 if score < best_score:
#                     best_score = score
#                     best_splits = test.copy()
#     nX, nY, nZ = best_splits

#     sizeX = xLength / nX
#     sizeY = yLength / nY
#     sizeZ = zLength / nZ
#     rel_pos = xyz - np.array([xMin, yMin, zMin])
#     indices = np.floor(rel_pos / np.array([sizeX, sizeY, sizeZ])).astype(np.int32)
#     indices = np.minimum(indices, np.array([nX - 1, nY - 1, nZ - 1]))
#     elemPatchNumber = (indices[:, 0] + nX * indices[:, 1] + nX * nY * indices[:, 2]).astype(np.int32)
#     return elemPatchNumber
# --- Pure Structural Optimization Function ---
def optimizationFunction_structural(
    x, fe_solver, to_params, vae_info, patchwork_colors, num_patches, num_elems, num_design_var, H, Hs, KE, materialEncoder, shared_vars, gamma=100, debug=False
):
    if 'J0' not in shared_vars or shared_vars['J0'] is None:
        shared_vars['J0'] = None
    x = np.asarray(x).flatten()
    x = vae_info.unnormalize_last_n(arr=x, n=2*num_patches)
    xTensor = torch.tensor(x).float()
    xTensor.requires_grad = True
    xDesign = x[0:num_elems]
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
    J0 = shared_vars['J0']
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
    grad_obj[0:num_elems] = (H * grad_obj[0:num_elems]) / Hs
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
    print(f"Iteration: Target mass = {target_mass:.4f}, Current mass = {totalMass.item():.4f}")
    shared_vars['EDesign'] = EDesign.copy()
    shared_vars['zDesign'] = zDesign.clone()
    if 'history' not in shared_vars:
        shared_vars['history'] = {'compliance': [], 'volfrac': []}
    shared_vars['history']['compliance'].append(float(obj))
    shared_vars['history']['volfrac'].append(float(vf))
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
    import math

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

# --- Temp-Dependent Optimization Function ---
def optimizationFunction_tempdependent(
    x, fe_solver_structural, fe_solver_thermal, to_params, vae_info, patchwork_colors, num_patches, num_elems, num_design_var, H, Hs, KE, shared_vars, gamma=100, debug=False
):
    if 'J0' not in shared_vars or shared_vars['J0'] is None:
        shared_vars['J0'] = None
    x = np.asarray(x).flatten()
    x = vae_info.unnormalize_last_n(arr=x, n=2*num_patches)
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
    grad_obj[0:num_elems] = (H * grad_obj[0:num_elems]) / Hs
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
    print(f"Iteration: Target mass = {target_mass:.4f}, Current mass = {totalMass.item():.4f}")
    shared_vars['EDesign'] = EDesign.detach().cpu().numpy().copy()
    shared_vars['zDesign'] = zDesign.clone()
    shared_vars['thermalConductivity'] = thermalConductivity_elem.detach().cpu().numpy().copy()
    shared_vars['massDensity'] = massDensity_c.detach().cpu().numpy().copy()
    if 'history' not in shared_vars:
        shared_vars['history'] = {'compliance': [], 'volfrac': []}
    shared_vars['history']['compliance'].append(float(obj))
    shared_vars['history']['volfrac'].append(float(vf))
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
    import math

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

# --- Preprocess Data Functions ---
def preprocessData_structural():
    import pandas as pd
    df = pd.read_excel('./data/TeledyneDatabase.xlsx')
    rawData = df.iloc[:, [5, 10]].to_numpy()
    feature_names = ['MassDensity', 'ElasticModulus']
    YoungsModulus = rawData[:, 1]
    EMax = np.max(YoungsModulus)
    trainInfo = np.log10(rawData)
    dataScaleMax = torch.tensor(np.max(trainInfo, axis=0))
    dataScaleMin = torch.tensor(np.min(trainInfo, axis=0))
    normalizedData = (torch.tensor(trainInfo) - dataScaleMin) / (dataScaleMax - dataScaleMin)
    trainingData = normalizedData.clone().float()
    dataInfo = {}
    for i, name in enumerate(feature_names):
        dataInfo[name] = {'idx': i, 'scaleMin': dataScaleMin[i], 'scaleMax': dataScaleMax[i]}
    dataIdentifier = {
        'name': df[df.columns[0]],
        'className': df[df.columns[1]],
        'classID': df[df.columns[2]]
    }
    return trainingData, dataInfo, dataIdentifier, trainInfo, EMax

def preprocessData_tempdependent():
    """
    Loads and normalizes material data for temperature-dependent polynomial mode.
    Returns:
        trainingData: torch.Tensor of normalized features [MassDensity, Ea, Eb, Ec, Ed]
        dataInfo: dict with normalization info for each feature
        dataIdentifier: dict with material names and classes
        trainInfo: np.ndarray of normalized data (for reference)
        EMax: None (not used in this mode)
    """
    import pandas as pd
    df = pd.read_excel('./data/TeledyneDatabase2_Temp_scaled.xlsx')

    # MassDensity (6th col, index 5), Ea (13th, 12), Eb (14th, 13), Ec (15th, 14), Ed (16th, 15)
    rawData = df.iloc[:, [5, 12, 13, 14, 15, 16]].to_numpy()
    feature_names = ['MassDensity', 'Ea', 'Eb', 'Ec', 'Ed','ThermalConductivity']

    # Only log-transform MassDensity (col 0), min-max normalize the rest
    mass_density = rawData[:, 0]
    mass_density = np.where(mass_density <= 0, 1e-8, mass_density)
    log_mass_density = np.log10(mass_density)
    md_min, md_max = log_mass_density.min(), log_mass_density.max()
    norm_mass_density = (log_mass_density - md_min) / (md_max - md_min)

    poly_coeffs = rawData[:, 1:5]
    poly_min = poly_coeffs.min(axis=0)
    poly_max = poly_coeffs.max(axis=0)
    norm_poly_coeffs = np.zeros_like(poly_coeffs)
    for i in range(4):
        if poly_max[i] == poly_min[i]:
            norm_poly_coeffs[:, i] = poly_coeffs[:, i]
            print(f"{feature_names[i+1]} not normalized (constant value).")
        else:
            norm_poly_coeffs[:, i] = (poly_coeffs[:, i] - poly_min[i]) / (poly_max[i] - poly_min[i])
    # Thermal conductivity normalization
    thermal_cond = rawData[:, 5]
    tc_min, tc_max = thermal_cond.min(), thermal_cond.max()
    if tc_max == tc_min:
        norm_thermal_cond = thermal_cond
        print("Thermal conductivity not normalized (constant value).")
    else:
        norm_thermal_cond = (thermal_cond - tc_min) / (tc_max - tc_min)

    normalizedData = np.column_stack([norm_mass_density, norm_poly_coeffs, norm_thermal_cond])
    trainingData = torch.tensor(normalizedData).float()

    dataInfo = {
        'MassDensity': {'idx': 0, 'scaleMin': md_min, 'scaleMax': md_max, 'is_log': True},
        'Ea': {'idx': 1, 'scaleMin': poly_min[0], 'scaleMax': poly_max[0], 'is_log': False},
        'Eb': {'idx': 2, 'scaleMin': poly_min[1], 'scaleMax': poly_max[1], 'is_log': False},
        'Ec': {'idx': 3, 'scaleMin': poly_min[2], 'scaleMax': poly_max[2], 'is_log': False},
        'Ed': {'idx': 4, 'scaleMin': poly_min[3], 'scaleMax': poly_max[3], 'is_log': False},
        'ThermalConductivity': {'idx': 5, 'scaleMin': tc_min, 'scaleMax': tc_max, 'is_log': False}
    }
    dataIdentifier = {
        'name': df[df.columns[0]],
        'className': df[df.columns[1]],
        'classID': df[df.columns[2]]
    }
    trainInfo = normalizedData
    Emax=None
    return trainingData, dataInfo, dataIdentifier, trainInfo, Emax

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
