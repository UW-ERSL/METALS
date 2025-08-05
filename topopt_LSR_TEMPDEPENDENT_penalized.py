

from LSRImports import *
from METALS_thermal_examples import plot_thermal_bc_voxel

_LARGE_NUMBER = 1.e9


def patchwork(mesh, nPatchesDesired=8):
    """Create patches using geometric partitioning similar to deflation groups.
    
    Args:
        mesh: Mesh object containing element centers
        nPatchesDesired (int): Target number of patches
        
    Returns:
        np.ndarray: Array of patch IDs for each element
    """
    # Handle edge cases: if nPatchesDesired is >= num_elems, <= 1, or None
    # treat each element as its own patch (no patching)
    if (nPatchesDesired is None or 
        nPatchesDesired <= 1 or 
        nPatchesDesired >= mesh.num_elems):
        print(f"nPatchesDesired ({nPatchesDesired}) is None, <= 1, or >= num_elems ({mesh.num_elems}). Each element will be its own patch (no patching).")
        return np.arange(mesh.num_elems, dtype=np.int32)
    
    xyz = mesh.elem_centers

    xMin = np.min(xyz[:,0])
    yMin = np.min(xyz[:,1])
    zMin = np.min(xyz[:,2])
    xLength = np.max(xyz[:,0]) - xMin
    yLength = np.max(xyz[:,1]) - yMin
    zLength = np.max(xyz[:,2]) - zMin

    # Calculate patch dimensions to achieve desired number of patches
    temp = xLength * yLength * zLength
    alpha = (nPatchesDesired / temp) ** (1.0 / 3)
    
    nX = max(round(alpha*xLength), 1)
    nY = max(round(alpha*yLength), 1)
    nZ = max(round(alpha*zLength), 1)

    # Initialize patch data structures
    nPatchesTentative = nX * nY * nZ
    print("Patch dimensions:", [nX, nY, nZ])
    
    # Calculate patch sizes
    sizeX = xLength / nX
    sizeY = yLength / nY
    sizeZ = zLength / nZ
    
    # Initialize arrays for patch assignments
    elemPatchNumber = np.zeros(mesh.num_elems, dtype=np.int32)
    patchCount = np.zeros(nPatchesTentative, dtype=np.int32)
    patchCenter = np.zeros((nPatchesTentative, 3))
    
    # Assign elements to patches using vectorized operations
    rel_pos = xyz - np.array([xMin, yMin, zMin])
    indices = np.floor(rel_pos / np.array([sizeX, sizeY, sizeZ])).astype(np.int32)
    indices = np.minimum(indices, np.array([nX - 1, nY - 1, nZ - 1]))

    # Compute patch IDs for all elements at once
    elemPatchNumber = (indices[:, 0] + 
                      nX * indices[:, 1] + 
                      nX * nY * indices[:, 2]).astype(np.int32)

    # Count elements per patch using numpy
    patchCount = np.bincount(elemPatchNumber, minlength=nPatchesTentative)

    # Compute patch centers using vectorized operations
    patchCenter = np.zeros((nPatchesTentative, 3))
    for i in range(3):
        np.add.at(patchCenter[:, i], elemPatchNumber, xyz[:, i])

    for patch in range(nPatchesTentative):
        if (patchCount[patch] > 0):
            for i in range(3):
                patchCenter[patch,i] /= patchCount[patch]
    
    print(f"Created {nPatchesTentative} patches from {mesh.num_elems} elements")
    
    return elemPatchNumber

def topopt_mma_lsr_combined(
    fe_solver_structural,
    fe_solver_thermal,
    to_params,
    vae_info: None,
    minMMAIterations: int = 50,
    maxMMAIterations: int = 300, 
    timeLimit: float = 72000,
    penal: float = 3.0,
    move_limit: float = 0.2,
    kkt_tol: float = 1.e-6,
    move_tol: float = 0.025,
    continuationScheme: bool = False,
    rel_conv_tol: float = 1.e-3,
    debug: bool = False,
    random_latent_init: bool = False,
    nPatchesDesired: int = None,
    use_ellipse_LSR: bool = True,
) -> tuple[np.ndarray, dict, bool, np.ndarray]:
    """
    MMA based topology optimization for minimum compliance with temperature-dependent materials.
    Uses element-wise optimization only (no patches). Requires both structural and thermal FE solvers.
    """
    num_elems = fe_solver_structural.mesh.num_elems
    assert fe_solver_structural.mesh.num_elems == fe_solver_thermal.mesh.num_elems, \
        "Structural and thermal solvers must use the same number of elements"

    num_design_var = num_elems + num_elems * 2  # = 3 * num_elems

    material_model = MaterialModel.SIMP

    tStart = time.time()
    history = {'compliance': [], 'volume': [], 'change': []}
    [H, Hs] = createFilters(fe_solver_structural, to_params)

    elemsWithForces = find_elements_with_forces(fe_solver_structural.mesh, fe_solver_structural.bc.force, 3)

    constraintType = to_params.Constraints[0][0]
    if constraintType == TO_QOI.VOLUME_FRACTION:
        volFractionConstraint = to_params.Constraints[0][2]
    else:
        volFractionConstraint = 1

    if random_latent_init:
        latent_init = np.random.uniform(0, 1, size=(2 * num_elems, 1))
    else:
        latent_init = np.zeros((2 * num_elems, 1))
    
    mma_init = np.concatenate(
        (0.5 * np.ones((num_elems, 1)), latent_init), axis=0
    )

    # --- COMPUTE STIFFNESS MATRICES FOR BOTH PHYSICS ---
    if isinstance(fe_solver_structural.mat_prop, list):
        KE_structural_list = [
            hex_element_stiffness.hex8_stiffness_matrix_structural(
                mp.youngs_modulus, mp.poissons_ratio, fe_solver_structural.mesh.elem_size
            )
            for mp in fe_solver_structural.mat_prop
        ]
        KE_structural = KE_structural_list[0]
    else:
        KE_structural = hex_element_stiffness.hex8_stiffness_matrix_structural(
            fe_solver_structural.mat_prop.youngs_modulus,
            fe_solver_structural.mat_prop.poissons_ratio,
            fe_solver_structural.mesh.elem_size,
        )

    if isinstance(fe_solver_thermal.mat_prop, list):
        KE_thermal_list = [
            hex_element_stiffness.hex8_stiffness_matrix_thermal(
                mp.thermal_conductivity, fe_solver_thermal.mesh.elem_size
            )
            for mp in fe_solver_thermal.mat_prop
        ]
        KE_thermal = KE_thermal_list[0]
    else:
        KE_thermal = hex_element_stiffness.hex8_stiffness_matrix_thermal(
            fe_solver_thermal.mat_prop.thermal_conductivity, 
            fe_solver_thermal.mesh.elem_size
        )

    if fe_solver_structural.elem_body_force is not None:
        elem_force = fe_solver_structural.elem_body_force.copy()
        nNodes = fe_solver_structural.mesh.num_nodes
        nodal_body_force = np.zeros((nNodes * 3,))
        nodal_body_force[0::3] = fe_solver_structural.mesh.elem_to_node_field_mapping @ elem_force[0::3]
        nodal_body_force[1::3] = fe_solver_structural.mesh.elem_to_node_field_mapping @ elem_force[1::3]
        nodal_body_force[2::3] = fe_solver_structural.mesh.elem_to_node_field_mapping @ elem_force[2::3]
    else:
        nodal_body_force = None
    if continuationScheme:
        penal = 1.2

    success = True
    shared_vars = {}
    timing = {'FEA': 0.0, 'Thermal': 0.0, 'MMA': 0.0}
    J0_container = {'value': None}   
    def optimizationFunction(x):
        x = np.asarray(x).flatten()
        if use_ellipse_LSR:
            x = vae_info.map_to_ellipse_torch_patch(x, 2 * num_elems)
        else:
            x = vae_info.unnormalize_last_n(arr=x, n=2*num_elems)
        
        xTensor = torch.tensor(x).float()
        xTensor.requires_grad = True
        xDesign = x[0:num_elems]
        zD = xTensor[num_elems:]
        zDesign = zD.view(2, -1).T  # Shape: (num_elems, 2)
        decoded = vae_info.vaeNet.decoder(zDesign)
        Emin, Emid, Emax, Tmin, Tmax, massDensity, thermalConductivity = vae_info.getMaterialProperties_tempdependent(decoded)
        Tmin_np = Tmin.detach().numpy()
        Tmax_np = Tmax.detach().numpy()
        Emin_np = Emin.detach().numpy()
        Emid_np = Emid.detach().numpy()
        Emax_np = Emax.detach().numpy()

        thermalConductivityDesign = thermalConductivity.detach().numpy()

        # --- THERMAL ANALYSIS ---
        fe_solver_thermal.mat_prop = [
            mat_lib.create_material_with_defaults(
                name=f"ThermalMaterial_{i+1}",
                thermal_conductivity=thermalConductivityDesign[i]
            )
            for i in range(num_elems)
        ]
        fe_solver_thermal.set_thermal_material(fe_solver_thermal.mat_prop)

        thermalStart = time.time()
        T_full = fe_solver_thermal.solve(xDesign)  # Nodal temperatures
        timing['Thermal'] += time.time() - thermalStart
        edofMat = fe_solver_thermal.mesh.edofMat
        T = np.mean(T_full[edofMat], axis=1)  # Element-wise temperature
        print(f"Min temperature: {T.min():.4f}, Max temperature: {T.max():.4f}")

        # --- STRUCTURAL ANALYSIS WITH TEMPERATURE-DEPENDENT MATERIALS ---
        # Quadratic Young's modulus relation
        EDesign = (
            (2 * (Emax_np - 2 * Emid_np + Emin_np) / (Tmax_np - Tmin_np) ** 2) * T ** 2 +
            (-(Emax_np * Tmax_np - 4 * Emid_np * Tmax_np + 3 * Emax_np * Tmin_np + 3 * Emin_np * Tmax_np - 4 * Emid_np * Tmin_np + Emin_np * Tmin_np) / (Tmax_np - Tmin_np) ** 2) * T +
            (Emax_np * Tmin_np ** 2 + Emin_np * Tmax_np ** 2 + Emax_np * Tmax_np * Tmin_np - 4 * Emid_np * Tmax_np * Tmin_np + Emin_np * Tmax_np * Tmin_np) / (Tmax_np ** 2 - 2 * Tmax_np * Tmin_np + Tmin_np ** 2)
        )

        fe_solver_structural.mat_prop = [
            mat_lib.create_material_with_defaults(
                name=f"Material_{i+1}", 
                youngs_modulus=EDesign[i], 
                poissons_ratio=0.3
            )
            for i in range(num_elems)
        ]
        fe_solver_structural.set_structural_material(fe_solver_structural.mat_prop)

        print(f"Done with material properties")
        timeFEAStart = time.time()
        sol = fe_solver_structural.solve(xDesign, material_model)
        obj = np.einsum('i, i -> ', fe_solver_structural.total_force, sol)
        if J0_container['value'] is None:
            J0_container['value'] = obj
        J0 = J0_container['value']
        obj_norm = obj / J0
        timing['FEA'] += time.time() - timeFEAStart
        objhis = np.array([obj])
        history['compliance'].append(objhis[0])
        
        ce = (np.dot(sol[fe_solver_structural.mesh.edofMat].reshape(num_elems, 24), KE_structural) * 
              sol[fe_solver_structural.mesh.edofMat].reshape(num_elems, 24)).sum(1)

        penal = 3.0
        dJ_dxDesign = (-penal * xDesign ** (penal - 1)) * EDesign * ce
        dJ_dE = torch.tensor((xDesign ** penal) * ce)
        T_torch = torch.tensor(T, dtype=Emin.dtype, device=Emin.device)
        # E(T) quadratic relation
        E = (
            (2 * (Emax - 2 * Emid + Emin) / (Tmax - Tmin) ** 2) * T_torch ** 2 +
            (-(Emax * Tmax - 4 * Emid * Tmax + 3 * Emax * Tmin + 3 * Emin * Tmax - 4 * Emid * Tmin + Emin * Tmin) / (Tmax - Tmin) ** 2) * T_torch +
            (Emax * Tmin ** 2 + Emin * Tmax ** 2 + Emax * Tmax * Tmin - 4 * Emid * Tmax * Tmin + Emin * Tmax * Tmin) / (Tmax ** 2 - 2 * Tmax * Tmin + Tmin ** 2)
        )
        xTensor.grad = None

        E.backward(dJ_dE, retain_graph=True)
        dJ_dzDesign = xTensor.grad[num_elems:].detach().numpy()
        grad_obj = np.concatenate((dJ_dxDesign, -dJ_dzDesign.flatten()))

        if nodal_body_force is not None:
            ce_body_force = (sol[fe_solver_structural.mesh.edofMat].reshape(num_elems, 24) * 
                           nodal_body_force[fe_solver_structural.mesh.edofMat].reshape(num_elems, 24)).sum(1)
            grad_obj += 2 * ce_body_force

        grad_obj[0:num_elems] = (H * grad_obj[0:num_elems]) / Hs

        if elemsWithForces.size > 0:
            grad_obj[elemsWithForces] = min(grad_obj)

        if to_params.ElemsToKeep is not None:
            grad_obj[to_params.ElemsToKeep] = min(grad_obj)

        vf = np.mean(xDesign)
        history['volume'].append(vf)
        grad_obj=grad_obj/J0
        # Element-wise constraint computation
        xConstraint_tensor = torch.tensor(x).float()
        xConstraint_tensor.requires_grad = True
        pseudoDensity = xConstraint_tensor[0:num_elems]
        zcTensor = xConstraint_tensor[num_elems:]
        zc = zcTensor.view(2, -1).T
        decoded = vae_info.vaeNet.decoder(zc)
        Emin_c, Emid_c, Emax_c, Tmin_c, Tmax_c, massDensity_c, _ = vae_info.getMaterialProperties_tempdependent(decoded)

        md = massDensity_c
        elem_volume = fe_solver_structural.mesh.elem_size[0] * fe_solver_structural.mesh.elem_size[1] * fe_solver_structural.mesh.elem_size[2]
        totalMass = torch.einsum('m,m->m', md, pseudoDensity).sum() * elem_volume
        print(f"Element volume: {elem_volume:.6f} m^3")
        print(f"Current mass: {totalMass.item():.4f}, Target mass: {to_params.Constraints[0][2]:.4f}")
        massConstraint = ((totalMass / to_params.Constraints[0][2]) - 1.0)
        massConstraint.backward()
        cons = massConstraint.detach().numpy()
        grad_cons = xConstraint_tensor.grad.detach().numpy()
        
        shared_vars['EDesign'] = EDesign.copy()
        shared_vars['zDesign'] = zDesign.clone()
        
        grad_obj = np.array([grad_obj]).reshape((num_design_var, 1))
        cons = np.array([cons]).reshape((1, 1))
        grad_cons = grad_cons.reshape((1, num_design_var))
        
        # --- Latent space penalization ---
        # Get training set latent vectors
        Z_data = vae_info.training_latents.to(zDesign.device)  # shape (N_train, latentDim)
        p_softmin = -1
        # gamma = 1

        d_ij = torch.cdist(zDesign, Z_data, p=2)  # shape (num_elems, N_train)
        soft_i = torch.sum(d_ij ** p_softmin, dim=1).pow(1.0/p_softmin)
        penalty = gamma * torch.sum(soft_i)/num_elems

        # Add penalty to objective
        obj = obj_norm + penalty.item()

        # Backprop for penalty gradient
        xTensor.grad = None
        penalty.backward(retain_graph=True)
        dpen = xTensor.grad[num_elems:].detach().numpy().reshape(-1, 2)  # shape (num_elems, latentDim)

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

    x0 = mma_init.reshape(-1, 1)
    lowerBound = np.zeros(num_design_var, dtype=float).reshape(-1, 1)
    upperBound = np.ones(num_design_var, dtype=float).reshape(-1, 1)
    nVariables = num_design_var
    nConstraints = 1
    calls_per_stage = 10000  # Number of objective calls before doubling gamma
    gamma_max = 100
    gamma_init = 100
    gamma_factor = 5.0
    def build_itertrack_obj(func, gamma_init, calls_per_stage, gamma_max,gamma_factor):
        state = {'calls': 0, 'gamma': gamma_init}
        def wrapped(x):
            state['calls'] += 1
            if state['calls'] % calls_per_stage == 0:
                if state['gamma'] < gamma_max:
                    state['gamma'] *= gamma_factor
                else:
                    state['gamma'] = gamma_max
                print(f"Stage {state['calls']//calls_per_stage}, gamma = {state['gamma']}")
            # Inject gamma into global scope for optimizationFunction
            func.__globals__['gamma'] = state['gamma']
            return func(x)
        return wrapped

    itertrack_obj = build_itertrack_obj(
        optimizationFunction,
        gamma_init,
        calls_per_stage,
        gamma_max,
        gamma_factor
    )
    mma_start = time.time()
    [xOptimal, f0val, df0dx, gval, dgdx, nFEAs] = runMMA(
        nVariables, nConstraints, itertrack_obj, x0, lowerBound,
        upperBound, maxIterations=maxMMAIterations, timeLimitSecs=timeLimit, 
        move_limit=move_limit, kktTol=kkt_tol, fTolerance=rel_conv_tol,
        gTolerance=rel_conv_tol, verbose=True
    )   

    # [xOptimal, f0val, df0dx, gval, dgdx, nFEAs] = runMMA(
    #     nVariables, nConstraints, optimizationFunction, x0, lowerBound,
    #     upperBound, maxIterations=maxMMAIterations, timeLimitSecs=timeLimit, 
    #     move_limit=move_limit, kktTol=kkt_tol, fTolerance=rel_conv_tol,
    #     gTolerance=rel_conv_tol, verbose=True
    # )
    timing['MMA'] = time.time() - mma_start

    x = np.asarray(xOptimal).flatten()
    xDesign = x[0:num_elems]
    zDesign = shared_vars['zDesign']
    EDesign = shared_vars['EDesign']
    fe_solver_structural.mesh.setPseudoDensity(x[0:num_elems])
    
    decoded = vae_info.vaeNet.decoder(zDesign)
    Emin, Emid, Emax, Tmin, Tmax, massDensity, thermalConductivity = vae_info.getMaterialProperties_tempdependent(decoded)
    md = massDensity.detach().numpy()
    md[xDesign < 0.001] = 1e-3
    EDesign[xDesign < 0.001] = 1e-3
    
    plt.hist(xDesign, bins=10, alpha=0.7, label='Pseudo-densities')
    plt.legend()
    plt.title('Pseudo-densities Histogram')
    plt.xlabel('Pseudo-density')
    plt.ylabel('Count')
    plt.show()

    plt.hist(np.asarray(EDesign), bins=10, alpha=0.7, label='Young\'s Moduli')
    plt.legend()
    plt.title('Young\'s Moduli Histogram')
    plt.xlabel('Young\'s Modulus (Pa)')
    plt.ylabel('Count')
    plt.show()

    # Plot histogram of thermal conductivity for all elements
    plt.hist(np.asarray(thermalConductivity.detach().cpu().numpy()), bins=10, alpha=0.7, label='Thermal Conductivity')
    plt.legend()
    plt.title('Thermal Conductivity Histogram')
    plt.xlabel('Thermal Conductivity (W/mK)')
    plt.ylabel('Count')
    plt.show()

    # Plot thermal conductivity field on the mesh (voxel-wise)
    fe_solver_structural.plot_elem_field(
        np.asarray(thermalConductivity.detach().cpu().numpy()),
        title='Thermal Conductivity',
        colormap='plasma'
    )   
    history['timeFEA'] = timing.get('FEA', 0.0)
    history['timeThermal'] = timing.get('Thermal', 0.0)
    history['timeMMA'] = timing.get('MMA', 0.0)
    return np.asarray(EDesign), history, success, zDesign.detach().cpu().numpy()

import pickle

def preprocessData():
    import pandas as pd
    df = pd.read_excel('./data/TeledyneDatabase2_Temp.xlsx')

    # MassDensity (6th col, index 5), Emax (13th, 12), Emid (14th, 13), Emin (15th, 14), Tmin (16th, 15), Tmax (17th, 16), Thermal Conductivity (18th, 17)
    rawData = df.iloc[:, [5, 12, 13, 14, 15, 16, 17]].to_numpy()
    feature_names = ['MassDensity', 'Emax', 'Emid', 'Emin', 'Tmin', 'Tmax', 'ThermalConductivity']

    # Log-transform MassDensity, min-max normalize the rest
    mass_density = rawData[:, 0]
    mass_density = np.where(mass_density <= 0, 1e-8, mass_density)
    log_mass_density = np.log10(mass_density)
    md_min, md_max = log_mass_density.min(), log_mass_density.max()
    norm_mass_density = (log_mass_density - md_min) / (md_max - md_min)

    param_cols = rawData[:, 1:6]
    param_min = param_cols.min(axis=0)
    param_max = param_cols.max(axis=0)
    norm_params = np.zeros_like(param_cols)
    for i in range(5):
        if param_max[i] == param_min[i]:
            norm_params[:, i] = param_cols[:, i]
            print(f"{feature_names[i+1]} not normalized (constant value).")
        else:
            norm_params[:, i] = (param_cols[:, i] - param_min[i]) / (param_max[i] - param_min[i])

    # Thermal conductivity normalization
    thermal_cond = rawData[:, 6]
    tc_min, tc_max = thermal_cond.min(), thermal_cond.max()
    if tc_max == tc_min:
        norm_thermal_cond = thermal_cond
        print("Thermal conductivity not normalized (constant value).")
    else:
        norm_thermal_cond = (thermal_cond - tc_min) / (tc_max - tc_min)

    normalizedData = np.column_stack([norm_mass_density, norm_params, norm_thermal_cond])
    trainingData = torch.tensor(normalizedData).float()

    dataInfo = {
        'MassDensity': {'idx': 0, 'scaleMin': md_min, 'scaleMax': md_max, 'is_log': True},
        'Emax': {'idx': 1, 'scaleMin': param_min[0], 'scaleMax': param_max[0], 'is_log': False},
        'Emid': {'idx': 2, 'scaleMin': param_min[1], 'scaleMax': param_max[1], 'is_log': False},
        'Emin': {'idx': 3, 'scaleMin': param_min[2], 'scaleMax': param_max[2], 'is_log': False},
        'Tmin': {'idx': 4, 'scaleMin': param_min[3], 'scaleMax': param_max[3], 'is_log': False},
        'Tmax': {'idx': 5, 'scaleMin': param_min[4], 'scaleMax': param_max[4], 'is_log': False},
        'ThermalConductivity': {'idx': 6, 'scaleMin': tc_min, 'scaleMax': tc_max, 'is_log': False},
    }
    dataIdentifier = {
        'name': df[df.columns[0]],
        'className': df[df.columns[1]],
        'classID': df[df.columns[2]]
    }
    trainInfo = normalizedData
    return trainingData, dataInfo, dataIdentifier, trainInfo

def create_meshgrid(n):
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, y)

    return X, Y

def meshgrid_to_tensor(X, Y):
    # Flatten X and Y and stack them along the last axis to create (n*n, 2) shape
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    points_tensor = torch.stack([torch.tensor(X_flat), torch.tensor(Y_flat)], dim=1)
    return points_tensor.float()

def unlognorm(x, scaleMax, scaleMin):
    return 10.**(x*(scaleMax-scaleMin) + scaleMin)
# Define a function to evaluate at the grid points (example: a simple function)



if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import sys
    import os

    from METALS_TO_examples import METALSTOExamples, getMETALSTOProblem
    from METALS_thermal_examples import METALSThermalExamples, getMETALSThermalProblem

    script_dir = os.path.dirname(__file__)
    rel_path = "./data/vaeNet_ref.nt"
    abs_file_path = os.path.join(script_dir, rel_path)

    # --- Load and/or train material encoder ---
    trainingData, dataInfo, dataIdentifier, trainInfo = preprocessData()
    latentDim, hiddenDim = 2, 250
    numEpochs = 40000
    klFactor = 5e-5
    learningRate = 2e-3
    savedNet = './data/vaeNet_ref.nt'
    vaeSettings = {
        'encoder': {'inputDim': trainingData.shape[1], 'hiddenDim': hiddenDim, 'latentDim': latentDim},
        'decoder': {'latentDim': latentDim, 'hiddenDim': hiddenDim, 'outputDim': trainingData.shape[1]}
    }
    load_pretrained_vae = True# Set to False to train from scratch

    materialEncoder = MaterialEncoder(trainingData, dataInfo, dataIdentifier, vaeSettings)
    with torch.no_grad():
        enc_out = materialEncoder.vaeNet.encoder(trainingData)
        if isinstance(enc_out, tuple):
            z_mu = enc_out[0]
        else:
            z_mu = enc_out
        materialEncoder.training_latents = z_mu.cpu()  # shape (N_train, latentDim)
    print(sys.path)
    if load_pretrained_vae:
        print("Loading pre-trained autoencoder from file...")
        materialEncoder.loadAutoencoderFromFile(abs_file_path)
    else:
        print("Training autoencoder from scratch...")
        materialEncoder.trainAutoencoder(numEpochs, klFactor, savedNet, learningRate)

    # --- Problem setup for EdgeCantilever ---
    to_problem = METALSTOExamples.EdgeCantilever
    nDOFDesired = 5000
    
    mesh_structural, mat_prop_struct, bc_struct, elem_body_force, to_params = getMETALSTOProblem(to_problem,nDOFDesired =nDOFDesired )
    mesh_thermal, mat_prop_thermal, bc_thermal = getMETALSThermalProblem(METALSThermalExamples.EdgeCantilever_TempBC)
    print(f"Structural mesh elements: {mesh_structural.num_elems}")
    print(f"Thermal mesh elements:    {mesh_thermal.num_elems}")
    from METALS_thermal_examples import plot_thermal_mesh_with_bc
    plot_thermal_mesh_with_bc(mesh_thermal, bc_thermal, title="Thermal BCs on Voxel Mesh")

    # --- Create solvers ---
    import hex_structural_fea
    import hex_thermal_fea
    solver = lin_solv.Solvers.PARDISO
    dsolver = deflation.DeflationSolver()
    if solver == lin_solv.Solvers.DPCG:
        nGroups = min(dsolver.maxGroups, max(dsolver.minGroups, round(3 * mesh_structural.num_nodes / dsolver.dofPerGroup)))
        dsolver.create_deflation_groups(mesh_structural, nGroups)
        dsolver.create_deflation_matrix(mesh_structural)
        dsolver.W = dsolver.W[bc_struct.free_dofs, :]

    fe_solver_structural = hex_structural_fea.HexStructuralFEA(
        mesh=mesh_structural,
        mat_prop=mat_prop_struct,
        bc=bc_struct,
        solver=solver,
        dsolver=dsolver,
        rtol=1e-8,
        elem_body_force=elem_body_force
    )
    fe_solver_structural.plot_mesh(title="Structural Mesh with Boundary Conditions", plot_bc=True)
    fe_solver_thermal = hex_thermal_fea.HexThermalFEA(
        mesh=mesh_thermal,
        mat_prop=mat_prop_thermal,
        bc=bc_thermal,
        solver=solver,
        rtol=1e-8
    )

    # --- Optimization parameters ---
    debug = False
    random_latent_init = True
    nPatchesDesired = None  # Element-wise only
    use_ellipse_LSR = False

    # --- Run thermo-structural optimization ---
    startTime = time.time()
    u, history, success, zDesign = topopt_mma_lsr_combined(
        fe_solver_structural=fe_solver_structural,
        fe_solver_thermal=fe_solver_thermal,
        to_params=to_params,
        vae_info=materialEncoder,
        debug=debug,
        random_latent_init=random_latent_init,
        nPatchesDesired=nPatchesDesired,
        use_ellipse_LSR=use_ellipse_LSR
    )
    timeTaken = time.time() - startTime

    # --- Results and plotting ---
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Compliance', color='tab:blue')
    ax1.plot(history['compliance'], color='tab:blue', label='Compliance')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    if 'volume' in history:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Volume Fraction', color='tab:orange')
        ax2.plot(history['volume'], color='tab:orange', linestyle=':', label='Volume Fraction')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)
    else:
        ax1.legend()

    plt.title('MMA: Volume and Compliance vs. Iterations')
    plt.grid(True)
    plt.show(block=True)

    title = f"MMA: nDOF: {3 * fe_solver_structural.mesh.num_nodes}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"
    print(f"Time taken: {timeTaken:.0f} s")

    fe_solver_structural.plot_elem_field(u, title='YoungModulus', colormap='viridis')

    # --- Plot optimized vs real materials in latent space with LSR ellipse ---
    import matplotlib.pyplot as plt


    # Get real material latents
    with torch.no_grad():
        z_real = materialEncoder.vaeNet.encoder(trainingData)
        if isinstance(z_real, tuple):
            z_real = z_real[0]
        z_real_np = z_real.cpu().numpy()

    # Get optimized latent variables
    z_opt = zDesign if isinstance(zDesign, np.ndarray) else zDesign.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot real materials (large, opaque stars)
    ax.scatter(z_real_np[:, 0], z_real_np[:, 1], 
            c='black', marker='*', s=80, label='Real Materials', alpha=1.0)

    # Plot optimized materials (smaller, semi-transparent circles)
    ax.scatter(z_opt[:, 0], z_opt[:, 1], 
            c='red', marker='o', s=40, label='Optimized Materials', alpha=0.5)

    ax.set_xlabel('$z_1$')
    ax.set_ylabel('$z_2$')
    ax.set_title('Optimized Materials vs Real Materials in Latent Space')
    ax.legend()
    ax.set_aspect('equal', 'box')
    plt.grid(True)
    plt.show()