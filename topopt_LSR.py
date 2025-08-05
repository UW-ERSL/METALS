from LSRImports import *

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
    fe_solver,
    to_params,
    vae_info: None,
    minMMAIterations: int = 50,
    maxMMAIterations: int = 100, 
    timeLimit: float = 7200,
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
) -> tuple[np.ndarray, dict]:
    """
    MMA based topology optimization for minimum compliance.
    Always uses patch-based approach where each element can be its own patch if needed.
    
    Args:
        use_ellipse_LSR: If True, uses ellipse constraints and mappings. If False, 
                        uses standard optimization without ellipse-specific logic.
    """
    num_elems = fe_solver.mesh.num_elems

    # Always create patches - if nPatchesDesired is invalid, each element becomes its own patch
    patchwork_colors = patchwork(fe_solver.mesh, nPatchesDesired)
    num_patches = np.unique(patchwork_colors).size
    num_design_var = num_elems + num_patches * 2
    
    print(f"Using patch-based optimization with {num_patches} patches")
    if num_patches == num_elems:
        print(f"Note: Each element is its own patch (equivalent to element-wise)")

    material_model = MaterialModel.SIMP

    tStart = time.time()
    history = {'compliance': [], 'volume': [], 'change': []}
    [H, Hs] = createFilters(fe_solver, to_params)

    elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force, 3)

    # mma_params = mma.MMAParams(
    #     max_iter=maxMMAIterations,
    #     kkt_tol=kkt_tol,
    #     step_tol=move_tol,
    #     move_limit=move_limit,
    #     num_design_var=num_design_var,
    #     num_cons=1,
    #     lower_bound=np.zeros((num_design_var, 1)),
    #     upper_bound=np.ones((num_design_var, 1)),
    # )
    constraintType = to_params.Constraints[0][0]
    if constraintType == TO_QOI.VOLUME_FRACTION:
        volFractionConstraint = to_params.Constraints[0][2]
    else:
        volFractionConstraint = 1
    print(f"volFractionConstraint: {volFractionConstraint:.3f}")

    # Always use patch-based initialization
    if random_latent_init:
        latent_init = np.random.uniform(0, 1, size=(2 * num_patches, 1))
    else:
        latent_init = np.zeros((2 * num_patches, 1))
    
    mma_init = np.concatenate(
        (0.5 * np.ones((num_elems, 1)), latent_init), axis=0
    )
    # mma_state = mma.init_mma(mma_init, mma_params)

    # KE setup (shared)
    if isinstance(fe_solver.mat_prop, list):
        if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
            KE_list = [
                hex_element_stiffness.hex8_stiffness_matrix_structural(
                    mp.youngs_modulus, mp.poissons_ratio, fe_solver.mesh.elem_size
                )
                for mp in fe_solver.mat_prop
            ]
            KE = KE_list[0]
        elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
            KE_list = [
                hex_element_stiffness.hex8_stiffness_matrix_thermal(
                    mp.thermal_conductivity, fe_solver.mesh.elem_size
                )
                for mp in fe_solver.mat_prop
            ]
            KE = KE_list[0]
        print("Assuming all elements have the same material properties")
    else:
        if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
            KE = hex_element_stiffness.hex8_stiffness_matrix_structural(
                fe_solver.mat_prop.youngs_modulus,
                fe_solver.mat_prop.poissons_ratio,
                fe_solver.mesh.elem_size,
            )
        elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
            KE = hex_element_stiffness.hex8_stiffness_matrix_thermal(
                fe_solver.mat_prop.thermal_conductivity, fe_solver.mesh.elem_size
            )

    # x_old = volFractionConstraint * np.ones(num_design_var, dtype=float)
    # timeFEA = 0
    # timeMMA = 0

    if fe_solver.elem_body_force is not None:
        elem_force = fe_solver.elem_body_force.copy()
        nNodes = fe_solver.mesh.num_nodes
        nodal_body_force = np.zeros((nNodes * 3,))
        nodal_body_force[0::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[0::3]
        nodal_body_force[1::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[1::3]
        nodal_body_force[2::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[2::3]
    else:
        nodal_body_force = None
    if continuationScheme:
        penal = 1.2

    success = True
    shared_vars = {}
    timing = {'FEA': 0.0}
    def optimizationFunction(x):
        x = np.asarray(x).flatten()
        if use_ellipse_LSR:
            x = vae_info.map_to_ellipse_torch_patch(x, 2 * num_patches)
        else:
            x = vae_info.unnormalize_last_n(arr=x, n = 2*num_patches)
        xTensor = torch.tensor(x).float()
        xTensor.requires_grad = True
        xDesign = x[0:num_elems]
        zD = xTensor[num_elems:]
        zDesign = zD.view(2, -1).T
        decoded = materialEncoder.vaeNet.decoder(zDesign)
        youngsModulus, _ = materialEncoder.getMaterialProperties(decoded)
        ym = youngsModulus.detach().numpy()
        # Redistribute material properties to elements based on patches
        EDesign = np.zeros_like(patchwork_colors, dtype=float)
        for patch_id in range(num_patches):
            EDesign[patchwork_colors == patch_id] = ym[patch_id]

        fe_solver.mat_prop = [
            mat_lib.create_material_with_defaults(name=f"Material_{i+1}", youngs_modulus=EDesign[i])
            for i in range(EDesign.shape[0])
        ]
        fe_solver.set_structural_material(fe_solver.mat_prop)

        print(f"Done with material properties")
        timeFEAStart = time.time()
        sol = fe_solver.solve(xDesign, material_model)
        obj = np.einsum('i, i -> ', fe_solver.total_force, sol)
        timing['FEA'] += time.time() - timeFEAStart
        objhis = np.array([obj])
        history['compliance'].append(objhis[0])
        ce = (np.dot(sol[fe_solver.mesh.edofMat].reshape(num_elems, 24), KE) * sol[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)

        penal = 3.0  # Use a constant penalty factor
        dJ_dxDesign = (-penal * xDesign ** (penal - 1)) * EDesign * ce
        dJ_dEDesign = np.asarray((xDesign ** penal) * ce)

        # Always use patch-based gradient computation
        reduced_dJ_dEDesign = np.zeros(num_patches)
        for patch_id in range(num_patches):
            reduced_dJ_dEDesign[patch_id] = np.mean(dJ_dEDesign[patchwork_colors == patch_id])
        dJ_dEDesign_tensor = torch.tensor(reduced_dJ_dEDesign)

        youngsModulus.backward(dJ_dEDesign_tensor)
        dJ_dzDesign = xTensor.grad.detach().numpy()
        grad_obj = np.concatenate((dJ_dxDesign, -dJ_dzDesign[num_elems:].flatten()))

        if nodal_body_force is not None:
            ce_body_force = (sol[fe_solver.mesh.edofMat].reshape(num_elems, 24) * nodal_body_force[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
            grad_obj += 2 * ce_body_force

        grad_obj[0:num_elems] = (H * grad_obj[0:num_elems]) / Hs

        if elemsWithForces.size > 0:
            grad_obj[elemsWithForces] = min(grad_obj)

        if to_params.ElemsToKeep is not None:
            grad_obj[to_params.ElemsToKeep] = min(grad_obj)

        vf = np.mean(xDesign)

        # Always use patch-based constraint computation
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
        print(f"Current mass: {totalMass.item():.4f}, Target mass: {to_params.Constraints[0][2]:.4f}")
        massConstraint = ((totalMass / to_params.Constraints[0][2]) - 1.0)
        massConstraint.backward()
        cons = massConstraint.detach().numpy()
        grad_cons = xConstraint_tensor.grad.detach().numpy()
        print(f"Size of grad_cons: {grad_cons.shape}")
        shared_vars['EDesign'] = EDesign.copy()
        shared_vars['zDesign'] = zDesign.clone()
        grad_obj= np.array([grad_obj]).reshape((num_design_var, 1))
        cons = np.array([cons]).reshape((1, 1))
        grad_cons = grad_cons.reshape((1, num_design_var))
        # obj=np.array([obj.item()])
        print(f"Objective shape: {obj.shape}, grad_obj shape: {grad_obj.shape}, c shape: {cons.shape}, dcdx shape: {grad_cons.shape}")
        return obj, grad_obj, cons, grad_cons  #CHECK: Ensure the return is correct for MMA

    x0=mma_init.reshape(-1,1)
    lowerBound = np.zeros(num_design_var, dtype = float).reshape(-1, 1)
    upperBound = np.ones(num_design_var, dtype = float).reshape(-1, 1)
    nVariables = num_design_var
    nConstraints = 1
    print(f"Size of x0: {x0.shape}")
    [xOptimal,f0val, df0dx, gval, dgdx,nFEAs] = runMMA(nVariables,nConstraints,optimizationFunction,x0,lowerBound,
			 upperBound, maxIterations = maxMMAIterations,timeLimitSecs= timeLimit, move_limit = move_limit,kktTol = kkt_tol, fTolerance=rel_conv_tol,gTolerance=rel_conv_tol,verbose = True)

    # Always use patch-based return logic
    x = np.asarray(xOptimal).flatten()
    xDesign= x[0:num_elems]
    zDesign = shared_vars['zDesign']
    EDesign = shared_vars['EDesign']
    fe_solver.mesh.setPseudoDensity(x[0:num_elems])
    # Recompute md using the final zDesign
    decoded = materialEncoder.vaeNet.decoder(zDesign)
    youngModulus, massDensity = materialEncoder.getMaterialProperties(decoded)
    print(f"Final Young's Modulus: {youngModulus.detach().numpy()} GPa")
    print(f"Final mass density: {massDensity.detach().numpy()} kg/m^3")
    md = np.zeros(patchwork_colors.size, dtype=np.float32)
    for patch_id in range(num_patches):
        md[patchwork_colors == patch_id] = massDensity[patch_id].item() if hasattr(massDensity[patch_id], 'item') else massDensity[patch_id]
    md[xDesign < 0.001] = 1e-3
    EDesign[xDesign < 0.001] = 1e-3
    plt.hist(xDesign, bins=10)
    plt.hist(np.asarray(EDesign), bins=10)
    plt.show()
    history['timeFEA'] = timing.get('FEA', 0.0)
    # history['timeMMA'] = timeMMA
    return np.asarray(EDesign), history, success, zDesign.detach().cpu().numpy()

import pickle

def preprocessData(criticality_threshold=None, use_reduced_features=False):
    df = pd.read_excel('./data/TeledyneDatabase.xlsx')
    # Always filter if threshold is provided and column exists
    if criticality_threshold is not None and 'Criticality Index' in df.columns:
        df = df[df['Criticality Index'] < criticality_threshold]
        print(f"Number of materials with Criticality Index < {criticality_threshold}: {len(df)}")
    else:
        print(f"Number of materials: {len(df)}")

    dataIdentifier = {
        'name': df[df.columns[0]],
        'className': df[df.columns[1]],
        'classID': df[df.columns[2]]
    }

    if use_reduced_features:
        # Density is 6th column (index 5), Elastic Modulus is 11th (index 10)
        rawData = df.iloc[:, [5, 10]].to_numpy()
        feature_names = ['MassDensity', 'ElasticModulus']
        YoungsModulus = rawData[:, 1]
    else:
        # Use all columns after the first three
        rawData = df.iloc[:, 3:].to_numpy()
        feature_names = list(df.columns[3:])
        # Young's modulus is at index 10 in the original DataFrame
        YoungsModulus = df.iloc[:, 10].to_numpy()

    EMax = np.max(YoungsModulus)
    print("Max E: ", EMax, " GPa")

    trainInfo = np.log10(rawData)
    dataScaleMax = torch.tensor(np.max(trainInfo, axis=0))
    dataScaleMin = torch.tensor(np.min(trainInfo, axis=0))
    normalizedData = (torch.tensor(trainInfo) - dataScaleMin) / (dataScaleMax - dataScaleMin)
    trainingData = normalizedData.clone().float()

    dataInfo = {}
    for i, name in enumerate(feature_names):
        dataInfo[name] = {'idx': i, 'scaleMin': dataScaleMin[i], 'scaleMax': dataScaleMax[i]}

    return trainingData, dataInfo, dataIdentifier, trainInfo, EMax
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
    from METALS_TO_examples import *
    from LSRImports import *
    import time
    import pandas as pd
    import sys 
    import os
    import warnings
    import matplotlib
    import matplotlib.cm as cm  
    def functional_value(points_tensor):
        decoded = materialEncoder.vaeNet.decoder(points_tensor)
        youngModulus = unlognorm(decoded[:,materialEncoder.dataInfo['ElasticModulus']['idx']], 
                                 materialEncoder.dataInfo['ElasticModulus']['scaleMax'],
                                 materialEncoder.dataInfo['ElasticModulus']['scaleMin'])
        physicalDensity = unlognorm(decoded[:,materialEncoder.dataInfo['MassDensity']['idx']],
                                   materialEncoder.dataInfo['MassDensity']['scaleMax'],
                                   materialEncoder.dataInfo['MassDensity']['scaleMin'])
        return youngModulus.detach().numpy().reshape((100,100)), physicalDensity.detach().numpy().reshape((100,100))

    def plot_filled_contour(X, Y, Z):
        plt.figure(figsize=(6, 6))
        contour = plt.contourf(X, Y, Z, cmap='viridis')  # Use a colormap
        plt.colorbar(label='Young\'s Modulus in GPa')
        plt.title('Filled Contour Plot')
        plt.axis('equal')

    script_dir = os.path.dirname(__file__)
    rel_path = "../data/vaeNet_ref.nt"
    abs_file_path = os.path.join(script_dir, rel_path)

    # thresholds = [2.55,1.5,1.25,1,0.75,0.5] 
    thresholds = [2.55] 
    final_compliances = []
    use_reduced_features = True  # Set to False for full feature set

    for threshold in thresholds:
        print(f"\nProcessing {'reduced features' if use_reduced_features else f'Criticality Index < {threshold}'}")
        trainingData, dataInfo, dataIdentifier, trainInfo, EMax = preprocessData(
            criticality_threshold=threshold, use_reduced_features=use_reduced_features)
        numMaterialsInTrainingData, numFeatures = trainingData.shape

        latentDim, hiddenDim = 2, 250
        numEpochs = 40000
        klFactor = 5e-5
        learningRate = 2e-3
        savedNet = './data/vaeNet_ref.nt'
        vaeSettings = {'encoder':{'inputDim':numFeatures, 'hiddenDim':hiddenDim, 'latentDim':latentDim},
                       'decoder':{'latentDim':latentDim, 'hiddenDim':hiddenDim, 'outputDim':numFeatures}}

        materialEncoder = MaterialEncoder(trainingData, dataInfo, dataIdentifier, vaeSettings)
        convgHistory = materialEncoder.trainAutoencoder(numEpochs, klFactor, savedNet, learningRate)
        materialEncoder.loadAutoencoderFromFile(savedNet)
        predData =  materialEncoder.vaeNet(trainingData)
        zReal = materialEncoder.vaeNet.encoder.z.detach().numpy()          
        # Ellipse constraint setup (only if using ellipse LSR)
        use_ellipse_LSR = True  # Control flag for ellipse-related logic
        if use_ellipse_LSR:
            print("Ellipse constraints ARE being used.")
          
            enclosing_ellipse = welzl(np.array(zReal, dtype=float))
            center,a,b,t = enclosing_ellipse
            constraints = {'distance': {'isOn':False, 'center':center, 'a':a, 'b':b, 'theta':t, 'delta':0.0, 'beta':20}}  # Adjust as needed  
            materialEncoder.constraints = constraints
        else:
            # No ellipse constraints
            print("Ellipse constraints are NOT being used.")
            materialEncoder.constraints = {}

        to_problem = METALSTOExamples.EdgeCantilever
        solver = lin_solv.Solvers.PARDISO
        debug = False 
        # to_params.nDOFDesired = 10000
        # to_params.TargetMass = 1
        mesh, mat_prop, bc, elem_body_force, to_params = getMETALSTOProblem(to_problem)

        elem_body_force = None

        dsolver = deflation.DeflationSolver()
        if (solver == lin_solv.Solvers.DPCG):
            nGroups =  min(dsolver.maxGroups, max(dsolver.minGroups, round(3*mesh.num_nodes/dsolver.dofPerGroup)))
            dsolver.create_deflation_groups(mesh, nGroups)
            dsolver.create_deflation_matrix(mesh)
            dsolver.W = dsolver.W[bc.free_dofs, :]

        fe_solver = hex_structural_fea.HexStructuralFEA(
            mesh=mesh,
            mat_prop=mat_prop,
            bc=bc,
            solver=solver,
            dsolver=dsolver,
            rtol=1e-8,
            elem_body_force=elem_body_force
        )

        # --- Plot colored patches for the original, non-optimized design ---

        nPatchesDesired = 8  # Set the desired number of patches here
        
        # Determine if we should show patches based on nPatchesDesired
        show_patches = (nPatchesDesired is not None and 
                       nPatchesDesired > 1 and 
                       nPatchesDesired < fe_solver.mesh.num_elems)
        
        if show_patches:
            patchwork_colors = patchwork(fe_solver.mesh, nPatchesDesired=nPatchesDesired)
            elem_centers = fe_solver.mesh.elem_centers
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            num_patches = len(np.unique(patchwork_colors))
            
            # Generate more distinct colors for large number of patches
            if num_patches > 50:
                # Use HSV color space for better distinction with many colors
                # Generate evenly spaced hues with varying saturation and value
                np.random.seed(42)  # For reproducible colors
                colors = []
                for i in range(num_patches):
                    hue = (i * 137.508) % 360  # Golden angle spacing for better distribution
                    sat = 0.6 + 0.4 * (i % 3) / 2  # Vary saturation between 0.6-1.0
                    val = 0.7 + 0.3 * ((i // 3) % 3) / 2  # Vary value between 0.7-1.0
                    rgb = matplotlib.colors.hsv_to_rgb([hue/360, sat, val])
                    colors.append(rgb)
                colors = np.array(colors)
                
                # Create custom colormap
                from matplotlib.colors import ListedColormap
                cmap = ListedColormap(colors)
            else:
                # Use standard colormap for smaller number of patches
                cmap = cm.get_cmap('nipy_spectral', num_patches)

            sc = ax.scatter(
                elem_centers[:, 0], elem_centers[:, 1], elem_centers[:, 2],
                c=patchwork_colors, cmap=cmap, s=40
            )
            plt.title(f"Patchwork Coloring of Original Design ({num_patches} patches)", fontsize=18)
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
        else:
            print(f"No patch visualization - using element-wise optimization (nPatchesDesired={nPatchesDesired})")
            
        fe_solver.plot_mesh()
        startTime = time.time()
        u, history, success, zDesign = topopt_mma_lsr_combined(
            fe_solver=fe_solver,
            vae_info=materialEncoder,
            to_params=to_params,
            debug=debug, 
            random_latent_init=True,
            nPatchesDesired=nPatchesDesired,
            use_ellipse_LSR=use_ellipse_LSR
        )
        timeTaken = time.time() - startTime
        # Print FEA and MMA times
        print(f"Time taken for FEA: {history.get('timeFEA', 'N/A'):.2f} seconds")
        # print(f"Time taken for MMA optimization: {history.get('timeMMA', 'N/A'):.2f} seconds")
        print(f"Total time taken: {timeTaken:.2f} seconds")
        final_compliances.append(history['compliance'][-1])
        fe_solver.plot_elem_field(u, title=f"Youngs Modulus for Criticality Index < {threshold} (Time: {timeTaken:.2f} s)")

        n = 100  # Number of points in each direction
        X, Y = create_meshgrid(n)
        points_tensor = meshgrid_to_tensor(X, Y)
        Z0, Z1 = functional_value(points_tensor)
        plot_filled_contour(X, Y, Z0)
        plt.scatter(zDesign[:,0], zDesign[:,1], c='r', s=10, label='Optimized Materials')
        plt.scatter(zReal[:,0], zReal[:,1], c='k', s=10, label='Real Materials')
        plt.xlabel('z0')
        plt.ylabel('z1')
        plt.legend()
        plt.show()

    # Save results as pickle
    with open('compliance_vs_criticality_EdgeCantilever_Temp.pkl', 'wb') as f:
        pickle.dump({'thresholds': thresholds, 'final_compliances': final_compliances}, f)

    print("Saved compliance vs. criticality index threshold to compliance_vs_criticality_EdgeCantilever_Temp.pkl")

    # Plotting compliance vs threshold
    plt.figure()
    plt.plot(thresholds, final_compliances, marker='o')
    plt.xlabel('Criticality Index Threshold')
    plt.ylabel('Final Compliance')
    plt.title('Final Compliance vs. Criticality Index Threshold')
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.show()