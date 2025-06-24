
from LSRImports import *

_LARGE_NUMBER = 1.e9

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


def patchwork(mesh):
    #KS: Generalize to input of nPatchesDesired
    num_patch_x = 2
    num_patch_y = 2
    num_patch_z = 2

    x = mesh.elem_centers[:, 0]
    y = mesh.elem_centers[:, 1]
    z = mesh.elem_centers[:, 2]
    patch_size_x = (x.max() - x.min()) / num_patch_x
    patch_size_y = (y.max() - y.min()) / num_patch_y
    patch_size_z = (z.max() - z.min()) / num_patch_z

    colors = np.zeros_like(x, dtype=int)

    for i in range(num_patch_x):
        for j in range(num_patch_y):
            for k in range(num_patch_z):
                x_min = x.min() + i * patch_size_x
                x_max = x_min + patch_size_x
                y_min = y.min() + j * patch_size_y
                y_max = y_min + patch_size_y
                z_min = z.min() + k * patch_size_z
                z_max = z_min + patch_size_z

                if i == num_patch_x - 1:
                    x_mask = (x >= x_min) & ((x <= x_max) | np.isclose(x, x_max))
                else:
                    x_mask = (x >= x_min) & (x < x_max)
                if j == num_patch_y - 1:
                    y_mask = (y >= y_min) & ((y <= y_max) | np.isclose(y, y_max))
                else:
                    y_mask = (y >= y_min) & (y < y_max)
                if k == num_patch_z - 1:
                    z_mask = (z >= z_min) & ((z <= z_max) | np.isclose(z, z_max))
                else:
                    z_mask = (z >= z_min) & (z < z_max)

                patch_indices = np.where(x_mask & y_mask & z_mask)[0]
                patch_color = i * num_patch_y * num_patch_z + j * num_patch_z + k
                colors[patch_indices] = patch_color #KS: Change to patch_id if needed

    if (False): # for plotting/debug
        elem_centers = fe_solver.mesh.elem_centers
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        num_patches = len(np.unique(colors))
        cmap = cm.get_cmap('nipy_spectral', num_patches)  # or 'gist_rainbow', 'hsv'

        sc = ax.scatter(
            elem_centers[:, 0], elem_centers[:, 1], elem_centers[:, 2],
            c=colors, cmap=cmap, s=40
        )
        plt.title("Patchwork Coloring of Original Design", fontsize=18)
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
    return colors
 
def topopt_mma_lsr_combined(
    fe_solver,
    to_params,
    vae_info: None,
    minMMAIterations: int = 50,
    maxMMAIterations: int = 200, 
    timeLimit: float = 3600,
    penal: float = 3.0,
    move_limit: float = 0.2,
    kkt_tol: float = 1.e-6,
    move_tol: float = 0.025,
    rel_conv_tol: float = 1.e-4,
    debug: bool = False,
    use_patches: bool = False, #KS Provide number of patches desired or None
    random_latent_init:bool = False,
) -> tuple[np.ndarray, dict]:
    """
    MMA based topology optimization for minimum compliance.
    If use_patches is True, uses patch-based latent variables.
    """
    num_elems = fe_solver.mesh.num_elems

    if use_patches:
        patchwork_colors = patchwork(fe_solver.mesh)
        num_patches = np.unique(patchwork_colors).size
        nLatentVariables = 2* num_patches
    else:
        nLatentVariables = num_elems

    num_design_var = num_elems + nLatentVariables

    material_model = MaterialModel.SIMP

    tStart = time.time()
    history = {'compliance': [], 'volume': [], 'change': []}
    [H, Hs] = createFilters(fe_solver, to_params)

    elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force, 3)

    mma_params = mma.MMAParams(
        max_iter=maxMMAIterations,
        kkt_tol=kkt_tol,
        step_tol=move_tol,
        move_limit=move_limit,
        num_design_var=num_design_var,
        num_cons=1,
        lower_bound=np.zeros((num_design_var, 1)),
        upper_bound=np.ones((num_design_var, 1)),
    )

    
    if random_latent_init:
        latent_init = np.random.uniform(0, 1, size=(nLatentVariables, 1))
    else:
        latent_init = np.zeros((nLatentVariables, 1))

    mma_init = np.concatenate((0.5 * np.ones((num_elems, 1)), latent_init), axis=0)

        
    mma_state = mma.init_mma(mma_init, mma_params)

    # KS: Simplified code is sufficient. Explicit set E = 1 and rho = 1 for clarity. Call it KETemplate
    # Get a Ke template that we will later modify depending on the material properties
    
    KE = hex_element_stiffness.hex8_stiffness_matrix_structural(E=1, nu=0.3, elem_size=fe_solver.mesh.elem_size)
  
    x_old = 0.5 * np.ones(num_design_var, dtype=float)
    timeFEA = 0
    timeMMA = 0

    if fe_solver.elem_body_force is not None:
        elem_force = fe_solver.elem_body_force.copy()
        nNodes = fe_solver.mesh.num_nodes
        nodal_body_force = np.zeros((nNodes * 3,))
        nodal_body_force[0::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[0::3]
        nodal_body_force[1::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[1::3]
        nodal_body_force[2::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[2::3]
    else:
        nodal_body_force = None

    success = True

    while not mma_state.is_converged:
        x = mma_state.x.reshape(-1)
        if use_patches:
            #KS Add comments for critical part of code
            x = vae_info.map_to_ellipse_torch_patch(x, nLatentVariables)# KS: What does this do?
            xTensor = torch.tensor(x).float()
            xTensor.requires_grad = True
            xDesign = x[0:num_elems]
            zD = xTensor[num_elems:]
            zDesign = zD.view(2, -1).T
            decoded = materialEncoder.vaeNet.decoder(zDesign)
            youngsModulus, _ = materialEncoder.getMaterialProperties(decoded)
            ym = youngsModulus.detach().numpy()
            #KS: Explain
            patchwork_colors = patchwork(fe_solver.mesh)
            num_patches = np.unique(patchwork_colors).size
            EDesign = np.zeros_like(patchwork_colors, dtype=float)
            for patch_id in range(num_patches):
                EDesign[patchwork_colors == patch_id] = ym[patch_id]
      
        else: # no need for a separate code?
            xTensor = torch.tensor(x).float()
            xTensor.requires_grad = True
            x = vae_info.map_to_ellipse_torch(xTensor)
            xDesign = x[0:num_elems].detach().numpy()
            zD = x[num_elems:]
            zDesign = zD.view(2, -1).T
            decoded = materialEncoder.vaeNet.decoder(zDesign)
            youngsModulus, _ = materialEncoder.getMaterialProperties(decoded)
            EDesign = youngsModulus.detach().numpy()

        fe_solver.mat_prop = [
            mat_lib.create_material_with_defaults(name=f"Material_{i+1}", youngs_modulus=EDesign[i])
            for i in range(EDesign.shape[0])
        ]
        fe_solver.set_structural_material(fe_solver.mat_prop)

        print(f"Done creating material properties")
        timeFEAStart = time.time()
        sol = fe_solver.solve(xDesign, material_model)
        obj = np.einsum('i, i -> ', fe_solver.total_force, sol)

        #KS: Will need to scale by obj0
        timeFEA += time.time() - timeFEAStart
        obj = np.array([obj])

        ce = (np.dot(sol[fe_solver.mesh.edofMat].reshape(num_elems, 24), KE) * sol[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)

        penal = SIMP_PENALTY

        #KS: Derive the equations for the gradient
        dJ_dxDesign = (-penal * xDesign ** (penal - 1)) * EDesign * ce #KS: why multiply by EDesign?
        dJ_dEDesign = np.asarray((xDesign ** penal) * ce) #KS: What is this?

        
        if use_patches: #KS: Why?
            # Reduce EDesign to the size of ym by averaging over patches
            reduced_dJ_dEDesign = np.zeros(num_patches)
            for patch_id in range(num_patches):
                reduced_dJ_dEDesign[patch_id] = np.mean(dJ_dEDesign[patchwork_colors == patch_id])
            dJ_dEDesign_tensor = torch.tensor(reduced_dJ_dEDesign)
        else:
            dJ_dEDesign_tensor = torch.tensor(dJ_dEDesign)

        youngsModulus.backward(dJ_dEDesign_tensor)
        dJ_dzDesign = xTensor.grad.detach().numpy()
        if use_patches: # KS: no difference?
            grad_obj = np.concatenate((dJ_dxDesign, -dJ_dzDesign[num_elems:].flatten()))
        else:
            grad_obj = np.concatenate((dJ_dxDesign, -dJ_dzDesign[num_elems:].flatten()))

        #KS: Divide by obj0 to normalize the gradient
        #KS: Numerically verify if this is correct?


        if nodal_body_force is not None:
            ce_body_force = (sol[fe_solver.mesh.edofMat].reshape(num_elems, 24) * nodal_body_force[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
            grad_obj += 2 * ce_body_force

        # KS: Use the weighted filer (optional)
        grad_obj[0:num_elems] = (H * grad_obj[0:num_elems]) / Hs

        if elemsWithForces.size > 0:
            grad_obj[elemsWithForces] = min(grad_obj)

        if to_params.ElemsToKeep is not None:
            grad_obj[to_params.ElemsToKeep] = min(grad_obj)

        vf = np.mean(xDesign)

        # Constraint
        if use_patches: #KS: Simplify
            xConstraint_tensor = torch.tensor(x).float()
            xConstraint_tensor.requires_grad = True
            pseudoDensity = xConstraint_tensor[0:num_elems]
            zcTensor = xConstraint_tensor[num_elems:]
            zc = zcTensor.view(2, -1).T
            decoded = materialEncoder.vaeNet.decoder(zc)
            _, massDensity = materialEncoder.getMaterialProperties(decoded) #KS Why not do this earlier along with E?
            md = torch.zeros(patchwork_colors.size, dtype=torch.float32)
            for patch_id in range(num_patches):
                md[patchwork_colors == patch_id] = massDensity[patch_id]
            totalMass = torch.einsum('m,m->m', md, pseudoDensity).sum() * fe_solver.mesh.elem_size[0] ** 3
            print(f"Current mass: {totalMass.item():.4f}, Target mass: {to_params.TargetMass:.4f}")
            massConstraint = ((totalMass / to_params.TargetMass) - 1.0)
            massConstraint.backward()
            cons = massConstraint.detach().numpy()
            grad_cons = xConstraint_tensor.grad.detach().numpy() #KS: Numerically verify if this is correct?
           
        else:
            xConstraint_tensor = x.clone().detach().requires_grad_(True)
            pseudoDensity = xConstraint_tensor[0:num_elems]
            zcTensor = xConstraint_tensor[num_elems:]
            zc = zcTensor.view(2, -1).T
            decoded = materialEncoder.vaeNet.decoder(zc)
            _, massDensity = materialEncoder.getMaterialProperties(decoded)
            totalMass = torch.einsum('m,m->m', massDensity, pseudoDensity).sum() * fe_solver.mesh.elem_size[0] * fe_solver.mesh.elem_size[1] * fe_solver.mesh.elem_size[2]
            print(f"Current mass: {totalMass.item():.4f}, Target mass: {to_params.TargetMass:.4f}")
            massConstraint = ((totalMass / to_params.TargetMass) - 1.0)
            massConstraint.backward()
            cons = massConstraint.detach().numpy()
            grad_cons = xConstraint_tensor.grad.detach().numpy()

        timeMMAStart = time.time()
        mma_state = mma.update_mma(
            mma_state,
            mma_params,
            obj,
            np.array([grad_obj]).reshape((num_design_var, 1)),
            np.array([cons]).reshape((1, 1)),
            grad_cons.reshape((1, num_design_var))
        )
        timeMMA += time.time() - timeMMAStart
        if use_patches:
            change = np.max(np.abs(x - x_old))
            x_old = x
        else:
            change = np.max(np.abs(x.detach().numpy() - x_old))
            x_old = x.detach().numpy()
        print(f"it.: {mma_state.epoch}, obj.: {obj[0]:.6g} vf: {vf:.3f}", f"ch: {change:.3f}")
        history['compliance'].append(obj[0])
        history['volume'].append(np.mean(xDesign))
        history['change'].append(change)

        if (len(history['compliance'])) >= minMMAIterations:
            dJ = (history['compliance'][-1] - history['compliance'][-2]) / history['compliance'][-2]
            if abs(dJ) < rel_conv_tol and (cons) < rel_conv_tol:
                break
      
        if time.time() - tStart > timeLimit:
            success = False
            print("MMA optimization terminated due to time limit.")
            break
        if (history['compliance'][-1] > 100 * history['compliance'][0]):
            print("Optimization terminated due to large compliance increase.")
            success = False
            break

    if mma_state.epoch >= maxMMAIterations:
        print("MMA optimization did not converge.")
        success = False

    if use_patches:
        fe_solver.mesh.setPseudoDensity(x[0:num_elems])
        md[xDesign < 0.001] = 1e-3
        md = md.detach().numpy()
        EDesign[xDesign < 0.001] = 1e-3
        plt.hist(xDesign, bins=10)
        plt.hist(np.asarray(EDesign), bins=10)
        plt.show()
        return np.asarray(EDesign), history, success, zDesign.detach().cpu().numpy()
    else:
        fe_solver.mesh.setPseudoDensity(x[0:num_elems].detach().numpy())
        EDesign[xDesign < 0.001] = 1e-3
        plt.hist(xDesign, bins=10)
        plt.hist(EDesign, bins=10)
        plt.show()
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

  
  
    # --- User sets threshold here ---
    criticality_threshold = 3.0 # <-- Set your desired threshold here for criticality index
    final_compliances = []

    use_reduced_features = True  # Set to False for full feature set
    use_patches = True  # Set to True for patch-based optimization
    random_latent_init = True  # Set to True for random latent initialization
    print(f"\nProcessing {'reduced features' if use_reduced_features else f'Criticality Index < {criticality_threshold}'}")

    trainingData, dataInfo, dataIdentifier, trainInfo, EMax = preprocessData(
        criticality_threshold=criticality_threshold, use_reduced_features=use_reduced_features)
    numMaterialsInTrainingData, numFeatures = trainingData.shape

    latentDim, hiddenDim = 2, 250
    numEpochs = 40000
    klFactor = 5e-5
    learningRate = 2e-3
    savedNet = './data/vaeNet_ref.nt'
    vaeSettings = {'encoder':{'inputDim':numFeatures, 'hiddenDim':hiddenDim, 'latentDim':latentDim},
                   'decoder':{'latentDim':latentDim, 'hiddenDim':hiddenDim, 'outputDim':numFeatures}}

    materialEncoder = MaterialEncoder(trainingData, dataInfo, dataIdentifier, vaeSettings)
    train = False
    if train:
        print("Training VAE...")
        convgHistory = materialEncoder.trainAutoencoder(numEpochs, klFactor, savedNet, learningRate)
    else:
        print("Loading pre-trained VAE...")
        materialEncoder.loadAutoencoderFromFile(savedNet)
    
    predData =  materialEncoder.vaeNet(trainingData)
    zReal = materialEncoder.vaeNet.encoder.z.detach().numpy()
    enclosing_ellipse = welzl(np.array(zReal, dtype=float))
    center,a,b,t = enclosing_ellipse

    constraints = {'distance': {'isOn':False, 'center':center, 'a':a, 'b':b, 'theta':t, 'delta':0.0, 'beta':20}}  # Adjust as needed  
    materialEncoder.constraints = constraints

    to_problem = METALSTOExamples.EdgeCantilever
    solver = lin_solv.Solvers.PARDISO
    debug = False 

    mesh, mat_prop, bc, elem_body_force, to_params = getMETALSTOProblem(to_problem)
    to_params.nDOFDesired = 50000
    to_params.TargetMass = 10
    elem_body_force = None

    dsolver = deflation.DeflationSolver()
    if (solver == lin_solv.Solvers.DPCG):
        nGroups =  min(dsolver.maxGroups, max(dsolver.minGroups, round(3*mesh.num_nodes/dsolver.dofPerGroup)))
        dsolver.create_deflation_groups(mesh, nGroups)
        dsolver.create_delfation_matrix(mesh)
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



    fe_solver.plot_mesh()
    startTime = time.time()
    u, history, success, zDesign = topopt_mma_lsr_combined(
        fe_solver=fe_solver,
        vae_info=materialEncoder,
        to_params=to_params,
        debug=debug, use_patches=use_patches, random_latent_init=random_latent_init
    )


    timeTaken = time.time() - startTime

    final_compliances.append(history['compliance'][-1])
    fe_solver.plot_elem_field(u, title=f"Youngs Modulus for Criticality Index < {criticality_threshold} (Time: {timeTaken:.2f} s)")

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
        pickle.dump({'thresholds': [criticality_threshold], 'final_compliances': final_compliances}, f)

    print("Saved compliance vs. criticality index threshold to compliance_vs_criticality_EdgeCantilever_Temp.pkl")

    # Plotting compliance vs threshold
    plt.figure()
    plt.plot([threshold], final_compliances, marker='o')
    plt.xlabel('Criticality Index Threshold')
    plt.ylabel('Final Compliance')
    plt.title('Final Compliance vs. Criticality Index Threshold')
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.show()