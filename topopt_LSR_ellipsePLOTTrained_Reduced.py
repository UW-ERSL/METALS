
from LSRImports import *

_LARGE_NUMBER = 1.e9



def topopt_mma_lsr(fe_solver,
			   			to_params,
							 vae_info: None,
			   			minMMAIterations: int = 5,
			   			 maxMMAIterations: int = 100, 
							timeLimit: float =3600, #1 hour
						   penal: float = 3.0,
							 move_limit: float = 0.2,
							 kkt_tol: float = 1.e-6,
							 move_tol: float = 0.025,
							 continuationScheme: bool = False,	 
							 rel_conv_tol: float = 1.e-5,
							 debug: bool = False,
							 ) -> tuple[np.ndarray, dict]:
	"""MMA based topology optimization for minimum compliance.
	Args:
		fe_solver: The structural FEA solver object.
		maxMMAIterations: Maximum number of MMA iterations.
		volfrac: The target volume fraction.
		penal: The penalization factor for the SIMP method.
		move_limit: The maximum change allowed for the design variables in each
			iteration.
		kkt_tol: The tolerance for the KKT conditions.
		step_tol: The tolerance for the step size.

	Returns: The displacement field of the optimized structure.
	"""
	num_elems = fe_solver.mesh.num_elems
	num_design_var = num_elems*3
	material_model = MaterialModel.SIMP #For no body forces, using SIMP material model


	tStart = time.time()
	num_elems= fe_solver.mesh.num_elems
	history = {'compliance': [], 'volume': [], 'change': []}
	
	[H,Hs] = createFilters(fe_solver, to_params)

	elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force,3)

	mma_params = mma.MMAParams(max_iter=maxMMAIterations,
														kkt_tol = kkt_tol,
														step_tol = move_tol,
														move_limit = move_limit,
														num_design_var = num_design_var,
														num_cons = 1,
														lower_bound = np.zeros((num_design_var, 1)),
														upper_bound = np.ones((num_design_var, 1)),
														)
	constraintType = to_params.Constraints[0][0] # assume this is the first constraint
	if (constraintType == TO_QOI.VOLUME_FRACTION):
		volFractionConstraint = to_params.Constraints[0][2]
	else:
		volFractionConstraint =1 # default value
	print(f"volFractionConstraint: {volFractionConstraint:.3f}")
	mma_init = np.concatenate((0.5 * np.ones((num_elems, 1)), 0.0 * np.ones((2*num_elems, 1))), axis = 0)
	mma_state = mma.init_mma(mma_init, mma_params)
	# mma_state = mma.init_mma(0.5 * np.ones((num_design_var, 1)), mma_params)
	# KE = elem_stiff.hex8_stiffness_matrix_structural( fe_solver.mat_prop,fe_solver.mesh.elem_size)
	print(f"MAT PROPS: {fe_solver.mat_prop}")
	if isinstance(fe_solver.mat_prop, list): # multiple materials
		if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
			KE_list = [hex_element_stiffness.hex8_stiffness_matrix_structural( mp.youngs_modulus,mp.poissons_ratio,fe_solver.mesh.elem_size)
				for mp in fe_solver.mat_prop]
			KE = KE_list[0]
		elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
			KE_list = [hex_element_stiffness.hex8_stiffness_matrix_thermal( mp.thermal_conductivity,fe_solver.mesh.elem_size)
				for mp in fe_solver.mat_prop]
			KE = KE_list[0]	
		print("Assuming all elements have the same material properties")
	else: # single material
		if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
			KE = hex_element_stiffness.hex8_stiffness_matrix_structural( fe_solver.mat_prop.youngs_modulus,
															    fe_solver.mat_prop.poissons_ratio,
																fe_solver.mesh.elem_size)
		elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
			KE = hex_element_stiffness.hex8_stiffness_matrix_thermal( fe_solver.mat_prop.thermal_conductivity,fe_solver.mesh.elem_size)
	
	print(f"KE shape: {KE.shape}")
	x_old = volFractionConstraint*np.ones(num_design_var, dtype = float)
	timeFEA = 0
	timeMMA = 0
	if (fe_solver.elem_body_force is not None):
		elem_force = fe_solver.elem_body_force.copy()
		nNodes = fe_solver.mesh.num_nodes
		nodal_body_force = np.zeros((nNodes * 3,))
		nodal_body_force[0::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[0::3]
		nodal_body_force[1::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[1::3]
		nodal_body_force[2::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[2::3]
	else:
		nodal_body_force = None
	if (continuationScheme):
		penal = 1.2
	
	success = True
	
	while not mma_state.is_converged:
		x = mma_state.x.reshape(-1)
		xTensor = torch.tensor(x).float()
		xTensor.requires_grad = True
		x = vae_info.map_to_ellipse_torch(xTensor)
		xDesign = x[0:num_elems].detach().numpy()
		zD = x[num_elems:]
		zDesign = zD.view(2,-1).T
		decoded = materialEncoder.vaeNet.decoder(zDesign)
		youngsModulus,_ = materialEncoder.getMaterialProperties(decoded)
		EDesign = youngsModulus.detach().numpy()

		fe_solver.mat_prop = [
			mat_lib.create_material_with_defaults(name=f"Material_{i+1}", youngs_modulus=EDesign[i])
			for i in range(EDesign.shape[0])
		]

		# Pass the updated mat_prop to set_structural_material
		print(f"Setting material properties")
		fe_solver.set_structural_material(fe_solver.mat_prop)

		print(f"Done with material properties")
		timeFEAStart = time.time()
		sol = fe_solver.solve(xDesign, material_model)
		# obj, grad_obj = compliance(sol,x, fe_solver,KE, material_model)
	
		# obj, grad_obj = compute_objective_and_gradient(to_params,sol,xDesign, fe_solver,fe_solver.elem_stiff, material_model)
		obj=np.einsum('i, i -> ', fe_solver.total_force, sol)
		timeFEA += time.time() - timeFEAStart
		obj = np.array([obj])

		ce = (np.dot(sol[fe_solver.mesh.edofMat].reshape(num_elems, 24), KE) * sol[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
		
		# For SIMP material model: x**penal
		penal = SIMP_PENALTY
		dJ_dxDesign = (-penal * xDesign ** (penal - 1)) * EDesign * ce
		dJ_dEDesign = np.asarray((xDesign ** penal) * ce)
		dJ_dEDesign_tensor = torch.tensor(dJ_dEDesign)
		youngsModulus.backward(dJ_dEDesign_tensor)
		dJ_dzDesign = xTensor.grad.detach().numpy()
		grad_obj = np.concatenate((dJ_dxDesign, -dJ_dzDesign[num_elems:].flatten()))
		# grad_obj = (-penal * x ** (penal - 1)) * ce

			
		if (nodal_body_force is not None):
			ce_body_force = (sol[fe_solver.mesh.edofMat].reshape(num_elems, 24) * nodal_body_force[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
			grad_obj +=  2*ce_body_force # Assumes body force is linear w.r.t. x : https://doi.org/10.1002/nme.2499 , https://doi.org/10.1016/j.cma.2017.04.021 
	
		grad_obj[0:num_elems] = (H * grad_obj[0:num_elems])/Hs

		if (elemsWithForces.size > 0):
			grad_obj[elemsWithForces] = min(grad_obj)


		# print(f"TO PARAMS ELEMS TO KEEP",to_params.ElemsToKeep)
		# to_params.ElemsToKeep = None
		if (to_params.ElemsToKeep is not None):
			grad_obj[to_params.ElemsToKeep] = min(grad_obj)
			#x[to_params.ElemsToKeep] = 1.0

		vf = np.mean(xDesign)
		# cons = _volume_constraint(xDesign, to_params.DesiredVolFraction)
		# grad_cons = np.zeros_like(x)
		# grad_cons[0:num_elems] = np.ones(num_elems)/to_params.DesiredVolFraction/num_elems
		# print(f"elem_size: {fe_solver.mesh.elem_size}")

		# if to_params.TargetMass is not None:
		xConstraint_tensor = x.clone().detach().requires_grad_(True)
		# xConstraint_tensor.requires_grad = True
		pseudoDensity = xConstraint_tensor[0:num_elems]
		zcTensor = xConstraint_tensor[num_elems:]
		zc = zcTensor.view(2,-1).T
		decoded = materialEncoder.vaeNet.decoder(zc)
		_, massDensity = materialEncoder.getMaterialProperties(decoded)
		totalMass = torch.einsum('m,m->m',massDensity, pseudoDensity).sum() * fe_solver.mesh.elem_size[0]* fe_solver.mesh.elem_size[1]* fe_solver.mesh.elem_size[2]
		massConstraint = ((totalMass/to_params.TargetMass) - 1.0)
		massConstraint.backward()
		cons = massConstraint.detach().numpy()
		print(f"totalMass: {totalMass:.3f}, TargetMass: {to_params.TargetMass:.3f}")
		grad_cons = xConstraint_tensor.grad.detach().numpy()
	
		
		timeMMAStart = time.time()
		mma_state = mma.update_mma(mma_state,
														   mma_params,
														 	 obj,
															 np.array([grad_obj]).reshape((num_design_var, 1)),
														 	 np.array([cons]).reshape((1, 1)),
															 grad_cons.reshape((1, num_design_var))
															 )
		timeMMA += time.time() - timeMMAStart
		change = np.max(np.abs(x.detach().numpy() - x_old))
		x_old = x.detach().numpy()
		print(f"it.: {mma_state.epoch}, obj.: {obj[0]:.6g} vf: {vf:.3f}",
					f"ch: {change:.3f}")
		history['compliance'].append(obj[0])
		history['volume'].append(np.mean(x.detach().numpy()))
		history['change'].append(change)

		if (len(history['compliance'])) >= minMMAIterations:
			dJ = (history['compliance'][-1] - history['compliance'][-2]) / history['compliance'][-2]
			if abs(dJ) < rel_conv_tol and (cons) < rel_conv_tol:
				break
		if (continuationScheme):
			penal *= 1.1
			penal = min(penal, 3.0)
		if time.time() - tStart > timeLimit:
			success = False
			print("MMA optimization terminated due to time limit.")
			break
		if (history['compliance'][-1] > 100*history['compliance'][0]):
			print("Optimization terminated due to large compliance increase.")
			success = False
			break

	if mma_state.epoch >= maxMMAIterations:
		print("MMA optimization did not converge.")
		success = False
		
	fe_solver.mesh.setPseudoDensity(x[0:num_elems].detach().numpy())
	print(f"Time FEA: {timeFEA:.2f} s, Time MMA: {timeMMA:.2f} s")
	print(f"Total Time: {timeFEA+timeMMA:.2f} s")
	EDesign[xDesign < 0.001] = 1e-3
	plt.hist(xDesign, bins = 10)
	plt.hist(EDesign, bins = 10)
	plt.show()
	return np.asarray(EDesign), history,success

import pickle

def preprocessData(criticality_threshold=None, use_reduced_features=False):
    df = pd.read_excel('./data/TeledyneDatabase2.xlsx')
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
        feature_names = ['Density', 'ElasticModulus']
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




if __name__ == "__main__":    
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from matplotlib.patches import Ellipse

    # thresholds = [2.55,1.5,1.25,1,0.75,0.5] 
    thresholds = [2.55]  # Criticality Index thresholds
    ellipse_params = []
    zReal_list = []
    numEpochs = 2000
    klFactor = 0.01
    savedNet = None
    learningRate = 1e-3
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
        vaeSettings = {'encoder':{'inputDim':numFeatures, 'hiddenDim':hiddenDim, 'latentDim':latentDim},\
                    'decoder':{'latentDim':latentDim, 'hiddenDim':hiddenDim, 'outputDim':numFeatures}}

        materialEncoder = MaterialEncoder(trainingData, dataInfo, dataIdentifier, vaeSettings)
        constraints = {}  # No constraints for initial training
        materialEncoder.constraints = constraints
        convgHistory = materialEncoder.trainAutoencoder(numEpochs, klFactor, savedNet, learningRate)

        predData = materialEncoder.vaeNet(trainingData)
        zReal = materialEncoder.vaeNet.encoder.z.detach().numpy()
        zReal_list.append(zReal)
        enclosing_ellipse = welzl(np.array(zReal, dtype=float))
        ellipse_params.append(enclosing_ellipse)

    # Plot all ellipses and their points
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
    for i, (zReal, (center, a, b, t)) in enumerate(zip(zReal_list, ellipse_params)):
        ax.plot(zReal[:, 0], zReal[:, 1], marker='*', linestyle='', color=colors[i], label=f'Threshold < {thresholds[i]}')
        ellipse = Ellipse(xy=center, width=2 * a, height=2 * b, angle=np.degrees(t),
                          edgecolor=colors[i], fc='none', lw=2, label=f'Ellipse {thresholds[i]}')
        ax.add_patch(ellipse)

    ax.set_aspect('equal', adjustable='datalim')
    ax.legend()
    plt.title('Latent Space Ellipses for Different Criticality Thresholds')
    plt.tight_layout()
    plt.show()
