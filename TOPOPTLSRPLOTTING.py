
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

def preprocessData(criticality_threshold=None, feature_mode="density_youngs"):
    df = pd.read_excel('./data/TeledyneDatabase2.xlsx')
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

    if feature_mode == "density_youngs":
        rawData = df.iloc[:, [5, 10]].to_numpy()
        feature_names = ['MassDensity', 'ElasticModulus']
        YoungsModulus = rawData[:, 1]
    elif feature_mode == "density_youngs_yield":
        rawData = df.iloc[:, [5, 4, 10]].to_numpy()
        feature_names = ['MassDensity', 'YieldStress', 'ElasticModulus']
        YoungsModulus = rawData[:, 2]
    elif feature_mode == "all_but_criticality":
        rawData = df.iloc[:, 3:-1].to_numpy()
        feature_names = [
            'UltimateStrength', 'YieldStress', 'MassDensity', 'CostPerPound',
            'MeltingTempC', 'MaxUseTempC', 'Elong2Fail', 'ElasticModulus'
        ]
        YoungsModulus = df.iloc[:, 10].to_numpy()
    elif feature_mode == "all_including_criticality":
        rawData = df.iloc[:, 3:].to_numpy()
        feature_names = [
            'UltimateStrength', 'YieldStress', 'MassDensity', 'CostPerPound',
            'MeltingTempC', 'MaxUseTempC', 'Elong2Fail', 'ElasticModulus', 'CriticalityIdx'
        ]
        YoungsModulus = df.iloc[:, 10].to_numpy()
    else:
        raise ValueError("Unknown feature_mode")

    EMax = np.max(YoungsModulus)
    print("Max E: ", EMax, " GPa")

    trainInfo = np.log10(rawData)
    dataScaleMax = torch.tensor(np.max(trainInfo, axis=0))
    dataScaleMin = torch.tensor(np.min(trainInfo, axis=0))
    normalizedData = (torch.tensor(trainInfo) - dataScaleMin) / (dataScaleMax - dataScaleMin)
    trainingData = normalizedData.clone().float()

    # Use explicit dataInfo mapping for all_but_criticality and all_including_criticality
    if feature_mode == "all_but_criticality":
        dataInfo = {
            'UltimateStrength': {'idx': 0, 'scaleMin': dataScaleMin[0], 'scaleMax': dataScaleMax[0]},
            'YieldStress':      {'idx': 1, 'scaleMin': dataScaleMin[1], 'scaleMax': dataScaleMax[1]},
            'MassDensity':      {'idx': 2, 'scaleMin': dataScaleMin[2], 'scaleMax': dataScaleMax[2]},
            'CostPerPound':     {'idx': 3, 'scaleMin': dataScaleMin[3], 'scaleMax': dataScaleMax[3]},
            'MeltingTempC':     {'idx': 4, 'scaleMin': dataScaleMin[4], 'scaleMax': dataScaleMax[4]},
            'MaxUseTempC':      {'idx': 5, 'scaleMin': dataScaleMin[5], 'scaleMax': dataScaleMax[5]},
            'Elong2Fail':       {'idx': 6, 'scaleMin': dataScaleMin[6], 'scaleMax': dataScaleMax[6]},
            'ElasticModulus':   {'idx': 7, 'scaleMin': dataScaleMin[7], 'scaleMax': dataScaleMax[7]},
        }
    elif feature_mode == "all_including_criticality":
        dataInfo = {
            'UltimateStrength': {'idx': 0, 'scaleMin': dataScaleMin[0], 'scaleMax': dataScaleMax[0]},
            'YieldStress':      {'idx': 1, 'scaleMin': dataScaleMin[1], 'scaleMax': dataScaleMax[1]},
            'MassDensity':      {'idx': 2, 'scaleMin': dataScaleMin[2], 'scaleMax': dataScaleMax[2]},
            'CostPerPound':     {'idx': 3, 'scaleMin': dataScaleMin[3], 'scaleMax': dataScaleMax[3]},
            'MeltingTempC':     {'idx': 4, 'scaleMin': dataScaleMin[4], 'scaleMax': dataScaleMax[4]},
            'MaxUseTempC':      {'idx': 5, 'scaleMin': dataScaleMin[5], 'scaleMax': dataScaleMax[5]},
            'Elong2Fail':       {'idx': 6, 'scaleMin': dataScaleMin[6], 'scaleMax': dataScaleMax[6]},
            'ElasticModulus':   {'idx': 7, 'scaleMin': dataScaleMin[7], 'scaleMax': dataScaleMax[7]},
            'CriticalityIdx':   {'idx': 8, 'scaleMin': dataScaleMin[8], 'scaleMax': dataScaleMax[8]},
        }
    else:
        dataInfo = {}
        for i, name in enumerate(feature_names):
            dataInfo[name] = {'idx': i, 'scaleMin': dataScaleMin[i], 'scaleMax': dataScaleMax[i]}

    return trainingData, dataInfo, dataIdentifier, trainInfo, EMax




# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import pandas as pd
#     import numpy as np
#     import torch

#     feature_modes = [
#         "density_youngs",
#         "density_youngs_yield",
#         "all_but_criticality",
#         "all_including_criticality"
#     ]
#     feature_mode_styles = {
#         "density_youngs":           {"color": "tab:blue",  "marker": "*", "label": "E & density"},
#         "density_youngs_yield":     {"color": "tab:orange","marker": "s", "label": "E, density, yield"},
#         "all_but_criticality":      {"color": "tab:green", "marker": "o", "label": "All except criticality"},
#         "all_including_criticality":{"color": "tab:red",   "marker": "D", "label": "All attributes"},
#     }
#     thresholds = [2.55]

#     def unlognorm(x, scaleMax, scaleMin):
#         return 10.**(x * (scaleMax - scaleMin) + scaleMin)

#     def sample_points_in_ellipse(center, a, b, theta, n_points=1000):
#         r = np.sqrt(np.random.uniform(0, 1, n_points))
#         angles = np.random.uniform(0, 2 * np.pi, n_points)
#         xs = r * np.cos(angles)
#         ys = r * np.sin(angles)
#         xs = a * xs
#         ys = b * ys
#         rot = np.array([[np.cos(theta), -np.sin(theta)],
#                         [np.sin(theta),  np.cos(theta)]])
#         points = np.dot(np.stack([xs, ys], axis=1), rot.T)
#         points += np.array(center)
#         return points

#     all_sampled = []
#     all_sampled_labels = []
#     all_sampled_markers = []
#     all_sampled_colors = []

#     for feature_mode in feature_modes:
#         for threshold in thresholds:
#             trainingData, dataInfo, dataIdentifier, trainInfo, EMax = preprocessData(
#                 criticality_threshold=threshold, feature_mode=feature_mode)
#             latentDim, hiddenDim = 2, 250
#             numEpochs = 40000
#             klFactor = 5e-5
#             learningRate = 2e-3
#             savedNet = './data/vaeNet_ref.nt'
#             vaeSettings = {'encoder': {'inputDim': trainingData.shape[1], 'hiddenDim': hiddenDim, 'latentDim': latentDim},
#                            'decoder': {'latentDim': latentDim, 'hiddenDim': hiddenDim, 'outputDim': trainingData.shape[1]}}
#             materialEncoder = MaterialEncoder(trainingData, dataInfo, dataIdentifier, vaeSettings)
#             materialEncoder.constraints = {}
#             materialEncoder.trainAutoencoder(numEpochs, klFactor, savedNet, learningRate)

#             zReal = materialEncoder.vaeNet.encoder.z.detach().numpy()
#             enclosing_ellipse = welzl(np.array(zReal, dtype=float))
#             center, a, b, t = enclosing_ellipse
#             sampled_points = sample_points_in_ellipse(center, a, b, t, n_points=500)
#             points_tensor = torch.tensor(sampled_points, dtype=torch.float32)
#             decoded = materialEncoder.vaeNet.decoder(points_tensor)
#             density_sampled = unlognorm(
#                 decoded[:, dataInfo['MassDensity']['idx']],
#                 dataInfo['MassDensity']['scaleMax'],
#                 dataInfo['MassDensity']['scaleMin']
#             ).detach().numpy()
#             youngs_sampled = unlognorm(
#                 decoded[:, dataInfo['ElasticModulus']['idx']],
#                 dataInfo['ElasticModulus']['scaleMax'],
#                 dataInfo['ElasticModulus']['scaleMin']
#             ).detach().numpy()
#             all_sampled.append((density_sampled, youngs_sampled))
#             all_sampled_labels.append(feature_mode_styles[feature_mode]["label"])
#             all_sampled_markers.append(feature_mode_styles[feature_mode]["marker"])
#             all_sampled_colors.append(feature_mode_styles[feature_mode]["color"])

#     # Plot combined property space scatter plot for all feature modes
#     plt.figure(figsize=(12, 10))
#     for i, (density_sampled, youngs_sampled) in enumerate(all_sampled):
#         plt.scatter(
#             density_sampled, youngs_sampled,
#             c=all_sampled_colors[i], marker=all_sampled_markers[i], s=60, alpha=0.5, label=all_sampled_labels[i]
#         )
#     # Plot real materials only once (in black)
#     df = pd.read_excel('./data/TeledyneDatabase2.xlsx')
#     density_real = df.iloc[:, 5].to_numpy()
#     youngs_real = df.iloc[:, 10].to_numpy()
#     plt.scatter(density_real, youngs_real, c='k', s=120, label='Real Materials', alpha=0.7, edgecolors='w')

#     plt.xlabel('Density', fontsize=24)
#     plt.ylabel("Young's Modulus", fontsize=24)
#     plt.title("Sampled Materials in Property Space (All Feature Modes)", fontsize=24)
#     plt.legend(fontsize=16)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.show()

# --- MAIN FUNCTION 1: JITTERED COMBINED PLOT ---
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import pandas as pd
#     import numpy as np
#     import torch

#     feature_modes = [
#         "density_youngs",
#         "density_youngs_yield",
#         "all_but_criticality",
#         "all_including_criticality"
#     ]
#     feature_mode_styles = {
#         "density_youngs":           {"color": "tab:blue",  "marker": "*", "label": "E & density"},
#         "density_youngs_yield":     {"color": "tab:orange","marker": "s", "label": "E, density, yield"},
#         "all_but_criticality":      {"color": "tab:green", "marker": "o", "label": "All except criticality"},
#         "all_including_criticality":{"color": "tab:red",   "marker": "D", "label": "All attributes"},
#     }
#     thresholds = [2.55]

#     def unlognorm(x, scaleMax, scaleMin):
#         return 10.**(x * (scaleMax - scaleMin) + scaleMin)

#     def sample_points_in_ellipse(center, a, b, theta, n_points=1000):
#         r = np.sqrt(np.random.uniform(0, 1, n_points))
#         angles = np.random.uniform(0, 2 * np.pi, n_points)
#         xs = r * np.cos(angles)
#         ys = r * np.sin(angles)
#         xs = a * xs
#         ys = b * ys
#         rot = np.array([[np.cos(theta), -np.sin(theta)],
#                         [np.sin(theta),  np.cos(theta)]])
#         points = np.dot(np.stack([xs, ys], axis=1), rot.T)
#         points += np.array(center)
#         return points

#     all_sampled = []
#     all_sampled_labels = []
#     all_sampled_markers = []
#     all_sampled_colors = []

#     for feature_mode in feature_modes:
#         for threshold in thresholds:
#             trainingData, dataInfo, dataIdentifier, trainInfo, EMax = preprocessData(
#                 criticality_threshold=threshold, feature_mode=feature_mode)
#             latentDim, hiddenDim = 2, 250
#             numEpochs = 40000
#             klFactor = 5e-5
#             learningRate = 2e-3
#             savedNet = './data/vaeNet_ref.nt'
#             vaeSettings = {'encoder': {'inputDim': trainingData.shape[1], 'hiddenDim': hiddenDim, 'latentDim': latentDim},
#                            'decoder': {'latentDim': latentDim, 'hiddenDim': hiddenDim, 'outputDim': trainingData.shape[1]}}
#             materialEncoder = MaterialEncoder(trainingData, dataInfo, dataIdentifier, vaeSettings)
#             materialEncoder.constraints = {}
#             materialEncoder.trainAutoencoder(numEpochs, klFactor, savedNet, learningRate)

#             zReal = materialEncoder.vaeNet.encoder.z.detach().numpy()
#             enclosing_ellipse = welzl(np.array(zReal, dtype=float))
#             center, a, b, t = enclosing_ellipse
#             sampled_points = sample_points_in_ellipse(center, a, b, t, n_points=500)
#             points_tensor = torch.tensor(sampled_points, dtype=torch.float32)
#             decoded = materialEncoder.vaeNet.decoder(points_tensor)
#             density_sampled = unlognorm(
#                 decoded[:, dataInfo['MassDensity']['idx']],
#                 dataInfo['MassDensity']['scaleMax'],
#                 dataInfo['MassDensity']['scaleMin']
#             ).detach().numpy()
#             youngs_sampled = unlognorm(
#                 decoded[:, dataInfo['ElasticModulus']['idx']],
#                 dataInfo['ElasticModulus']['scaleMax'],
#                 dataInfo['ElasticModulus']['scaleMin']
#             ).detach().numpy()
#             all_sampled.append((density_sampled, youngs_sampled))
#             all_sampled_labels.append(feature_mode_styles[feature_mode]["label"])
#             all_sampled_markers.append(feature_mode_styles[feature_mode]["marker"])
#             all_sampled_colors.append(feature_mode_styles[feature_mode]["color"])

#     # Plot combined property space scatter plot for all feature modes with jitter
#     plt.figure(figsize=(12, 10))
#     jitter_scale = 0.01  # 1% of data range
#     for i, (density_sampled, youngs_sampled) in enumerate(all_sampled):
#         rng = np.random.default_rng(i)
#         dx = jitter_scale * (density_sampled.max() - density_sampled.min())
#         dy = jitter_scale * (youngs_sampled.max() - youngs_sampled.min())
#         plt.scatter(
#             density_sampled + rng.uniform(-dx, dx, size=density_sampled.shape),
#             youngs_sampled + rng.uniform(-dy, dy, size=youngs_sampled.shape),
#             c=all_sampled_colors[i], marker=all_sampled_markers[i], s=60, alpha=0.5, label=all_sampled_labels[i]
#         )
#     # Plot real materials only once (in black)
#     df = pd.read_excel('./data/TeledyneDatabase2.xlsx')
#     density_real = df.iloc[:, 5].to_numpy()
#     youngs_real = df.iloc[:, 10].to_numpy()
#     plt.scatter(density_real, youngs_real, c='k', s=120, label='Real Materials', alpha=0.7, edgecolors='w')

#     plt.xlabel('Density', fontsize=24)
#     plt.ylabel("Young's Modulus", fontsize=24)
#     plt.title("Sampled Materials in Property Space (All Feature Modes, Jittered)", fontsize=24)
#     plt.legend(fontsize=16)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.show()


# --- MAIN FUNCTION 2: FACET SUBPLOTS ---
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import pandas as pd
#     import numpy as np
#     import torch

#     # Define the order you want
#     feature_modes = [
#         "all_including_criticality",
#         "all_but_criticality",
#         "density_youngs_yield",
#         "density_youngs"
#     ]
#     feature_mode_styles = {
#         "density_youngs":           {"color": "tab:blue",  "marker": "*", "label": "E & density"},
#         "density_youngs_yield":     {"color": "tab:orange","marker": "s", "label": "E, density, yield"},
#         "all_but_criticality":      {"color": "tab:green", "marker": "o", "label": "All except criticality"},
#         "all_including_criticality":{"color": "tab:red",   "marker": "D", "label": "All attributes"},
#     }
#     thresholds = [2.55]

#     def unlognorm(x, scaleMax, scaleMin):
#         return 10.**(x * (scaleMax - scaleMin) + scaleMin)

#     def sample_points_in_ellipse(center, a, b, theta, n_points=1000):
#         r = np.sqrt(np.random.uniform(0, 1, n_points))
#         angles = np.random.uniform(0, 2 * np.pi, n_points)
#         xs = r * np.cos(angles)
#         ys = r * np.sin(angles)
#         xs = a * xs
#         ys = b * ys
#         rot = np.array([[np.cos(theta), -np.sin(theta)],
#                         [np.sin(theta),  np.cos(theta)]])
#         points = np.dot(np.stack([xs, ys], axis=1), rot.T)
#         points += np.array(center)
#         return points

#     all_sampled = []
#     all_sampled_labels = []
#     all_sampled_markers = []
#     all_sampled_colors = []

#     for feature_mode in feature_modes:
#         for threshold in thresholds:
#             trainingData, dataInfo, dataIdentifier, trainInfo, EMax = preprocessData(
#                 criticality_threshold=threshold, feature_mode=feature_mode)
#             latentDim, hiddenDim = 2, 250
#             numEpochs = 40000
#             klFactor = 5e-5
#             learningRate = 2e-3
#             savedNet = './data/vaeNet_ref.nt'
#             vaeSettings = {'encoder': {'inputDim': trainingData.shape[1], 'hiddenDim': hiddenDim, 'latentDim': latentDim},
#                            'decoder': {'latentDim': latentDim, 'hiddenDim': hiddenDim, 'outputDim': trainingData.shape[1]}}
#             materialEncoder = MaterialEncoder(trainingData, dataInfo, dataIdentifier, vaeSettings)
#             materialEncoder.constraints = {}
#             materialEncoder.trainAutoencoder(numEpochs, klFactor, savedNet, learningRate)

#             zReal = materialEncoder.vaeNet.encoder.z.detach().numpy()
#             enclosing_ellipse = welzl(np.array(zReal, dtype=float))
#             center, a, b, t = enclosing_ellipse
#             sampled_points = sample_points_in_ellipse(center, a, b, t, n_points=500)
#             points_tensor = torch.tensor(sampled_points, dtype=torch.float32)
#             decoded = materialEncoder.vaeNet.decoder(points_tensor)
#             density_sampled = unlognorm(
#                 decoded[:, dataInfo['MassDensity']['idx']],
#                 dataInfo['MassDensity']['scaleMax'],
#                 dataInfo['MassDensity']['scaleMin']
#             ).detach().numpy()
#             youngs_sampled = unlognorm(
#                 decoded[:, dataInfo['ElasticModulus']['idx']],
#                 dataInfo['ElasticModulus']['scaleMax'],
#                 dataInfo['ElasticModulus']['scaleMin']
#             ).detach().numpy()
#             all_sampled.append((density_sampled, youngs_sampled))
#             all_sampled_labels.append(feature_mode_styles[feature_mode]["label"])
#             all_sampled_markers.append(feature_mode_styles[feature_mode]["marker"])
#             all_sampled_colors.append(feature_mode_styles[feature_mode]["color"])

#     # Plot facet grid (2x2) for all feature modes 
#     fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
#     df = pd.read_excel('./data/TeledyneDatabase2.xlsx')
#     density_real = df.iloc[:, 5].to_numpy()
#     youngs_real = df.iloc[:, 10].to_numpy()
#     for i, (density_sampled, youngs_sampled) in enumerate(all_sampled):
#         ax = axs.flat[i]
#         # Real materials in black
#         ax.scatter(density_real, youngs_real, c='k', s=120, label='Real Materials', alpha=1.0, edgecolors='w')
#         marker = all_sampled_markers[i]
#         if marker == '*':
#             marker_size = 35
#         else:
#             marker_size = 20  # Reduce size for squares, circles, diamonds
#         ax.scatter(
#             density_sampled, youngs_sampled,
#             c=all_sampled_colors[i], marker=marker, s=marker_size, alpha=0.7, label=all_sampled_labels[i]
#         )
#         ax.set_title(all_sampled_labels[i], fontsize=18)
#         ax.grid(True)
#         ax.set_xlabel('Density')
#         ax.set_ylabel("Young's Modulus")
#         ax.legend(fontsize=12)
#     plt.tight_layout()
#     plt.show()
	
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import pandas as pd
#     import numpy as np
#     import torch

#     # Define the order you want
#     feature_modes = [
#         "all_including_criticality",
#         "all_but_criticality",
#         "density_youngs_yield",
#         "density_youngs"
#     ]
#     feature_mode_styles = {
#         "density_youngs":           {"color": "gray",      "marker": "*", "label": "E & density"},
#         "density_youngs_yield":     {"color": "dimgray",   "marker": "s", "label": "E, density, yield"},
#         "all_but_criticality":      {"color": "darkgray",  "marker": "o", "label": "All except criticality"},
#         "all_including_criticality":{"color": "tab:red",   "marker": "D", "label": "All attributes"},
#     }
#     thresholds = [2.55]

#     def unlognorm(x, scaleMax, scaleMin):
#         return 10.**(x * (scaleMax - scaleMin) + scaleMin)

#     def sample_points_in_ellipse(center, a, b, theta, n_points=1000):
#         r = np.sqrt(np.random.uniform(0, 1, n_points))
#         angles = np.random.uniform(0, 2 * np.pi, n_points)
#         xs = r * np.cos(angles)
#         ys = r * np.sin(angles)
#         xs = a * xs
#         ys = b * ys
#         rot = np.array([[np.cos(theta), -np.sin(theta)],
#                         [np.sin(theta),  np.cos(theta)]])
#         points = np.dot(np.stack([xs, ys], axis=1), rot.T)
#         points += np.array(center)
#         return points

#     all_sampled = []
#     all_sampled_labels = []
#     all_sampled_markers = []
#     all_sampled_colors = []

#     for feature_mode in feature_modes:
#         for threshold in thresholds:
#             trainingData, dataInfo, dataIdentifier, trainInfo, EMax = preprocessData(
#                 criticality_threshold=threshold, feature_mode=feature_mode)
#             latentDim, hiddenDim = 2, 250
#             numEpochs = 40000
#             klFactor = 5e-5
#             learningRate = 2e-3
#             savedNet = './data/vaeNet_ref.nt'
#             vaeSettings = {'encoder': {'inputDim': trainingData.shape[1], 'hiddenDim': hiddenDim, 'latentDim': latentDim},
#                            'decoder': {'latentDim': latentDim, 'hiddenDim': hiddenDim, 'outputDim': trainingData.shape[1]}}
#             materialEncoder = MaterialEncoder(trainingData, dataInfo, dataIdentifier, vaeSettings)
#             materialEncoder.constraints = {}
#             materialEncoder.trainAutoencoder(numEpochs, klFactor, savedNet, learningRate)

#             zReal = materialEncoder.vaeNet.encoder.z.detach().numpy()
#             enclosing_ellipse = welzl(np.array(zReal, dtype=float))
#             center, a, b, t = enclosing_ellipse
#             sampled_points = sample_points_in_ellipse(center, a, b, t, n_points=500)
#             points_tensor = torch.tensor(sampled_points, dtype=torch.float32)
#             decoded = materialEncoder.vaeNet.decoder(points_tensor)
#             density_sampled = unlognorm(
#                 decoded[:, dataInfo['MassDensity']['idx']],
#                 dataInfo['MassDensity']['scaleMax'],
#                 dataInfo['MassDensity']['scaleMin']
#             ).detach().numpy()
#             youngs_sampled = unlognorm(
#                 decoded[:, dataInfo['ElasticModulus']['idx']],
#                 dataInfo['ElasticModulus']['scaleMax'],
#                 dataInfo['ElasticModulus']['scaleMin']
#             ).detach().numpy()
#             all_sampled.append((density_sampled, youngs_sampled))
#             all_sampled_labels.append(feature_mode_styles[feature_mode]["label"])
#             all_sampled_markers.append(feature_mode_styles[feature_mode]["marker"])
#             all_sampled_colors.append(feature_mode_styles[feature_mode]["color"])

#     # Plot combined property space scatter plot for all feature modes
#     plt.figure(figsize=(12, 10))
#     for i, (density_sampled, youngs_sampled) in enumerate(all_sampled):
#         # Only the first (all_including_criticality) is in color, others are grayscale
#         plt.scatter(
#             density_sampled, youngs_sampled,
#             c=all_sampled_colors[i], marker=all_sampled_markers[i], s=60, alpha=0.7, label=all_sampled_labels[i]
#         )
#     # Plot real materials only once (in black)
#     df = pd.read_excel('./data/TeledyneDatabase2.xlsx')
#     density_real = df.iloc[:, 5].to_numpy()
#     youngs_real = df.iloc[:, 10].to_numpy()
#     plt.scatter(density_real, youngs_real, c='k', s=120, label='Real Materials', alpha=0.7, edgecolors='w')

#     plt.xlabel('Density', fontsize=24)
#     plt.ylabel("Young's Modulus", fontsize=24)
#     plt.title("Sampled Materials in Property Space (All Feature Modes)", fontsize=24)
#     plt.legend(fontsize=16)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import torch

    # Define the feature modes for the two subplots
    feature_modes = [
        "all_including_criticality",  # Top subplot
        "density_youngs"             # Bottom subplot
    ]
    feature_mode_styles = {
        "density_youngs":           {"color": "gray",      "marker": "*", "label": "E & density"},
        "all_including_criticality":{"color": "tab:red",   "marker": "D", "label": "All attributes"},
    }
    thresholds = [2.55]

    def unlognorm(x, scaleMax, scaleMin):
        return 10.**(x * (scaleMax - scaleMin) + scaleMin)

    def sample_points_in_ellipse(center, a, b, theta, n_points=1000):
        r = np.sqrt(np.random.uniform(0, 1, n_points))
        angles = np.random.uniform(0, 2 * np.pi, n_points)
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        xs = a * xs
        ys = b * ys
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
        points = np.dot(np.stack([xs, ys], axis=1), rot.T)
        points += np.array(center)
        return points

    all_sampled = []
    all_sampled_labels = []
    all_sampled_markers = []
    all_sampled_colors = []

    for feature_mode in feature_modes:
        for threshold in thresholds:
            trainingData, dataInfo, dataIdentifier, trainInfo, EMax = preprocessData(
                criticality_threshold=threshold, feature_mode=feature_mode)
            latentDim, hiddenDim = 2, 250
            numEpochs = 40000
            klFactor = 5e-5
            learningRate = 2e-3
            savedNet = './data/vaeNet_ref.nt'
            vaeSettings = {'encoder': {'inputDim': trainingData.shape[1], 'hiddenDim': hiddenDim, 'latentDim': latentDim},
                           'decoder': {'latentDim': latentDim, 'hiddenDim': hiddenDim, 'outputDim': trainingData.shape[1]}}
            materialEncoder = MaterialEncoder(trainingData, dataInfo, dataIdentifier, vaeSettings)
            materialEncoder.constraints = {}
            materialEncoder.trainAutoencoder(numEpochs, klFactor, savedNet, learningRate)

            zReal = materialEncoder.vaeNet.encoder.z.detach().numpy()
            enclosing_ellipse = welzl(np.array(zReal, dtype=float))
            center, a, b, t = enclosing_ellipse
            sampled_points = sample_points_in_ellipse(center, a, b, t, n_points=500)
            points_tensor = torch.tensor(sampled_points, dtype=torch.float32)
            decoded = materialEncoder.vaeNet.decoder(points_tensor)
            density_sampled = unlognorm(
                decoded[:, dataInfo['MassDensity']['idx']],
                dataInfo['MassDensity']['scaleMax'],
                dataInfo['MassDensity']['scaleMin']
            ).detach().numpy()
            youngs_sampled = unlognorm(
                decoded[:, dataInfo['ElasticModulus']['idx']],
                dataInfo['ElasticModulus']['scaleMax'],
                dataInfo['ElasticModulus']['scaleMin']
            ).detach().numpy()
            all_sampled.append((density_sampled, youngs_sampled))
            all_sampled_labels.append(feature_mode_styles[feature_mode]["label"])
            all_sampled_markers.append(feature_mode_styles[feature_mode]["marker"])
            all_sampled_colors.append(feature_mode_styles[feature_mode]["color"])

    # Plot two subplots: All attributes (top) and E & density (bottom)
    fig, axs = plt.subplots(2, 1, figsize=(12, 16), sharex=True, sharey=True)
    df = pd.read_excel('./data/TeledyneDatabase2.xlsx')
    density_real = df.iloc[:, 5].to_numpy()
    youngs_real = df.iloc[:, 10].to_numpy()

    for i, (density_sampled, youngs_sampled) in enumerate(all_sampled):
        ax = axs[i]
        ax.scatter(density_real, youngs_real, c='k', s=120, label='Real Materials', alpha=0.7, edgecolors='w')
        ax.scatter(
            density_sampled, youngs_sampled,
            c=all_sampled_colors[i], marker=all_sampled_markers[i], s=60, alpha=0.7, label=all_sampled_labels[i]
        )
        ax.grid(True)
        ax.set_xlabel('Density', fontsize=16)
        ax.set_ylabel("Young's Modulus", fontsize=16)
        ax.legend(fontsize=12)

    plt.tight_layout()
    plt.show()