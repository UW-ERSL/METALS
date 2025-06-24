
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




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from matplotlib.patches import Ellipse
    import torch

    feature_modes = [
        "density_youngs",
        "density_youngs_yield",
        "all_but_criticality",
        "all_including_criticality"
    ]
    feature_mode_titles = {
        "all_including_criticality": "Using all attributes",
        "all_but_criticality": "Using all attributes except the criticality index",
        "density_youngs_yield": "Using only elastic modulus, density, and yield stress",
        "density_youngs": "Using only elastic modulus and density"
    }
    # Shorter names for summary table
    feature_mode_titles_short = {
        "all_including_criticality": "All attributes",
        "all_but_criticality": "All except criticality",
        "density_youngs_yield": "E, density, yield",
        "density_youngs": "E & density"
    }
    thresholds = [2.55]
    ellipse_params_all = []
    zReal_list_all = []
    sampled_points_list_all = []
    error_table_rows = []

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

    for feature_mode in feature_modes:
        ellipse_params = []
        zReal_list = []
        sampled_points_list = []
        print(f"\n=== Processing feature_mode: {feature_mode} ===")
        for threshold in thresholds:
            trainingData, dataInfo, dataIdentifier, trainInfo, EMax = preprocessData(
                criticality_threshold=threshold, feature_mode=feature_mode)
            numMaterialsInTrainingData, numFeatures = trainingData.shape

            latentDim, hiddenDim = 2, 250
            numEpochs = 40000
            klFactor = 5e-5
            learningRate = 2e-3
            savedNet = './data/vaeNet_ref.nt'
            vaeSettings = {'encoder': {'inputDim': numFeatures, 'hiddenDim': hiddenDim, 'latentDim': latentDim},
                           'decoder': {'latentDim': latentDim, 'hiddenDim': hiddenDim, 'outputDim': numFeatures}}

            materialEncoder = MaterialEncoder(trainingData, dataInfo, dataIdentifier, vaeSettings)
            constraints = {}
            materialEncoder.constraints = constraints
            convgHistory = materialEncoder.trainAutoencoder(numEpochs, klFactor, savedNet, learningRate)

            predData = materialEncoder.vaeNet(trainingData)
            zReal = materialEncoder.vaeNet.encoder.z.detach().numpy()
            zReal_list.append(zReal)
            enclosing_ellipse = welzl(np.array(zReal, dtype=float))
            ellipse_params.append(enclosing_ellipse)

            center, a, b, t = enclosing_ellipse
            sampled_points = sample_points_in_ellipse(center, a, b, t, n_points=500)
            sampled_points_list.append(sampled_points)

            points_tensor = torch.tensor(sampled_points, dtype=torch.float32)
            decoded = materialEncoder.vaeNet.decoder(points_tensor)

            # Get density and Young's modulus for sampled points
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

            # --- Get real materials from Teledyne database ---
            df = pd.read_excel('./data/TeledyneDatabase2.xlsx')
            if 'Criticality Index' in df.columns and threshold is not None:
                df = df[df['Criticality Index'] < threshold]
            density_real = df.iloc[:, 5].to_numpy()
            youngs_real = df.iloc[:, 10].to_numpy()

            # --- Plot real materials and sampled points in property space ---
            plt.figure(figsize=(10, 8))
            plt.scatter(density_real, youngs_real, c='k', s=120, label='Real Materials')
            plt.scatter(density_sampled, youngs_sampled, c='r', s=60, marker='*', label='Sampled Materials', alpha=0.5)
            plt.xlabel('Density', fontsize=24)
            plt.ylabel("Young's Modulus", fontsize=24)
            plt.title(feature_mode_titles[feature_mode], fontsize=24)
            plt.legend(fontsize=20)
            plt.grid(True)
            plt.tight_layout()
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.show()

            # --- Compare original and reconstructed properties for first 20 materials ---
            num_compare = 20
            df_compare = df.iloc[:num_compare]

            if feature_mode == "density_youngs":
                density_orig = df_compare.iloc[:, 5].to_numpy()
                youngs_orig = df_compare.iloc[:, 10].to_numpy()
                rawData_compare = df_compare.iloc[:, [5, 10]].to_numpy()
                trainInfo_compare = np.log10(rawData_compare)
                dataScaleMax = [dataInfo['MassDensity']['scaleMax'], dataInfo['ElasticModulus']['scaleMax']]
                dataScaleMin = [dataInfo['MassDensity']['scaleMin'], dataInfo['ElasticModulus']['scaleMin']]
            elif feature_mode == "density_youngs_yield":
                density_orig = df_compare.iloc[:, 5].to_numpy()
                youngs_orig = df_compare.iloc[:, 10].to_numpy()
                rawData_compare = df_compare.iloc[:, [5, 4, 10]].to_numpy()
                trainInfo_compare = np.log10(rawData_compare)
                dataScaleMax = [dataInfo['MassDensity']['scaleMax'], dataInfo['YieldStress']['scaleMax'], dataInfo['ElasticModulus']['scaleMax']]
                dataScaleMin = [dataInfo['MassDensity']['scaleMin'], dataInfo['YieldStress']['scaleMin'], dataInfo['ElasticModulus']['scaleMin']]
            elif feature_mode == "all_but_criticality":
                density_orig = df_compare.iloc[:, 5].to_numpy()
                youngs_orig = df_compare.iloc[:, 10].to_numpy()
                rawData_compare = df_compare.iloc[:, 3:-1].to_numpy()
                trainInfo_compare = np.log10(rawData_compare)
                dataScaleMax = np.max(trainInfo_compare, axis=0)
                dataScaleMin = np.min(trainInfo_compare, axis=0)
            elif feature_mode == "all_including_criticality":
                density_orig = df_compare.iloc[:, 5].to_numpy()
                youngs_orig = df_compare.iloc[:, 10].to_numpy()
                rawData_compare = df_compare.iloc[:, 3:].to_numpy()
                trainInfo_compare = np.log10(rawData_compare)
                dataScaleMax = np.max(trainInfo_compare, axis=0)
                dataScaleMin = np.min(trainInfo_compare, axis=0)
            else:
                raise ValueError("Unknown feature_mode")

            normalized_compare = (torch.tensor(trainInfo_compare) - torch.tensor(dataScaleMin)) / (torch.tensor(dataScaleMax) - torch.tensor(dataScaleMin))
            normalized_compare = normalized_compare.float()
            with torch.no_grad():
                z_compare = materialEncoder.vaeNet.encoder(normalized_compare)
                if isinstance(z_compare, tuple):
                    z_compare = z_compare[0]
                decoded_compare = materialEncoder.vaeNet.decoder(z_compare)
            density_pred = unlognorm(
                decoded_compare[:, dataInfo['MassDensity']['idx']],
                dataInfo['MassDensity']['scaleMax'],
                dataInfo['MassDensity']['scaleMin']
            ).detach().numpy()
            youngs_pred = unlognorm(
                decoded_compare[:, dataInfo['ElasticModulus']['idx']],
                dataInfo['ElasticModulus']['scaleMax'],
                dataInfo['ElasticModulus']['scaleMin']
            ).detach().numpy()
            x = np.arange(num_compare)
            width = 0.35
            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            plt.bar(x - width/2, density_orig, width, label='Original')
            plt.bar(x + width/2, density_pred, width, label='Predicted')
            plt.xlabel('Material Index', fontsize=24)
            plt.ylabel('Density', fontsize=24)
            plt.title('Density: Original vs Predicted', fontsize=24)
            plt.legend(fontsize=20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            plt.subplot(1, 2, 2)
            plt.bar(x - width/2, youngs_orig, width, label='Original')
            plt.bar(x + width/2, youngs_pred, width, label='Predicted')
            plt.xlabel('Material Index', fontsize=24)
            plt.ylabel("Young's Modulus", fontsize=24)
            plt.title("Young's Modulus: Original vs Predicted", fontsize=24)
            plt.legend(fontsize=20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            plt.tight_layout()
            plt.show()

            # Calculate error percentages
            youngs_error = 100 * np.abs(youngs_pred - youngs_orig) / np.abs(youngs_orig)
            density_error = 100 * np.abs(density_pred - density_orig) / np.abs(density_orig)

            # Get material names (assuming first column is the name)
            material_names = df_compare.iloc[:, 0].to_numpy()

            # Collect error data for the table
            for idx in range(num_compare):
                error_table_rows.append({
                    "Material": material_names[idx],
                    "Density Error (%)": f"{density_error[idx]:.2f}",
                    "Young's Modulus Error (%)": f"{youngs_error[idx]:.2f}",
                    "Feature Type": feature_mode_titles[feature_mode]
                })

        # --- Separate Latent Space Ellipse Plot ---
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
        for i, (zReal, (center, a, b, t), sampled_points) in enumerate(zip(zReal_list, ellipse_params, sampled_points_list)):
            # Real materials: much larger points, sampled: smaller, ellipse: thick
            ax.scatter(zReal[:, 0], zReal[:, 1], c='k', s=300, marker='o', label='Real Materials')
            ax.scatter(sampled_points[:, 0], sampled_points[:, 1], c='r', s=40, marker='*', label='Sampled Materials', alpha=0.5)
            ellipse = Ellipse(xy=center, width=2 * a, height=2 * b, angle=np.degrees(t),
                              edgecolor='b', fc='none', lw=5, label='LSR Ellipse')
            ax.add_patch(ellipse)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('Latent Dimension 1', fontsize=24)
        ax.set_ylabel('Latent Dimension 2', fontsize=24)
        ax.set_title(feature_mode_titles[feature_mode], fontsize=24)
        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicate legend entries
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.show()

        # Save for possible further use
        ellipse_params_all.append(ellipse_params)
        zReal_list_all.append(zReal_list)
        sampled_points_list_all.append(sampled_points_list)

    # --- Final Table with Only Max Errors for Each Feature Type (Short Names) ---
    summary_rows = []
    for key, short_title in feature_mode_titles_short.items():
        rows = [row for row in error_table_rows if row["Feature Type"] == feature_mode_titles[key]]
        if not rows:
            continue
        density_errors = [float(row["Density Error (%)"]) for row in rows]
        youngs_errors = [float(row["Young's Modulus Error (%)"]) for row in rows]
        idx_density = np.argmax(density_errors)
        row_density = rows[idx_density]
        summary_rows.append({
            "Feature Type": short_title,
            "Error Type": "Max Density Error",
            "Material": row_density["Material"],
            "Error (%)": row_density["Density Error (%)"]
        })
        idx_youngs = np.argmax(youngs_errors)
        row_youngs = rows[idx_youngs]
        summary_rows.append({
            "Feature Type": short_title,
            "Error Type": "Max Young's Modulus Error",
            "Material": row_youngs["Material"],
            "Error (%)": row_youngs["Young's Modulus Error (%)"]
        })

    df_summary = pd.DataFrame(summary_rows, columns=["Feature Type", "Error Type", "Material", "Error (%)"])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    tbl = ax.table(
        cellText=df_summary.values,
        colLabels=df_summary.columns,
        cellLoc='center',
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(20)
    tbl.scale(1.2, 2.0)
    plt.title("Maximum Errors by Material and Feature Type", fontsize=28, pad=20)
    plt.tight_layout()
    plt.show()