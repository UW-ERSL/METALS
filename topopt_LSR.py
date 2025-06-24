
from LSRImports import *

_LARGE_NUMBER = 1.e9



def topopt_mma_lsr(fe_solver,
			   			to_params,
							 vae_info: None,
			   			minMMAIterations: int = 5,
			   			 maxMMAIterations: int = 50, 
							timeLimit: float =3600, #1 hour
						   penal: float = 3.0,
							 move_limit: float = 0.2,
							 kkt_tol: float = 1.e-6,
							 move_tol: float = 0.025,
							 continuationScheme: bool = False,	 
							 rel_conv_tol: float = 1.e-4,
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
	# mma_state = mma.init_mma(to_params.DesiredVolFraction * np.ones((num_design_var, 1)), mma_params)
	mma_state = mma.init_mma(0.5 * np.ones((num_design_var, 1)), mma_params)
	# KE = elem_stiff.hex8_stiffness_matrix_structural( fe_solver.mat_prop,fe_solver.mesh.elem_size)
	
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
	
	constraintType = to_params.Constraints[0][0] # assume this is the first constraint
	if (constraintType == TO_QOI.VOLUME_FRACTION):
		volFractionConstraint = to_params.Constraints[0][2]
	else:
		volFractionConstraint =1 # default value

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
		x = vae_info.unnormalize_last_n(arr=x, n = 2*num_elems)
		xTensor = torch.tensor(x).float()
		xTensor.requires_grad = True
		xDesign = x[0:num_elems]
		zD = xTensor[num_elems:]
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
		obj, grad_obj = compute_objective_and_gradient(to_params,sol,xDesign, fe_solver,KE, material_model)
		
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
		xConstraint_tensor = torch.tensor(x).float()
		xConstraint_tensor.requires_grad = True
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

		change = np.max(np.abs(x - x_old))
		x_old = x
		print(f"it.: {mma_state.epoch}, obj.: {obj[0]:.6g} vf: {vf:.3f}",
					f"ch: {change:.3f}")
		history['compliance'].append(obj[0])
		history['volume'].append(np.mean(x))
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
		
	fe_solver.mesh.setPseudoDensity(x[0:num_elems])
	print(f"Time FEA: {timeFEA:.2f} s, Time MMA: {timeMMA:.2f} s")
	print(f"Total Time: {timeFEA+timeMMA:.2f} s")
	EDesign[xDesign < 0.001] = 1e-3
	plt.hist(xDesign, bins = 10)

	return np.asarray(EDesign), history,success

def preprocessData():
	df = pd.read_excel('./data/TeledyneDatabase2.xlsx')# solidworksMaterialDatabaseCost # aluminum
	dataIdentifier = {'name': df[df.columns[0]], 'className':df[df.columns[1]], 'classID':df[df.columns[2]]} # name of the material and type
	
	rawData = df[df.columns[3:]].to_numpy()
	YoungsModulus = rawData[:,7]
	EMax = np.max(YoungsModulus) # GPa
	print("Max E: ", EMax, " GPa")
	trainInfo = np.log10(df[df.columns[3:]].to_numpy())
	dataScaleMax = torch.tensor(np.max(trainInfo, axis = 0))
	dataScaleMin = torch.tensor(np.min(trainInfo, axis = 0))
	normalizedData = (torch.tensor(trainInfo) - dataScaleMin)/(dataScaleMax - dataScaleMin)
	trainingData = normalizedData.clone().float()

	dataInfo = {'UltimateStrength':{'idx':0,'scaleMin':dataScaleMin[0], 'scaleMax':dataScaleMax[0]},\
			'YieldStress':	{'idx':1,'scaleMin':dataScaleMin[1], 'scaleMax':dataScaleMax[1]},\
			'MassDensity':	{'idx':2,'scaleMin':dataScaleMin[2], 'scaleMax':dataScaleMax[2]},\
			'CostPerPound':{'idx':3,'scaleMin':dataScaleMin[3], 'scaleMax':dataScaleMax[3]},\
			'MeltingTempC':	{'idx':4,'scaleMin':dataScaleMin[4], 'scaleMax':dataScaleMax[4]},\
			'MaxUseTempC':	{'idx':5,'scaleMin':dataScaleMin[5], 'scaleMax':dataScaleMax[5]},\
			'Elong2Fail':{'idx':6,'scaleMin':dataScaleMin[6], 'scaleMax':dataScaleMax[6]},\
			'ElasticModulus':{'idx':7,'scaleMin':dataScaleMin[7], 'scaleMax':dataScaleMax[7]},\
			'CriticalityIdx':{'idx':8,'scaleMin':dataScaleMin[8], 'scaleMax':dataScaleMax[8]}}

	return trainingData, dataInfo, dataIdentifier, trainInfo, EMax



if __name__ == "__main__":    
	from topopt_structural_benchmarks import *
	from LSRImports import *
	# import struct_fea as fea
	# import linear_solvers as lin_solv
	import time
	import matplotlib.pyplot as plt
	# import deflation
	# import os
	import pandas as pd
	import sys 
	import os
	script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
	rel_path = "./data/vaeNet_ref.nt"
	abs_file_path = os.path.join(script_dir, rel_path)
	# jax.config.update("jax_enable_x64", True)
	# optimizationMethod = TO_METHODS.DENSITYMMA_LSR # DENSITYMMA, DENSITYOC, PARETO, LEVELSET

	#runTOTests(); exit(0) # Run all tests for each example in the StructuralTOExamples enum
	
	trainingData, dataInfo, dataIdentifier, trainInfo,EMax = preprocessData()
	numMaterialsInTrainingData, numFeatures = trainingData.shape

	latentDim, hiddenDim = 2, 250
	numEpochs = 40000
	klFactor = 5e-5
	learningRate = 2e-3
	savedNet = './data/vaeNet_ref.nt'
	vaeSettings = {'encoder':{'inputDim':numFeatures, 'hiddenDim':hiddenDim, 'latentDim':latentDim},\
				'decoder':{'latentDim':latentDim, 'hiddenDim':hiddenDim, 'outputDim':numFeatures}}

	materialEncoder = MaterialEncoder(trainingData, dataInfo, dataIdentifier, vaeSettings)
	# start = time.perf_counter()
	print(sys.path)
	materialEncoder.loadAutoencoderFromFile(abs_file_path);




	# Choose the TO problem
	to_problem = StructuralTOExamples.BliskWithBladeMass
	solver = lin_solv.Solvers.PARDISO # Typically PARDISO, but DPCG for DOF > 200,000
	debug = False 

	# Get the structural problem
	mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)

	elem_body_force = None
	dsolver = deflation.DeflationSolver()
	# initialize the fe solver 
	if (solver == lin_solv.Solvers.DPCG):
		nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
		dsolver.create_deflation_groups(mesh, nGroups)
		dsolver.create_delfation_matrix(mesh)
		dsolver.W = dsolver.W[bc.free_dofs, :]

	fe_solver = hex_structural_fea.HexStructuralFEA(mesh = mesh,
				mat_prop = mat_prop,
				bc = bc,
				solver = solver,
				dsolver = dsolver,
				rtol = 1e-8,
				elem_body_force = elem_body_force)
	

	print('Solver: ', fe_solver.solver.name)
	print("nDof: ", 3*fe_solver.mesh.num_nodes)
	print("nElem: ", fe_solver.mesh.num_elems)	
	
	title = f'nDOF: {3*fe_solver.mesh.num_nodes}, nElem: {fe_solver.mesh.num_elems}'
	#plots.plotMesh(mesh, bc,title = title)


	startTime = time.time()
	print("OptimizationMethod: MMA_LSR")
	u, history,success = topopt_mma_lsr(fe_solver = fe_solver,
																		vae_info = materialEncoder,
								to_params = to_params,
								debug = debug)
	timeTaken = time.time() - startTime
	fig, ax1 = plt.subplots()

	# Plot compliance on left y-axis
	ax1.set_xlabel('Iterations')
	ax1.set_ylabel('Compliance', color='tab:blue')
	ax1.plot(history['compliance'], color='tab:blue', label='Compliance')
	ax1.tick_params(axis='y', labelcolor='tab:blue')

	# Plot volume fraction on right y-axis with dotted line
	ax2 = ax1.twinx()
	ax2.set_ylabel('Volume Fraction', color='tab:orange')
	ax2.plot(history['volume'], color='tab:orange', linestyle=':', label='Volume Fraction')
	ax2.tick_params(axis='y', labelcolor='tab:orange')
	ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

	plt.title('MMA: Volume and Compliance vs. Iterations')

	# Add legend
	lines1, labels1 = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax1.legend(lines1 + lines2, labels1 + labels2)

	plt.grid(True)
	plt.show(block=True)

	title = f"MMA: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"

	print(f"Time taken: {timeTaken:.0f} s")
	
	plots.plotMesh(fe_solver.mesh, bc = None, u=None, title = title)
	plots.plotElementFieldAndDensity(fe_solver.mesh, u,
                        title='YoungModulus', cmap='viridis') #**