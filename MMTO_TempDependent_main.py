import numpy as np
import torch
import time
from MMTO_TempDependent_examples import MMTOTempDependentExamples, getMMTOTempDependentProblem
from materialEncoder import MaterialEncoder
import matplotlib.pyplot as plt
from MMTO_TempDependent_obj_cons_sensitivities import (
    compute_mmto_objective_and_gradient,
    compute_mmto_constraint_and_gradient,
)
from PyTOImports import *
from InterpolationFunctions import bezierInterpolation
from enum import Enum
class Z0InitMethod(Enum):
    LIGHTEST = 'lightest'
    HEAVIEST = 'heaviest'
    ORIGIN = 'origin'
    UNIFORM = 'uniform'


# The main code for MMTO topology optimization
def run_topopt(
    to_problem,
    nIterationsWithoutPenalization=50,
    nIterationsWithPenalization=50,
    turnOnThermal=True,
    turnOnNonlinearThermal=False,
    timeLimit=7200,
    saveNet=None,
    use_pretrained_vae=True,
    z0_init_method=Z0InitMethod.ORIGIN,
    rel_conv_tol=1e-10,
    gamma_init=1e-3,
    gamma_max=0.1,
    gamma_factor=1.5):

    mesh_structural, mesh_thermal, mat_prop_struct, mat_prop_thermal, \
    bc_struct, bc_thermal, elem_body_force, to_params, vae_params = \
        getMMTOTempDependentProblem(to_problem)

    if to_params.MaterialsExcelFile is None:
        print("Please provide a valid MaterialsExcelFile in to_params.")
        return
    matEncoder = MaterialEncoder(vae_params)
    matEncoder.readExcel(to_params.MaterialsExcelFile)

    numAttributes = matEncoder.nAttributes

    if saveNet is None:
        base, _ = os.path.splitext(to_params.MaterialsExcelFile)
        saveNet = base + ".nt"
    vae_file_exists = os.path.exists(saveNet) and os.path.getsize(saveNet) > 0

    if use_pretrained_vae and vae_file_exists:
        print(f"Loading pre-trained autoencoder from file: {saveNet}")
        matEncoder.loadAutoencoderFromFile(saveNet)
    else:
        print(f"Training autoencoder from scratch and saving to: {saveNet}")
        matEncoder.trainAutoencoder(vae_params.numEpochs, vae_params.klFactor, saveNet, vae_params.learningRate)
        matEncoder.printEncodingErrors()

    with torch.no_grad():
        matEncoder.training_latents = matEncoder.vaeNet.encoder(matEncoder.scaledMaterialData).cpu()

    zRealTorch = matEncoder.training_latents

    if True:
        matEncoder.plotTemperatureVsMaterialProperty("E", semilogy=True)
        matEncoder.plotTemperatureVsMaterialProperty("Y", semilogy=True)
    if False:
        matEncoder.plotLSRContours("E0")
        matEncoder.plotLSRContours("E1")

    solver = linear_solvers.Solvers.PARDISO
    dsolver = deflation.DeflationSolver()
    fe_solver_structural = hex_structural_fea.HexStructuralFEA(
        mesh=mesh_structural,
        mat_prop=mat_prop_struct,
        bc=bc_struct,
        solver=solver,
        dsolver=dsolver,
        rtol=1e-4,
        elem_body_force=elem_body_force)

    fe_solver_thermal = hex_thermal_fea.HexThermalFEA(
        mesh=mesh_thermal,
        mat_prop=mat_prop_thermal,
        bc=bc_thermal,
        solver=solver,
        rtol=1e-8)

    KETemplate = hex_element_stiffness.hex8_stiffness_matrix_structural(
        mat_prop_struct.youngs_modulus, mat_prop_struct.poissons_ratio, mesh_structural.elem_size)

    num_elems = mesh_structural.num_elems
    num_design_var = num_elems + num_elems * 2

    print("Creating filter...")
    [H, Hs] = createFilters(fe_solver_structural, to_params)

    iterationCount = 0
    obj0 = None
    gamma = gamma_init
    if turnOnThermal and turnOnNonlinearThermal:
        print("Performing nonlinear thermal analysis...")
    elif turnOnThermal and not turnOnNonlinearThermal:
        print("Performing linear thermal analysis...")
    else:
        print("Thermal analysis is turned off...")
    last_Temp_nodes = None # Store last converged temperature field
    Temp_elem = None
    def MMTO_optimization_function(zeta):
        nonlocal iterationCount, obj0, gamma, zRealTorch, last_Temp_nodes, Temp_elem
        zeta = np.asarray(zeta).flatten()
        print("-------------- Iteration", iterationCount, "-----------------")
        zetaTensor = torch.tensor(zeta, dtype=torch.float32, requires_grad=True)
        xDesign = zetaTensor[0:num_elems]
        zD = zetaTensor[num_elems:]
        zPts = zD.view(2, -1).T

        decoded = matEncoder.vaeNet.decoder(zPts)
        material_properties = matEncoder.getMaterialProperties(decoded)

        if turnOnThermal:
            if turnOnNonlinearThermal:
                K0 = material_properties['K0'].detach().numpy()
                K1 = material_properties['K1'].detach().numpy()
                K2 = material_properties['K2'].detach().numpy()
                K3 = material_properties['K3'].detach().numpy()
                picard_tol = 1e-6
                max_picard_iter = 50
                init_K = K0

                # Use previous temperature field as initial guess
                if last_Temp_nodes is None:
                    # First iteration: use average conductivity
                    fe_solver_thermal.mat_prop = [
                        mat_lib.create_material_with_defaults(
                            name=f"K{i+1}",
                            thermal_conductivity=init_K[i].item())
                        for i in range(num_elems)]
                    fe_solver_thermal.set_thermal_material(fe_solver_thermal.mat_prop)
                    Temp_nodes = fe_solver_thermal.solve(xDesign.detach().cpu().numpy())
                else:
                    Temp_nodes = last_Temp_nodes.copy()

                for picard_iter in range(max_picard_iter):
                    edofMat = fe_solver_thermal.mesh.edofMat
                    Temp_elem = np.mean(Temp_nodes[edofMat], axis=1)
                    K_elem = bezierInterpolation(Temp_elem, K0, K1, K2, K3)
                    fe_solver_thermal.mat_prop = [
                        mat_lib.create_material_with_defaults(
                            name=f"K{i+1}",
                            thermal_conductivity=K_elem[i].item())
                        for i in range(num_elems)]
                    fe_solver_thermal.set_thermal_material(fe_solver_thermal.mat_prop)
                    Temp_nodes_new = fe_solver_thermal.solve(xDesign.detach().cpu().numpy())
                    norm_diff = np.linalg.norm(Temp_nodes_new - Temp_nodes)
                    if norm_diff < picard_tol:
                        Temp_nodes = Temp_nodes_new
                        #print(f"Converged in {picard_iter+1} iterations with norm: {norm_diff:.6e}")
                        break
                    Temp_nodes = Temp_nodes_new
                else:
                    print(f"Max Picard iterations reached with norm: {norm_diff:.6e}")

                last_Temp_nodes = Temp_nodes.copy()  # Save for next optimization iteration
                edofMat = fe_solver_thermal.mesh.edofMat
                Temp_elem = np.mean(Temp_nodes[edofMat], axis=1)
            else:
                thermalConductivity = material_properties['K0']
                thermalConductivity_elem = thermalConductivity.detach().numpy()
                fe_solver_thermal.mat_prop = [
                    mat_lib.create_material_with_defaults(
                        name=f"K{i+1}",
                        thermal_conductivity=thermalConductivity_elem[i].item())
                    for i in range(num_elems)]
                fe_solver_thermal.set_thermal_material(fe_solver_thermal.mat_prop)
                Temp_nodes = fe_solver_thermal.solve(xDesign.detach().cpu().numpy())
                edofMat = fe_solver_thermal.mesh.edofMat
                Temp_elem = np.mean(Temp_nodes[edofMat], axis=1)
        else:
            Temp_elem = np.ones(num_elems) * 50.0

        E = matEncoder.getMaterialPropertyAtTemperature("E", zPts, Temp_elem)
        Y = matEncoder.getMaterialPropertyAtTemperature("Y", zPts, Temp_elem)

        fe_solver_structural.mat_prop = [
            mat_lib.create_material_with_defaults(
                name=f"Material_{i+1}",
                youngs_modulus=E[i].item(),
                yield_strength=Y[i].item(),
                poissons_ratio=0.3
            )
            for i in range(num_elems)
        ]
        fe_solver_structural.set_structural_material(fe_solver_structural.mat_prop)

        uvw = fe_solver_structural.solve(xDesign.detach().cpu().numpy(), MaterialModel.SIMP)
        fe_solver_structural.postprocess()

        obj, grad_obj = compute_mmto_objective_and_gradient(
            to_params, uvw, Temp_elem, zeta, fe_solver_structural, KETemplate, matEncoder)

        cons, grad_cons = compute_mmto_constraint_and_gradient(
            to_params, uvw, Temp_elem, zeta, fe_solver_structural, KETemplate, matEncoder)

        if obj0 is None:
            obj0 = obj
            if np.any(cons > 0):
                print(50 * "-")
                print("Warning: Constraint(s) violated at start of optimization!")
                print("GCMMA may not converge for this problem. Consider changing constraints if convergence issues occur.")
                print(50 * "-")

        obj = obj / obj0
        grad_obj = grad_obj / obj0

        grad_obj[0:num_elems] = (H * grad_obj[0:num_elems]) / Hs
        grad_obj[num_elems:2*num_elems] = (H * grad_obj[num_elems:2*num_elems]) / Hs
        grad_obj[2*num_elems:3*num_elems] = (H * grad_obj[2*num_elems:3*num_elems]) / Hs

        for i in range(grad_cons.shape[0]):
            grad_cons[i, 0:num_elems] = (H * grad_cons[i, 0:num_elems]) / Hs
            grad_cons[i, num_elems:2*num_elems] = (H * grad_cons[i, num_elems:2*num_elems]) / Hs
            grad_cons[i, 2*num_elems:3*num_elems] = (H * grad_cons[i, 2*num_elems:3*num_elems]) / Hs

        grad_obj = np.array([grad_obj]).reshape((num_design_var, 1))
        cons = np.array(cons).reshape((-1, 1))
        grad_cons = np.array(grad_cons).reshape((len(cons), num_design_var))

        objective_name = getattr(to_params.Objective[0], 'name', str(to_params.Objective[0]))
        constraint_names = [getattr(c[0], 'name', str(c[0])) for c in to_params.Constraints]

        print(f"Min. Objective ({objective_name}): {obj*obj0:.3g}")
        for idx, val in enumerate(cons.flatten()):
            print(f"Constraint {idx+1} ({constraint_names[idx]}): {(val+1)*to_params.Constraints[idx][2]:.3g} <= {to_params.Constraints[idx][2]:.3g}?")

        if iterationCount > nIterationsWithoutPenalization:
            d_ij = torch.cdist(zPts, zRealTorch, p=2) + 1e-12
            min_i = torch.min(d_ij, dim=1).values
            min_i = min_i * xDesign
            penalty = gamma * torch.mean(min_i) 
            zetaTensor.grad = None
            penalty.backward(retain_graph=True)
            grad_obj[num_elems:, 0] += zetaTensor.grad[num_elems:].detach().numpy()
            obj = obj + penalty.item()
            if False:
                grad_obj[num_elems:2*num_elems, 0] = (H * grad_obj[num_elems:2*num_elems, 0]) / Hs
                grad_obj[2*num_elems:3*num_elems, 0] = (H * grad_obj[2*num_elems:3*num_elems, 0]) / Hs
            gamma = min(gamma * gamma_factor, gamma_max)

        iterationCount += 1
        return obj, grad_obj, cons, grad_cons

    x0 = 0.5 * np.ones(num_elems)
    x0 = (H * x0) / Hs

    z0 = np.zeros(2 * num_elems)
    if z0_init_method == Z0InitMethod.LIGHTEST:
        zLightest = matEncoder.getLightestMaterial()
        z0[0:num_elems] = zLightest[0]
        z0[num_elems:2*num_elems] = zLightest[1]
    elif z0_init_method == Z0InitMethod.HEAVIEST:
        zHeaviest = matEncoder.getHeaviestMaterial()
        z0[0:num_elems] = zHeaviest[0]
        z0[num_elems:2*num_elems] = zHeaviest[1]
    elif z0_init_method == Z0InitMethod.ORIGIN:
        z0[0:num_elems] = 0.0
        z0[num_elems:2*num_elems] = 0.0
    elif z0_init_method == Z0InitMethod.UNIFORM:
        z0[0:num_elems] = np.random.uniform(-0.5, 0.5, size=num_elems)
        z0[num_elems:2*num_elems] = np.random.uniform(-0.5, 0.5, size=num_elems)
    else:
        raise ValueError(f"Unknown z0_init_method: {z0_init_method}")

    z0[0:num_elems] = (H * z0[0:num_elems]) / Hs
    z0[num_elems:2*num_elems] = (H * z0[num_elems:2*num_elems]) / Hs
    zeta0 = np.concatenate((x0, z0), axis=0).reshape(-1, 1)

    lowerBound = np.zeros(num_design_var, dtype=float).reshape(-1, 1)
    upperBound = np.ones(num_design_var, dtype=float).reshape(-1, 1)
    lowerBound[num_elems:3*num_elems] = np.min(zRealTorch.cpu().numpy())
    upperBound[num_elems:3*num_elems] = np.max(zRealTorch.cpu().numpy())

    nVariables = num_design_var
    nConstraints = len(to_params.Constraints)
    tStart = time.time()
    maxMMAIterations = nIterationsWithoutPenalization + nIterationsWithPenalization

    optResults = runMMA(nVariables, nConstraints, MMTO_optimization_function, zeta0.reshape(-1, 1), lowerBound,
        upperBound, maxIterations=maxMMAIterations, timeLimitSecs=timeLimit,
        move_limit=0.2, kktTol=1e-6, fTolerance=rel_conv_tol, gTolerance=rel_conv_tol, verbose=False)
    zetaOptimal = optResults[0]
    tEnd = time.time()
    print(f"Total optimization time: {tEnd - tStart:.2f} seconds")

    zetaOptimal = np.asarray(zetaOptimal).flatten()
    xDesign = zetaOptimal[0:num_elems]
    zDesign = zetaOptimal[num_elems:]
    zDesignTensor = torch.tensor(zDesign).float()
    zPts = zDesignTensor.view(2, -1).T
    decoded = matEncoder.vaeNet.decoder(zPts)
    material_properties = matEncoder.getMaterialProperties(decoded)
    closest_index = matEncoder.getClosestRealMaterialIndex(zPts)

    E = matEncoder.getMaterialPropertyAtTemperature("E", zPts, Temp_elem)
    Y = matEncoder.getMaterialPropertyAtTemperature("Y", zPts, Temp_elem)
    T_Limit = matEncoder.getValuesAtLatentPoints("T_Limit", zPts)

    isTemperatureWithinLimits = (Temp_elem <= T_Limit.flatten()) | (xDesign < 0.5)
    print(f"Number of elements exceeding Temp Limit: {np.sum(~isTemperatureWithinLimits)} out of {num_elems}")

    fe_solver_structural.mesh.setPseudoDensity(xDesign)
    fe_solver_structural.solve(xDesign)
    fe_solver_structural.postprocess()

    # Plot closest_index
    fe_solver_structural.plot_elem_field(closest_index)
   
    # Plot Temperature
    fe_solver_structural.plot_elem_field(Temp_elem, title='Temperature', colormap='plasma')
   

    # Plot Is Within Limits
    fe_solver_structural.plot_elem_field(isTemperatureWithinLimits, title='Is Within Limits', colormap='RdYlGn')
    
    # Plot Young's Modulus
    fe_solver_structural.plot_elem_field(E, title='Young\'s Modulus', colormap='plasma')

    matEncoder.plotLSR(zRealTorch.detach().cpu().numpy(), zDesign.reshape(2, -1).T, xDesign=xDesign)

if __name__ == "__main__":
    
    # Temperature Dependent TO Problems (see MMTO_TempDependent_examples.py for details):
    
    # 1. LBracket_Compliance_Mass (LBracket design, Minimize Compliance with Mass constraints)
    # 2. LBracket_Compliance_MassCost (LBracket design, Minimize Compliance with Mass and Cost constraints)
    # 3. LBracket_Compliance_MassCriticality (LBracket design, Minimize Compliance with Mass and Criticality constraints)
    # 4. LBracket_Pnormstress_ComplianceMass (LBracket design, Minimize P-norm Stress with Compliance and Mass constraints)
    # 5. LBracket_Mass_ComplianceSafetyFactor (LBracket design, Minimize Mass with Compliance and Safety Factor constraints)

    to_problem = MMTOTempDependentExamples.LBracket_Compliance_Mass

    run_topopt(
        to_problem=to_problem,
        turnOnNonlinearThermal=False,
        use_pretrained_vae=True,
    )