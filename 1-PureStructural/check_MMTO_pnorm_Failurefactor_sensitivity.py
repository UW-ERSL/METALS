import numpy as np
import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../0-Common')))

from ExamplesPureStructural import MMTOExamplesPureStructural, getMMTOProblemPureStructural
from materialEncoder import MaterialEncoder # type: ignore[reportMissingImports]
from SensitivitiesPureStructural import (
    compute_mmto_objective_and_gradient,
    compute_mmto_constraint_and_gradient,
)
from PyTOImports import (linear_solvers, hex_structural_fea, hex_element_stiffness, mat_lib, MaterialModel, TO_QOI) # type: ignore

to_problem = MMTOExamplesPureStructural.LBracketTopLoad_Stress_VolumeFraction_Mass

mesh, mat_prop, bc, elem_body_force, to_params, vae_params = getMMTOProblemPureStructural(to_problem)
matEncoder = MaterialEncoder(vae_params)
matEncoder.readExcel(to_params.MaterialsExcelFile)

base, _ = os.path.splitext(to_params.MaterialsExcelFile)
saveNet = base + ".nt"

print(f"Loading pre-trained autoencoder from file: {saveNet}")
matEncoder.loadAutoencoderFromFile(saveNet)

solver = linear_solvers.Solvers.PARDISO

fe_solver_structural = hex_structural_fea.HexStructuralFEA(
    mesh=mesh,
    mat_prop=mat_prop,
    bc=bc,
    solver=solver,
    rtol=1e-8,
    elem_body_force=elem_body_force)

KETemplate = hex_element_stiffness.hex8_stiffness_matrix_structural(1.0, 0.3, mesh.elem_size)

num_elems = mesh.num_elems
latentDim = vae_params.latentDim
num_design_var = num_elems + num_elems * latentDim

zeta = np.zeros(num_design_var).flatten()
zeta[0:num_elems] = 0.5
zeta[num_elems:] = 0

def solve_fea_and_compute_objective(zeta_input, objectiveType):
    xDesign = zeta_input[0:num_elems]
    zDesign = zeta_input[num_elems:]
    zPoints = torch.tensor(zDesign, dtype=torch.float32).view(latentDim, -1).T

    decoded = matEncoder.vaeNet.decoder(zPoints)
    material_properties = matEncoder.getMaterialProperties(decoded)
    Youngs_Modulus = material_properties['Youngs_Modulus'].detach().cpu().numpy()
    mass_density = material_properties['Density'].detach().cpu().numpy()
    if 'Yield_Strength' in material_properties:
        Yield_Strength = material_properties['Yield_Strength'].detach().cpu().numpy()
    else:
        Yield_Strength = None

    fe_solver_structural.mat_prop = [
        mat_lib.create_material_with_defaults(
            name=f"Material_{i+1}",
            youngs_modulus=Youngs_Modulus[i],
            mass_density=mass_density[i],
            yield_strength=Yield_Strength[i] if Yield_Strength is not None else None
        )
        for i in range(len(Youngs_Modulus))]
    fe_solver_structural.set_material(fe_solver_structural.mat_prop)

    sol = fe_solver_structural.solve(xDesign, MaterialModel.SIMP)
    fe_solver_structural.mesh.setPseudoDensity(xDesign)
    fe_solver_structural.postprocess()

    to_params.Objective = (objectiveType, None)
    obj, grad = compute_mmto_objective_and_gradient(
        to_params,
        sol,
        zeta_input,
        fe_solver_structural,
        KETemplate,
        matEncoder
    )
    return obj, grad, sol

def get_failure_factor_constraint(zeta_input, sol):
    # Get the value and gradient for the failure factor constraint
    cvals, cgrads = compute_mmto_constraint_and_gradient(
        to_params, sol, zeta_input, fe_solver_structural, KETemplate, matEncoder
    )
    ff_idx = None
    for i, c in enumerate(to_params.Constraints):
        if c[0] == TO_QOI.STRESS_FAILURE_FACTOR:
            ff_idx = i
            break
    if ff_idx is None:
        raise RuntimeError("STRESS_FAILURE_FACTOR constraint not found in to_params.Constraints")
    obj_ff = cvals[ff_idx, 0]
    grad_ff = cgrads[ff_idx]
    return obj_ff, grad_ff

# Only check p-norm stress if it is present in your problem
has_pnorm_stress = (
    (to_params.Objective[0] == TO_QOI.PNORM_STRESS) or
    any(c[0] == TO_QOI.PNORM_STRESS for c in to_params.Constraints)
)
if has_pnorm_stress:
    print("Computing baseline objective and gradient for p-norm stress...")
    obj0, grad_obj, sol0 = solve_fea_and_compute_objective(zeta, TO_QOI.PNORM_STRESS)
    print(f"Baseline p-norm stress value: {obj0:.6e}\n")
else:
    print("PNORM STRESS is not present in this problem. Skipping p-norm stress sensitivity check.")
    obj0 = grad_obj = sol0 = None

# Baseline solution for failure factor (if available)
has_failure_factor = any(c[0] == TO_QOI.STRESS_FAILURE_FACTOR for c in to_params.Constraints)
if has_failure_factor:
    print("Computing baseline constraint value and gradient for failure factor...")
    # Solve for the actual objective (e.g., mass) to get the correct solution and postprocess
    obj_mass, grad_mass, sol_mass = solve_fea_and_compute_objective(zeta, to_params.Objective[0])
    fe_solver_structural.postprocess()  # Ensure stress fields are computed
    obj0_ff, grad_obj_ff = get_failure_factor_constraint(zeta, sol_mass)
    print(f"Baseline failure factor constraint value: {obj0_ff:.6e}\n")

nVariablesChosen = 5
zeta0 = zeta.copy()
np.random.seed(42)
random_elements = np.random.choice(num_elems, size=nVariablesChosen, replace=False)
print(f"Testing random elements: {random_elements}\n")

# General step sizes dictionary
step_sizes = {'SIMPdensity': 1e-3}
for jj in range(latentDim):
    step_sizes[f'latent_{jj}'] = 1e-3

if has_pnorm_stress:
    print("="*100)
    print("FEA SENSITIVITY VERIFICATION - CENTRAL DIFFERENCES (PNORM STRESS)")
    print("="*100)

    for jj in range(latentDim + 1):
        offset = jj * num_elems
        if jj == 0:
            var_type = 'SIMPdensity'
            perturbation = step_sizes['SIMPdensity']
            var_name = "DENSITY"
        else:
            var_type = f'latent_{jj-1}'
            perturbation = step_sizes[var_type]
            var_name = f"LATENT DIMENSION {jj-1}"

        print(f"\n{var_name} VARIABLES (h = {perturbation:.1e})")
        print("-" * 100)
        print(f"{'Element':<10} {'Analytic':<18} {'FD (Forward)':<18} {'FD (Central)':<18} {'Error (Fwd)':<12} {'Error (Ctr)':<12} {'Status':<10}")
        print("-" * 100)

        for elem in random_elements:
            idx = elem + offset
            # Forward perturbation
            zeta_plus = zeta0.copy()
            zeta_plus[idx] += perturbation
            obj_plus, _, _ = solve_fea_and_compute_objective(zeta_plus, TO_QOI.PNORM_STRESS)

            # Backward perturbation
            zeta_minus = zeta0.copy()
            zeta_minus[idx] -= perturbation
            obj_minus, _, _ = solve_fea_and_compute_objective(zeta_minus, TO_QOI.PNORM_STRESS)

            grad_fd_forward = (obj_plus - obj0) / perturbation
            grad_fd_central = (obj_plus - obj_minus) / (2 * perturbation)
            grad_analytic = grad_obj[idx]

            error_forward = abs(grad_analytic - grad_fd_forward) / (abs(grad_fd_forward) + 1e-12)
            error_central = abs(grad_analytic - grad_fd_central) / (abs(grad_fd_central) + 1e-12)

            if error_central < 1e-3:
                status = "✓ EXCELLENT"
            elif error_central < 1e-2:
                status = "✓ GOOD"
            elif error_central < 0.1:
                status = "⚠ OK"
            else:
                status = "✗ POOR"

            print(f"{elem:<10} {grad_analytic:<18.6e} {grad_fd_forward:<18.6e} {grad_fd_central:<18.6e} "
                  f"{error_forward:<12.3e} {error_central:<12.3e} {status:<10}")
else:
    print("PNORM STRESS is not present in this problem. Skipping p-norm stress sensitivity check.")

# Repeat for failure factor if available
if has_failure_factor:
    print("\n" + "="*100)
    print("FEA SENSITIVITY VERIFICATION - CENTRAL DIFFERENCES (FAILURE FACTOR)")
    print("="*100)

    for jj in range(latentDim + 1):
        offset = jj * num_elems
        if jj == 0:
            var_type = 'SIMPdensity'
            perturbation = step_sizes['SIMPdensity']
            var_name = "DENSITY"
        else:
            var_type = f'latent_{jj-1}'
            perturbation = step_sizes[var_type]
            var_name = f"LATENT DIMENSION {jj-1}"

        print(f"\n{var_name} VARIABLES (h = {perturbation:.1e})")
        print("-" * 100)
        print(f"{'Element':<10} {'Analytic':<18} {'FD (Forward)':<18} {'FD (Central)':<18} {'Error (Fwd)':<12} {'Error (Ctr)':<12} {'Status':<10}")
        print("-" * 100)

        for elem in random_elements:
            idx = elem + offset
            zeta_plus = zeta0.copy()
            zeta_plus[idx] += perturbation
            # Always use the actual problem objective for FEA solves
            _, _, sol_plus = solve_fea_and_compute_objective(zeta_plus, to_params.Objective[0])
            fe_solver_structural.postprocess()
            obj_plus_ff, _ = get_failure_factor_constraint(zeta_plus, sol_plus)

            zeta_minus = zeta0.copy()
            zeta_minus[idx] -= perturbation
            _, _, sol_minus = solve_fea_and_compute_objective(zeta_minus, to_params.Objective[0])
            fe_solver_structural.postprocess()
            obj_minus_ff, _ = get_failure_factor_constraint(zeta_minus, sol_minus)

            grad_fd_forward = (obj_plus_ff - obj0_ff) / perturbation
            grad_fd_central = (obj_plus_ff - obj_minus_ff) / (2 * perturbation)
            grad_analytic = grad_obj_ff[idx]

            error_forward = abs(grad_analytic - grad_fd_forward) / (abs(grad_fd_forward) + 1e-12)
            error_central = abs(grad_analytic - grad_fd_central) / (abs(grad_fd_central) + 1e-12)

            if error_central < 1e-3:
                status = "✓ EXCELLENT"
            elif error_central < 1e-2:
                status = "✓ GOOD"
            elif error_central < 0.1:
                status = "⚠ OK"
            else:
                status = "✗ POOR"

            print(f"{elem:<10} {grad_analytic:<18.6e} {grad_fd_forward:<18.6e} {grad_fd_central:<18.6e} "
                  f"{error_forward:<12.3e} {error_central:<12.3e} {status:<10}")

print("\nSensitivity verification completed.")