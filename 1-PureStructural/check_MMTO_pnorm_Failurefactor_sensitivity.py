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
try:
    from SensitivitiesPureStructural import (
        isotropic_constitutive_matrix_dnu,
        hex8_stiffness_matrix_structural_dnu,
    )
    HAS_ANALYTIC_NU_DERIVS = True
except Exception as e:
    print("[WARN] Could not import analytic nu-derivative helpers from SensitivitiesPureStructural.")
    print("       Add them there first (isotropic_constitutive_matrix_dnu and hex8_stiffness_matrix_structural_dnu).")
    print("       Error:", repr(e))
    HAS_ANALYTIC_NU_DERIVS = False
import argparse

# ----------------------------
# Poisson ratio FD toggle
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--use_constant_nu", type=str, default="True",
                    help="True: run FD check with constant Poisson ratio. False: variable Poisson ratio decoded from VAE.")
parser.add_argument("--constant_nu", type=float, default=0.3,
                    help="Constant Poisson ratio value used when --use_constant_nu True.")
args, _ = parser.parse_known_args()

USE_CONSTANT_POISSONS_RATIO = (args.use_constant_nu.lower() in ("true", "1", "yes", "y"))
CONSTANT_POISSONS_RATIO = float(args.constant_nu)

print(f"\n[FD MODE] USE_CONSTANT_POISSONS_RATIO={USE_CONSTANT_POISSONS_RATIO}, CONSTANT_POISSONS_RATIO={CONSTANT_POISSONS_RATIO}\n")

to_problem = MMTOExamplesPureStructural.LBracketTopLoad_Stress_BM1

mesh, mat_prop, bc, elem_body_force, to_params, vae_params = getMMTOProblemPureStructural(to_problem)
# Pass nu-mode down to sensitivity routines
to_params.use_constant_poissons_ratio = bool(USE_CONSTANT_POISSONS_RATIO)
to_params.constant_poissons_ratio = float(CONSTANT_POISSONS_RATIO)

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

nu_template = CONSTANT_POISSONS_RATIO if USE_CONSTANT_POISSONS_RATIO else 0.3  # template unused in variable-nu path
KETemplate = hex_element_stiffness.hex8_stiffness_matrix_structural(1.0, float(nu_template), mesh.elem_size)

def sanity_test_dD_dnu(E=1.0, nu=0.33, h=1e-6, seed=0):
    """
    1-element sanity check for constitutive matrix derivative dD/dnu:
      - compares analytic dD/dnu vs central FD on D(E,nu)
      - also checks a scalar quadratic form: eps^T D eps
    """
    if not HAS_ANALYTIC_NU_DERIVS:
        print("[SKIP] dD/dnu sanity test (analytic helpers not available).")
        return

    D_plus = hex_element_stiffness.isotropic_constitutive_matrix(E, nu + h)
    D_minus = hex_element_stiffness.isotropic_constitutive_matrix(E, nu - h)
    dD_fd = (D_plus - D_minus) / (2.0 * h)

    dD_an = isotropic_constitutive_matrix_dnu(E, nu)

    rel_err = np.linalg.norm(dD_an - dD_fd) / (np.linalg.norm(dD_fd) + 1e-16)

    # Quadratic form check (sign check is easiest to interpret)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(6)
    q_plus = eps @ D_plus @ eps
    q_minus = eps @ D_minus @ eps
    dq_fd = (q_plus - q_minus) / (2.0 * h)

    dq_an = eps @ dD_an @ eps

    print("\n" + "=" * 100)
    print("SANITY TEST 1/2: dD/dnu (3D isotropic constitutive matrix)")
    print("=" * 100)
    print(f"E={E:.6g}, nu={nu:.6g}, h={h:.1e}")
    print(f"||dD_an - dD_fd|| / ||dD_fd|| = {rel_err:.3e}")
    print(f"Quadratic form check: dq_fd={dq_fd:.6e}, dq_an={dq_an:.6e}, "
          f"rel_err={abs(dq_an-dq_fd)/(abs(dq_fd)+1e-16):.3e}")
    print("Interpretation: rel_err ~ 1e-6 to 1e-3 is typically fine (depends on h).")


def sanity_test_dKE_dnu(E=1.0, nu=0.33, h=1e-6, seed=0):
    """
    1-element sanity check for element stiffness derivative dKE/dnu:
      - compares analytic dKE/dnu vs central FD on KE(E,nu)
      - also checks a scalar quadratic form: u^T KE u
    """
    if not HAS_ANALYTIC_NU_DERIVS:
        print("[SKIP] dKE/dnu sanity test (analytic helpers not available).")
        return

    KE_plus = hex_element_stiffness.hex8_stiffness_matrix_structural(E, nu + h, mesh.elem_size)
    KE_minus = hex_element_stiffness.hex8_stiffness_matrix_structural(E, nu - h, mesh.elem_size)
    dKE_fd = (KE_plus - KE_minus) / (2.0 * h)

    dKE_an = hex8_stiffness_matrix_structural_dnu(E, nu, mesh.elem_size)

    rel_err = np.linalg.norm(dKE_an - dKE_fd) / (np.linalg.norm(dKE_fd) + 1e-16)

    # Quadratic form check (very interpretable)
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(24)
    q_plus = u @ KE_plus @ u
    q_minus = u @ KE_minus @ u
    dq_fd = (q_plus - q_minus) / (2.0 * h)

    dq_an = u @ dKE_an @ u

    print("\n" + "=" * 100)
    print("SANITY TEST 2/2: dKE/dnu (hex8 element stiffness)")
    print("=" * 100)
    print(f"E={E:.6g}, nu={nu:.6g}, h={h:.1e}")
    print(f"||dKE_an - dKE_fd|| / ||dKE_fd|| = {rel_err:.3e}")
    print(f"Quadratic form check: dq_fd={dq_fd:.6e}, dq_an={dq_an:.6e}, "
          f"rel_err={abs(dq_an-dq_fd)/(abs(dq_fd)+1e-16):.3e}")
    print("Interpretation: rel_err ~ 1e-6 to 1e-3 is typically fine (depends on h).")
# --- Run 1-element sanity tests for nu-derivatives (cheap, fast, independent of MMTO) ---
sanity_test_dD_dnu(E=1.0, nu=0.33, h=1e-6, seed=1)
sanity_test_dKE_dnu(E=1.0, nu=0.33, h=1e-6, seed=1)

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
    # ----------------------------
    # Poisson ratio selection (constant or decoded)
    # ----------------------------
    nu_key = None
    for k in ("Poissons_Ratio", "Poisson_Ratio", "poissons_ratio", "poisson_ratio", "nu", "Nu"):
        if k in material_properties:
            nu_key = k
            break

    if USE_CONSTANT_POISSONS_RATIO:
        nu_vals = np.full_like(Youngs_Modulus, CONSTANT_POISSONS_RATIO, dtype=float)
    else:
        if nu_key is None:
            raise RuntimeError("Variable-nu FD mode requested, but Poisson ratio key not found in decoded material_properties.")
        nu_vals = material_properties[nu_key].detach().cpu().numpy().astype(float)

    if 'Yield_Strength' in material_properties:
        Yield_Strength = material_properties['Yield_Strength'].detach().cpu().numpy()
    else:
        Yield_Strength = None
    fe_solver_structural.mat_prop = []
    for i in range(len(Youngs_Modulus)):
        mp = mat_lib.create_material_with_defaults(
            name=f"Material_{i+1}",
            youngs_modulus=float(Youngs_Modulus[i]),
            mass_density=float(mass_density[i]),
            yield_strength=float(Yield_Strength[i]) if Yield_Strength is not None else None
        )

        # Force Poisson's ratio on the material object (avoid relying on defaults)
        if hasattr(mp, "poissons_ratio"):
            mp.poissons_ratio = float(nu_vals[i])
        elif hasattr(mp, "nu"):
            mp.nu = float(nu_vals[i])
        else:
            raise RuntimeError("Material object has no attribute for Poisson ratio (poissons_ratio or nu).")

        fe_solver_structural.mat_prop.append(mp)

    fe_solver_structural.set_material(fe_solver_structural.mat_prop)

    # Debug check (prints once per FD evaluation call; comment out if noisy)
    nus = [float(getattr(m, "poissons_ratio", getattr(m, "nu"))) for m in fe_solver_structural.mat_prop]
    print(f"[FEA nu] min={min(nus):.6g}, max={max(nus):.6g}, key={'CONST' if USE_CONSTANT_POISSONS_RATIO else nu_key}")

    fe_solver_structural.set_material(fe_solver_structural.mat_prop)

    sol = fe_solver_structural.solve(xDesign, MaterialModel.SIMP)
    fe_solver_structural.mesh.setPseudoDensity(xDesign)
    fe_solver_structural.postprocess()

    to_params.Objective = (objectiveType, None)
    print(f"[SENS nu-mode] to_params.use_constant_poissons_ratio={getattr(to_params, 'use_constant_poissons_ratio', None)} "
          f"to_params.constant_poissons_ratio={getattr(to_params, 'constant_poissons_ratio', None)}")

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