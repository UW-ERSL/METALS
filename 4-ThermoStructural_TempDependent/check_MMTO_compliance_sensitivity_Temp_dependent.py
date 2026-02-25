import numpy as np
import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../0-Common')))

from ExamplesThermoStructural import MMTOThermostructuralExamples, getMMTOThermostructuralProblem
from materialEncoder import MaterialEncoder  # type: ignore

# TEMP-DEPENDENT sensitivities module (your new file)
from SensitivitiesThermoStructural_TempDep import (
    compute_mmto_objective_and_gradient_tempdep,
)

from PyTOImports import (  # type: ignore
    linear_solvers, hex_structural_fea, hex_element_stiffness, hex_thermal_fea,
    mat_lib, MaterialModel
)

# ----------------------------
# Problem setup
# ----------------------------
to_problem = MMTOThermostructuralExamples.MBBBeam

mesh, mat_prop, structural_bc, thermal_bc, elem_body_force, to_params, vae_params = \
    getMMTOThermostructuralProblem(to_problem)

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
    bc=structural_bc,
    solver=solver,
    rtol=1e-8,
    elem_body_force=elem_body_force
)

fe_solver_thermal = hex_thermal_fea.HexThermalFEA(
    mesh=mesh,
    mat_prop=mat_prop,
    bc=thermal_bc,
    solver=solver,
    rtol=1e-8
)

KETemplate = hex_element_stiffness.hex8_stiffness_matrix_structural(1.0, 0.3, mesh.elem_size)
KTTemplate = hex_element_stiffness.hex8_stiffness_matrix_thermal(1.0, mesh.elem_size)

num_elems = mesh.num_elems
latentDim = vae_params.latentDim
num_design_var = num_elems + num_elems * latentDim

# ----------------------------
# Initial design
# ----------------------------
zeta = np.zeros(num_design_var).flatten()
zeta[0:num_elems] = 0.5
zeta[num_elems:] = 0.0

# ----------------------------
# Temp-dependent thermal solve settings
# ----------------------------
# Use the same Ta/Tf you used in the example definition (or read from kwargs if you stored it).
# If you didn't store them in to_params, hard-code here to match the test.
Ta = 23.0
maxThermalPicard = 3
thermalPicardTol = 1e-6

def compute_Telem_from_Tnodes(T_nodes):
    T_e = T_nodes[mesh.edofMatThermal]   # (nelem, 8)
    return T_e.mean(axis=1)              # (nelem,)

# ----------------------------
# Helper: Solve FEA and compute objective + gradient (TEMP-DEPENDENT)
# ----------------------------
def solve_fea_and_compute_objective_tempdep(zeta_input):
    """
    Solve thermal+structural with temp-dependent E(T), alpha(T), K(T),
    then return objective and gradient from temp-dependent sensitivity module.
    """
    zeta_input = np.asarray(zeta_input).flatten()
    xDesign = zeta_input[0:num_elems]
    zDesign = zeta_input[num_elems:]
    zPoints = torch.tensor(zDesign, dtype=torch.float32).view(latentDim, -1).T  # (nelem, latentDim)

    # ---------- Picard loop for thermal with K(Telem) ----------
    Telem = Ta * np.ones(num_elems, dtype=float)
    temperature = None

    for pic in range(maxThermalPicard):
        # K(Telem)
        with torch.no_grad():
            K_T = matEncoder.getMaterialPropertyAtTemperatureTorch(
                "K", zPoints, torch.tensor(Telem, dtype=torch.float32)
            ).detach().cpu().numpy()

        # set thermal materials using K(T)
        fe_solver_thermal.mat_prop = [
            mat_lib.create_material_with_defaults(
                name=f"Material_{i+1}",
                thermal_conductivity=float(K_T[i]),
                youngs_modulus=1.0,
                thermal_expansion_coefficient=0.0
            )
            for i in range(num_elems)
        ]
        fe_solver_thermal.set_material(fe_solver_thermal.mat_prop)

        # solve thermal
        temperature = fe_solver_thermal.solve(xDesign, MaterialModel.SIMP)
        fe_solver_thermal.postprocess()

        Telem_new = compute_Telem_from_Tnodes(temperature)
        rel = np.linalg.norm(Telem_new - Telem) / max(1e-12, np.linalg.norm(Telem))
        Telem = Telem_new
        if rel < thermalPicardTol:
            break

    # ---------- Evaluate E(T), alpha(T), K(T) at final Telem ----------
    Telem_torch = torch.tensor(Telem, dtype=torch.float32)
    with torch.no_grad():
        E_T = matEncoder.getMaterialPropertyAtTemperatureTorch("E", zPoints, Telem_torch).detach().cpu().numpy()
        A_T = matEncoder.getMaterialPropertyAtTemperatureTorch("Alpha", zPoints, Telem_torch).detach().cpu().numpy()
        K_T = matEncoder.getMaterialPropertyAtTemperatureTorch("K", zPoints, Telem_torch).detach().cpu().numpy()

    # Density is NOT temp-dependent in your sheet; it is decoded directly
    with torch.no_grad():
        decoded = matEncoder.vaeNet.decoder(zPoints)
        props = matEncoder.getMaterialProperties(decoded)
        rho = props["Density"].detach().cpu().numpy()

    # set structural materials (E, alpha)
    fe_solver_structural.mat_prop = [
        mat_lib.create_material_with_defaults(
            name=f"Material_{i+1}",
            youngs_modulus=float(E_T[i]),
            thermal_expansion_coefficient=float(A_T[i]),
            thermal_conductivity=float(K_T[i]),
            mass_density=float(rho[i])
        )
        for i in range(num_elems)
    ]
    fe_solver_structural.set_material(fe_solver_structural.mat_prop)

    # ---------- FINAL CONSISTENT THERMAL STATE ----------
    # We must ensure that:
    #   (1) temperature was solved using the same fe_solver_thermal materials/stiffness
    #       that the adjoint later uses (fe_solver_thermal.stiff_mtrx)
    #   (2) get_thermoelastic_force() reads E/alpha consistent with that temperature.
    #
    # Since mat_prop is immutable, the safe approach is:
    #   set final full thermal mat_prop -> re-solve thermal once -> recompute Telem
    #   -> (optionally) re-evaluate E/alpha/K -> set materials again.

    # 1) Set full thermal mat_prop (E/alpha/K) and RE-SOLVE thermal once
    fe_solver_thermal.mat_prop = [
        mat_lib.create_material_with_defaults(
            name=f"Material_{i+1}",
            youngs_modulus=float(E_T[i]),
            thermal_expansion_coefficient=float(A_T[i]),
            thermal_conductivity=float(K_T[i]),
            mass_density=float(rho[i])
        )
        for i in range(num_elems)
    ]
    fe_solver_thermal.set_material(fe_solver_thermal.mat_prop)

    temperature = fe_solver_thermal.solve(xDesign, MaterialModel.SIMP)
    fe_solver_thermal.postprocess()

    Telem = compute_Telem_from_Tnodes(temperature)

    # 2) Re-evaluate properties at updated Telem (recommended)
    Telem_torch = torch.tensor(Telem, dtype=torch.float32)
    with torch.no_grad():
        E_T = matEncoder.getMaterialPropertyAtTemperatureTorch("E", zPoints, Telem_torch).detach().cpu().numpy()
        A_T = matEncoder.getMaterialPropertyAtTemperatureTorch("Alpha", zPoints, Telem_torch).detach().cpu().numpy()
        K_T = matEncoder.getMaterialPropertyAtTemperatureTorch("K", zPoints, Telem_torch).detach().cpu().numpy()

    # 3) Update BOTH solvers with the final consistent E/alpha/K
    fe_solver_structural.mat_prop = [
        mat_lib.create_material_with_defaults(
            name=f"Material_{i+1}",
            youngs_modulus=float(E_T[i]),
            thermal_expansion_coefficient=float(A_T[i]),
            thermal_conductivity=float(K_T[i]),
            mass_density=float(rho[i])
        )
        for i in range(num_elems)
    ]
    fe_solver_structural.set_material(fe_solver_structural.mat_prop)

    fe_solver_thermal.mat_prop = [
        mat_lib.create_material_with_defaults(
            name=f"Material_{i+1}",
            youngs_modulus=float(E_T[i]),
            thermal_expansion_coefficient=float(A_T[i]),
            thermal_conductivity=float(K_T[i]),
            mass_density=float(rho[i])
        )
        for i in range(num_elems)
    ]
    fe_solver_thermal.set_material(fe_solver_thermal.mat_prop)

    # 4) Thermoelastic force + structural solve
    thermo_elastic_force = fe_solver_thermal.get_thermoelastic_force(xDesign, MaterialModel.SIMP)
    fe_solver_structural.set_thermal_forces(thermo_elastic_force)

    displacement = fe_solver_structural.solve(xDesign, MaterialModel.SIMP)
    fe_solver_structural.mesh.setPseudoDensity(xDesign)
    fe_solver_structural.postprocess()

    # 5) Objective + gradient (IMPORTANT: pass UPDATED temperature and Telem)
    obj, grad = compute_mmto_objective_and_gradient_tempdep(
        to_params=to_params,
        displacement=displacement,
        temperature=temperature,
        Telem=Telem,
        zeta=zeta_input,
        fe_solver_structural=fe_solver_structural,
        KETemplate_unitE=KETemplate,
        KTTemplate_unitK=KTTemplate,
        matEncoder=matEncoder,
        fe_solver_thermal=fe_solver_thermal
    )
    return obj, grad


# ----------------------------
# Baseline
# ----------------------------
print("Computing baseline objective and gradient (TEMP-DEP)...")
obj0, grad_obj = solve_fea_and_compute_objective_tempdep(zeta)
print(f"Baseline objective value: {obj0:.6e}\n")

# ----------------------------
# FD tests
# ----------------------------
nVariablesChosen = 5
zeta0 = zeta.copy()

np.random.seed(42)
random_elements = np.random.choice(num_elems, size=nVariablesChosen, replace=False)
print(f"Testing random elements: {random_elements}\n")

# FD step sizes (temp-dependent coupling can make these more sensitive)
step_sizes = {
    'SIMPdensity': 1e-3,
    'latent_0': 1e-3,
    'latent_1': 1e-3,
}

print("=" * 100)
print("FEA SENSITIVITY VERIFICATION (TEMP-DEPENDENT) - CENTRAL DIFFERENCES")
print("=" * 100)

for jj in range(latentDim + 1):
    offset = jj * num_elems

    if jj == 0:
        perturbation = step_sizes['SIMPdensity']
        var_name = "DENSITY"
    else:
        perturbation = step_sizes[f'latent_{jj-1}']
        var_name = f"LATENT DIMENSION {jj-1}"

    print(f"\n{var_name} VARIABLES (h = {perturbation:.1e})")
    print("-" * 100)
    print(f"{'Element':<10} {'Analytic':<18} {'FD (Forward)':<18} {'FD (Central)':<18} {'Error (Fwd)':<12} {'Error (Ctr)':<12} {'Status':<10}")
    print("-" * 100)

    for elem in random_elements:
        idx = elem + offset

        zeta_plus = zeta0.copy()
        zeta_plus[idx] += perturbation
        obj_plus, _ = solve_fea_and_compute_objective_tempdep(zeta_plus)

        zeta_minus = zeta0.copy()
        zeta_minus[idx] -= perturbation
        obj_minus, _ = solve_fea_and_compute_objective_tempdep(zeta_minus)

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

print("\nTEMP-DEPENDENT sensitivity verification completed.")