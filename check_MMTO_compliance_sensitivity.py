import numpy as np
import torch
import os

from MMTO_TS_examples import MMTOThermostructuralExamples, getMMTOThermostructuralProblem
from materialEncoder import MaterialEncoder
from MMTO_obj_cons_sensitivities_TS import (
    compute_mmto_objective_and_gradient,
)
from PyTOImports import *

to_problem = MMTOThermostructuralExamples.MBBBeam

mesh, mat_prop, structural_bc, thermal_bc, elem_body_force, to_params, vae_params = getMMTOThermostructuralProblem(to_problem)
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
    elem_body_force=elem_body_force)

fe_solver_thermal = hex_thermal_fea.HexThermalFEA(
    mesh=mesh,
    mat_prop=mat_prop,
    bc=thermal_bc,
    solver=solver,
    rtol=1e-8)

KETemplate = hex_element_stiffness.hex8_stiffness_matrix_structural(1.0, 0.3, mesh.elem_size)
KTTemplate = hex_element_stiffness.hex8_stiffness_matrix_thermal(1.0, mesh.elem_size)

num_elems = mesh.num_elems
latentDim = vae_params.latentDim
num_design_var = num_elems + num_elems * latentDim

zeta = np.zeros(num_design_var).flatten()
zeta[0:num_elems] = 0.5
zeta[num_elems:] = 0
# Helper function to solve FEA and compute objective
def solve_fea_and_compute_objective(zeta_input):
    """Solve FEA and return objective value."""
    xDesign = zeta_input[0:num_elems]
    zDesign = zeta_input[num_elems:]
    zPoints = torch.tensor(zDesign, dtype=torch.float32).view(latentDim, -1).T
    
    decoded = matEncoder.vaeNet.decoder(zPoints)
    material_properties = matEncoder.getMaterialProperties(decoded)
    Youngs_Modulus = material_properties['Youngs_Modulus'].detach().cpu().numpy()
    Thermal_Conductivity = material_properties['Conductivity'].detach().cpu().numpy()
    Thermal_Expansion = material_properties['Thermal_Expansion'].detach().cpu().numpy()
    mass_density = material_properties['Density'].detach().cpu().numpy()
   
    
    # Set per-element material properties
    fe_solver_structural.mat_prop = [
        mat_lib.create_material_with_defaults(
            name=f"Material_{i+1}",
            youngs_modulus=Youngs_Modulus[i],
            thermal_expansion_coefficient=Thermal_Expansion[i],
            thermal_conductivity=Thermal_Conductivity[i],
            mass_density=mass_density[i]
        )
        for i in range(len(Youngs_Modulus))]
    fe_solver_structural.set_material(fe_solver_structural.mat_prop)
    
    fe_solver_thermal.mat_prop = [
        mat_lib.create_material_with_defaults(
            name=f"Material_{i+1}",
            youngs_modulus=Youngs_Modulus[i],
            thermal_expansion_coefficient=Thermal_Expansion[i],
            thermal_conductivity=Thermal_Conductivity[i],
            mass_density=mass_density[i]
        )
        for i in range(len(Thermal_Conductivity))]
    fe_solver_thermal.set_material(fe_solver_thermal.mat_prop)
    
    # Solve thermal problem
    temperature = fe_solver_thermal.solve(xDesign, MaterialModel.SIMP)
    fe_solver_thermal.postprocess()
    
    # Get thermoelastic force and solve structural problem
    thermo_elastic_force = fe_solver_thermal.get_thermoelastic_force(xDesign, MaterialModel.SIMP)
    fe_solver_structural.set_thermal_forces(thermo_elastic_force)
    displacement = fe_solver_structural.solve(xDesign, MaterialModel.SIMP)
    fe_solver_structural.mesh.setPseudoDensity(xDesign)
    fe_solver_structural.postprocess()
    
    obj, grad = compute_mmto_objective_and_gradient(
        to_params,
        displacement,
        temperature,
        zeta_input,
        fe_solver_structural,
        KETemplate,
        KTTemplate,
        matEncoder,
        fe_solver_thermal
    )
    
    return obj, grad

# Baseline solution
print("Computing baseline objective and gradient...")
obj0, grad_obj = solve_fea_and_compute_objective(zeta)
print(f"Baseline objective value: {obj0:.6e}\n")

# Test parameters
nVariablesChosen = 5
zeta0 = zeta.copy()

# Select 5 random elements (same for all variable types for fair comparison)
np.random.seed(42)  # For reproducibility
random_elements = np.random.choice(num_elems, size=nVariablesChosen, replace=False)
print(f"Testing random elements: {random_elements}\n")

# Adaptive step sizes
step_sizes = {
    'SIMPdensity': 1e-3,
    'latent_0': 1e-3,
    'latent_1': 1e-3,
}

print("="*100)
print("FEA SENSITIVITY VERIFICATION - CENTRAL DIFFERENCES")
print("="*100)

for jj in range(latentDim + 1):
    offset = jj * num_elems
    
    # Determine step size and variable type
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
    
    # Table header
    print(f"{'Element':<10} {'Analytic':<18} {'FD (Forward)':<18} {'FD (Central)':<18} {'Error (Fwd)':<12} {'Error (Ctr)':<12} {'Status':<10}")
    print("-" * 100)
    
    results = []
    
    for elem in random_elements:
        idx = elem + offset
        
        # Forward perturbation
        zeta_plus = zeta0.copy()
        zeta_plus[idx] += perturbation
        obj_plus, _ = solve_fea_and_compute_objective(zeta_plus)
        
        # Backward perturbation
        zeta_minus = zeta0.copy()
        zeta_minus[idx] -= perturbation
        obj_minus, _ = solve_fea_and_compute_objective(zeta_minus)
        
        # Compute gradients
        grad_fd_forward = (obj_plus - obj0) / perturbation
        grad_fd_central = (obj_plus - obj_minus) / (2 * perturbation)
        grad_analytic = grad_obj[idx]
        
        # Errors
        error_forward = abs(grad_analytic - grad_fd_forward) / (abs(grad_fd_forward) + 1e-12)
        error_central = abs(grad_analytic - grad_fd_central) / (abs(grad_fd_central) + 1e-12)
        
        # Status
        if error_central < 1e-3:
            status = "✓ EXCELLENT"
        elif error_central < 1e-2:
            status = "✓ GOOD"
        elif error_central < 0.1:
            status = "⚠ OK"
        else:
            status = "✗ POOR"
        
        # Print row
        print(f"{elem:<10} {grad_analytic:<18.6e} {grad_fd_forward:<18.6e} {grad_fd_central:<18.6e} "
              f"{error_forward:<12.3e} {error_central:<12.3e} {status:<10}")
        
        results.append({
            'elem': elem,
            'analytic': grad_analytic,
            'fd_fwd': grad_fd_forward,
            'fd_ctr': grad_fd_central,
            'err_fwd': error_forward,
            'err_ctr': error_central
        })
    
print("\nSensitivity verification completed.")