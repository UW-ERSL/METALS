import numpy as np
import linear_solvers
import bound_cond
from PyTOImports import *

# --- Support Functions ---
from MMTO_obj_cons_sensitivities import (
        compute_pnorm_safety_factor_and_sensitivity, 
        compute_pnorm_stress_and_sensitivity,
        compute_volumefraction_constraint_and_gradient
)
def solve_thermal_adjoint_multimaterial(x,displacement, fe_thermal_solver, fe_structural_solver, E, alpha, material_model):
    """
    Multi-material thermal adjoint equation:
    K_T * lambda_T = -sum_e (xi_e^p * E_e * alpha_e * H^T * d_e)
    """
    nelem = fe_thermal_solver.mesh.num_elems
    num_thermal_dofs = fe_thermal_solver.mesh.num_nodes

    rhs = np.zeros(num_thermal_dofs)
    dx, dy, dz = fe_structural_solver.mesh.elem_size

    # Get per-element Poisson's ratio
    if isinstance(fe_structural_solver.mat_prop, list):
        nu = np.array([fe_structural_solver.mat_prop[i].poissons_ratio for i in range(nelem)])
    else:
        nu = np.full(nelem, fe_structural_solver.mat_prop.poissons_ratio)

    for e in range(nelem):
        edof_s = fe_structural_solver.mesh.edofMatStructural[e, :]
        edof_t = fe_thermal_solver.mesh.edofMatThermal[e, :]
        # Compute HMatrix for this element's nu
        HMatrix_e = fe_thermal_solver.getHMatrix(dx, dy, dz, nu[e])
        # Contribution from this element
        rhs_e = -2 * E[e] * alpha[e] * get_structural_material_model_scaling(x[e], material_model) * HMatrix_e.T @ displacement[edof_s]
        rhs[edof_t] += rhs_e

    K_T = fe_thermal_solver.stiff_mtrx
    bcAdjoint = bound_cond.BC(
        force=np.zeros_like(fe_thermal_solver.bc.force),
        fixed_dofs=fe_thermal_solver.bc.fixed_dofs,
        dirichlet_values=np.zeros_like(fe_thermal_solver.bc.dirichlet_values)
    )
    lambda_T = linear_solvers.solve(
        K_T,
        rhs,
        fe_thermal_solver.solver,
        bcAdjoint,
        **fe_thermal_solver.kwargs
    )
    return lambda_T

def compute_thermoelastic_compliance_and_gradient_density_multimaterial(
    x, temperature, displacement, to_params,
    fe_thermal_solver, fe_structural_solver, E, alpha, K, KSTemplate, KTTemplate
):
    """
    Multi-material thermoelastic compliance sensitivity.
    Uses per-element H matrix and provided KETemplate/KTTemplate.
    """
    material_model = to_params.materialModel
    J = displacement.T @ fe_structural_solver.stiff_mtrx @ displacement
    nelem = fe_structural_solver.mesh.num_elems

    dJ_dx = np.zeros(nelem)


    # Get per-element Poisson's ratio
    if isinstance(fe_structural_solver.mat_prop, list):
        nu = np.array([fe_structural_solver.mat_prop[i].poissons_ratio for i in range(nelem)])
    else:
        nu = np.full(nelem, fe_structural_solver.mat_prop.poissons_ratio)

    #Solve thermal adjoint equation
    lambda_T = solve_thermal_adjoint_multimaterial(x, displacement, fe_thermal_solver, fe_structural_solver, E, alpha, material_model)

    term1 = np.zeros(nelem)
    term2 = np.zeros(nelem)
    term3 = np.zeros(nelem)

    for e in range(nelem):
        edof_s = fe_structural_solver.mesh.edofMatStructural[e, :]
        edof_t = fe_thermal_solver.mesh.edofMatThermal[e, :]
        d_e = displacement[edof_s]
        T_e = temperature[edof_t]
        lambda_T_e = lambda_T[edof_t]

        # Per-element H matrix
        dx, dy, dz = fe_structural_solver.mesh.elem_size
        HMatrix_e = fe_thermal_solver.getHMatrix(dx, dy, dz, nu[e])

        # Term 1: Direct structural stiffness contribution (multiply KSTemplate by E)
        term1[e] = -get_structural_material_model_sensitivity(x[e], material_model) * d_e.T @ (E[e] * KSTemplate) @ d_e

        # Term 2: Direct thermal force contribution (multi-material)
        T_diff = T_e - fe_thermal_solver.thermoElasticReferenceTemperature
        term2[e] = 2 * get_structural_material_model_sensitivity(x[e], material_model) * E[e] * alpha[e] * d_e.T @ HMatrix_e @ T_diff

        # Term 3: Adjoint thermal contribution (multiply KTTemplate by K)
        term3[e] = get_thermal_material_model_sensitivity(x[e], material_model) * lambda_T_e.T @ (K[e] * KTTemplate) @ T_e

        dJ_dx[e] = term1[e] + term2[e] + term3[e]
    
    return J, dJ_dx,lambda_T

def compute_thermoelastic_compliance_and_gradient_latent_multimaterial(
    zeta, temperature, displacement, to_params,
    fe_thermal_solver, fe_structural_solver,
    E, alpha, K,
    KSTemplate, KTTemplate,
    dE_dz, dAlpha_dz, dK_dz, material_model,
    lambda_T=None
):
    """
    Computes thermoelastic compliance gradient w.r.t. latent variables.
    Uses decoder gradients dE_dz, dAlpha_dz, dK_dz (shape: latent_dim x nelem).
    Returns dJdz with shape (nelem * latent_dim,)
    """

    nelem = fe_structural_solver.mesh.num_elems
    x = zeta[:nelem]
    latent_dim = dE_dz.shape[0]  # Now latent_dim is first axis

    J = displacement.T @ fe_structural_solver.stiff_mtrx @ displacement

    # If adjoint not provided, compute it
    if lambda_T is None:
        x = zeta[:nelem]
        lambda_T = solve_thermal_adjoint_multimaterial(
            x, displacement, fe_thermal_solver, fe_structural_solver,
            E, alpha, to_params.materialModel
        )

    dJ_dz = np.zeros((nelem, latent_dim))

    # Poisson ratio per element
    if isinstance(fe_structural_solver.mat_prop, list):
        nu = np.array([fe_structural_solver.mat_prop[i].poissons_ratio for i in range(nelem)])
    else:
        nu = np.full(nelem, fe_structural_solver.mat_prop.poissons_ratio)

    dx, dy, dz_elem = fe_structural_solver.mesh.elem_size
    Tref = fe_thermal_solver.thermoElasticReferenceTemperature

    for e in range(nelem):
        edof_s = fe_structural_solver.mesh.edofMatStructural[e, :]
        edof_t = fe_thermal_solver.mesh.edofMatThermal[e, :]

        d_e = displacement[edof_s]
        T_e = temperature[edof_t]
        lambda_T_e = lambda_T[edof_t]

        H_e = fe_thermal_solver.getHMatrix(dx, dy, dz_elem, nu[e])
        Tdiff = T_e - Tref

        # Decoder gradients for this element (now shape: (latent_dim,))
        dE_dz_e = dE_dz[:, e]
        dAlpha_dz_e = dAlpha_dz[:, e]
        dK_dz_e = dK_dz[:, e]

        # ---- Term 1: structural stiffness contribution ----
        term1 = get_structural_material_model_scaling(x[e], material_model) * dE_dz_e * (d_e.T @ KSTemplate @ d_e)

        # ---- Term 2: thermal force contribution (E sensitivity) ----
        term2 = -2.0 * get_structural_material_model_scaling(x[e], material_model) * dE_dz_e * alpha[e] * (d_e.T @ H_e @ Tdiff)

        # ---- Term 3: thermal force contribution (alpha sensitivity) ----
        term3 = 2 * get_structural_material_model_scaling(x[e], material_model) * E[e] * dAlpha_dz_e * (d_e.T @ H_e @ Tdiff)

        # ---- Term 4: thermal adjoint contribution (k sensitivity) ----
        term4 = get_thermal_material_model_scaling(x[e], material_model) * dK_dz_e * (lambda_T_e.T @ (KTTemplate @ T_e))

        dJ_dz[e, :] = term1 + term2 + term3 + term4

    # Return as flat array (ordering: [z_0, z_1, ..., z_{nelem-1}], each of length latent_dim)
    return J, dJ_dz.flatten(), lambda_T

# --- Main Objective/Constraint Functions ---
def compute_mmto_objective_and_gradient(
    to_params, sol,temperature, zeta, fe_solver_structural, KETemplate,KTTemplate, matEncoder,
 fe_solver_thermal
):
    """
    Compute objective value and its gradient for METALS LSR.
    Handles compliance, mass, cost, criticality, etc. in a modular way.
    """
    objectiveType = to_params.Objective[0]
    optionalParam = to_params.Objective[1]
    num_elems = fe_solver_structural.mesh.num_elems
    x = zeta[0:num_elems]
    latentDim = matEncoder.vae_params.latentDim

    zPts = zeta[num_elems:].reshape((latentDim, -1)).T
    material_properties, gradients = matEncoder.getMaterialPropertiesAtLatentPoints(zPts, compute_gradients=True)
    material_model = to_params.materialModel

    # Consistent extraction for all properties and gradients
    if 'Youngs_Modulus' in material_properties:
        EDesign = material_properties['Youngs_Modulus'].detach().cpu().numpy()
        dE_dz = gradients['Youngs_Modulus'].detach().cpu().numpy().T
        print("shape dE_dz:", dE_dz.shape)
    else:
        EDesign = None
        dE_dz = None

    if 'Thermal_Expansion' in material_properties:
        alphaDesign = material_properties['Thermal_Expansion'].detach().cpu().numpy()
        dAlpha_dz = gradients['Thermal_Expansion'].detach().cpu().numpy().T
    else:
        alphaDesign = None
        dAlpha_dz = None

    if 'Conductivity' in material_properties:
        KDesign = material_properties['Conductivity'].detach().cpu().numpy()
        dK_dz = gradients['Conductivity'].detach().cpu().numpy().T
    else:
        KDesign = None
        dK_dz = None

    if 'Density' in material_properties:
        mass_density = material_properties['Density'].detach().cpu().numpy()
        dMassDensity_dz = gradients['Density'].detach().cpu().numpy().T
    else:
        mass_density = None
        dMassDensity_dz = None

    if 'Cost' in material_properties:
        cost_per_unitmass = material_properties['Cost'].detach().cpu().numpy()
        dCost_dz = gradients['Cost'].detach().cpu().numpy().T
    else:
        cost_per_unitmass = None
        dCost_dz = None

    if 'Criticality' in material_properties:
        criticality = material_properties['Criticality'].detach().cpu().numpy()
        dCriticality_dz = gradients['Criticality'].detach().cpu().numpy().T
    else:
        criticality = None
        dCriticality_dz = None

    if 'Yield_Strength' in material_properties:
        YDesign = material_properties['Yield_Strength'].detach().cpu().numpy()
        dY_dz = gradients['Yield_Strength'].detach().cpu().numpy().T
    else:
        YDesign = None
        dY_dz = None

    pNormExponent = get_pNorm_exponent()

    if objectiveType == TO_QOI.COMPLIANCE:

        J, dJdx, lambda_T = compute_thermoelastic_compliance_and_gradient_density_multimaterial(
            x, temperature, sol, to_params,
            fe_solver_thermal, fe_solver_structural,
            EDesign, alphaDesign, KDesign,
            KETemplate, KTTemplate
        )
        _, dJdz, _ = compute_thermoelastic_compliance_and_gradient_latent_multimaterial(
            zeta, temperature, sol, to_params,
            fe_solver_thermal, fe_solver_structural,
            EDesign, alphaDesign, KDesign,
            KETemplate, KTTemplate,
            dE_dz, dAlpha_dz, dK_dz,material_model=material_model,
            lambda_T=lambda_T
        )    
        grad_obj = np.concatenate([dJdx, dJdz])
        print("Min of dj/dx:", np.min(np.abs(dJdx)), " Max of dj/dx:", np.max(np.abs(dJdx)))
        print("Min of dj/dz:", np.min(np.abs(dJdz)), " Max of dj/dz:", np.max(np.abs(dJdz)))
        return J, grad_obj

    elif objectiveType == TO_QOI.PNORM_STRESS:
        vm_pnorm, grad_vm_density, max_vm = compute_pnorm_stress_and_sensitivity(
            sol, x, fe_solver_structural, EDesign, KETemplate, MaterialModel.SIMP)
        sigma_vm = fe_solver_structural.vonMisesStress

        outer = (np.sum(sigma_vm ** pNormExponent)) ** (1.0 / pNormExponent - 1)
        d_sigma_vm_dE = sigma_vm / EDesign
        grad_vm_z = np.zeros(latentDim * num_elems)
        for d in range(latentDim):
            grad_vm_z[d * num_elems:(d + 1) * num_elems] = (
                pNormExponent * (sigma_vm ** (pNormExponent - 1)) * (d_sigma_vm_dE * dE_dz[:, d])
            )
        grad_vm_z = (1.0 / pNormExponent) * outer * grad_vm_z
        grad_pnorm_stress = np.zeros_like(zeta)
        grad_pnorm_stress[0:num_elems] = grad_vm_density
        grad_pnorm_stress[num_elems:] = grad_vm_z
        return vm_pnorm, grad_pnorm_stress

    elif objectiveType == TO_QOI.MASS:
        elemVolume = fe_solver_structural.mesh.elem_size[0] * fe_solver_structural.mesh.elem_size[1] * fe_solver_structural.mesh.elem_size[2]
        totalMass = np.sum(mass_density * x) * elemVolume if mass_density is not None else 0.0
        grad_mass = np.concatenate([
            mass_density * elemVolume if mass_density is not None else np.zeros(num_elems),
            (dMassDensity_dz * x * elemVolume).flatten() if dMassDensity_dz is not None else np.zeros(num_elems * latentDim)
        ])
        return totalMass, grad_mass

    elif objectiveType == TO_QOI.COST:
        elemVolume = fe_solver_structural.mesh.elem_size[0] * fe_solver_structural.mesh.elem_size[1] * fe_solver_structural.mesh.elem_size[2]
        totalCost = np.sum(mass_density * cost_per_unitmass * x) * elemVolume if (mass_density is not None and cost_per_unitmass is not None) else 0.0
        dTotalCost_dz = (dMassDensity_dz * cost_per_unitmass + mass_density * dCost_dz) * x * elemVolume if (dMassDensity_dz is not None and cost_per_unitmass is not None and mass_density is not None and dCost_dz is not None) else np.zeros((num_elems, latentDim))
        dTotalCost_dx = mass_density * cost_per_unitmass * elemVolume if (mass_density is not None and cost_per_unitmass is not None) else np.zeros(num_elems)
        grad_cost = np.concatenate([dTotalCost_dx, dTotalCost_dz.flatten()])
        return totalCost, grad_cost

    elif objectiveType == TO_QOI.MAX_CRITICALITY:
        if criticality is not None:
            pnorm_criticality = np.sum(criticality ** pNormExponent) ** (1.0 / pNormExponent)
            dpnorm_dcriticality = (criticality ** (pNormExponent - 1)) / (pnorm_criticality ** (pNormExponent - 1))
            dpnorm_dz = dCriticality_dz * dpnorm_dcriticality[np.newaxis, :]  # Broadcasting
            grad_crit_z = dpnorm_dz.flatten()
            grad_obj = np.zeros_like(zeta)
            grad_obj[num_elems:] = grad_crit_z
            return pnorm_criticality, grad_obj
        else:
            return 0.0, np.zeros_like(zeta)

    elif objectiveType == TO_QOI.MEAN_CRITICALITY:
        if criticality is not None:
            mean_criticality = np.mean(criticality)
            grad_obj = np.zeros_like(zeta)
            grad_obj[num_elems:] = dCriticality_dz.flatten() / len(criticality)
            return mean_criticality, grad_obj
        else:
            return 0.0, np.zeros_like(zeta)

    else:
        raise NotImplementedError(f"Objective {objectiveType} is not implemented yet.")
    
   
def compute_mmto_constraint_and_gradient(to_params, sol, zeta, fe_solver, KETemplate, matEncoder):
    """
    Compute constraint values and their gradients for METALS LSR.
    Handles:
      1. Compliance constraint
      2. Mass constraint
      3. Cost constraint
      4. Yield strength/safety factor constraint
    """
    nConstraints = len(to_params.Constraints)
    num_elems = fe_solver.mesh.num_elems
    x = zeta[0:num_elems]
    latentDim = matEncoder.vae_params.latentDim

    zPts = zeta[num_elems:].reshape((latentDim, -1)).T
    material_properties, gradients = matEncoder.getMaterialPropertiesAtLatentPoints(zPts, compute_gradients=True)
    material_model = to_params.materialModel

    if 'Youngs_Modulus' in material_properties:
        EDesign = material_properties['Youngs_Modulus'].detach().numpy()
        dE_dz = gradients['Youngs_Modulus'].detach().numpy().T
    else:
        EDesign = None
        dE_dz = None

    if 'Density' in material_properties:
        mass_density = material_properties['Density'].detach().numpy()
        dMassDensity_dz = gradients['Density'].detach().numpy().T
    else:
        mass_density = None
        dMassDensity_dz = None

    if 'Cost' in material_properties:
        cost_per_unitmass = material_properties['Cost'].detach().numpy()
        dCost_dz = gradients['Cost'].detach().numpy().T
    else:
        cost_per_unitmass = None
        dCost_dz = None

    if 'Criticality' in material_properties:
        criticality = material_properties['Criticality'].detach().numpy()
        dCriticality_dz = gradients['Criticality'].detach().numpy().T
    else:
        criticality = None
        dCriticality_dz = None

    if 'Yield_Strength' in material_properties:
        YDesign = material_properties['Yield_Strength'].detach().numpy()
        dY_dz = gradients['Yield_Strength'].detach().numpy().T
    else:
        YDesign = None
        dY_dz = None

    
    pNormExponent = get_pNorm_exponent()

    c = np.zeros((nConstraints, 1))
    num_design_var = zeta.size
    dc = np.zeros((nConstraints, num_design_var))
    for m in range(nConstraints):
        constraintType = to_params.Constraints[m][0]
        optionalParam = to_params.Constraints[m][1]
        constraintLimit = to_params.Constraints[m][2]
        if constraintType == TO_QOI.COMPLIANCE:
            compliance = np.einsum('i, i -> ', fe_solver.total_force, sol)
            ce = (np.dot(sol[fe_solver.mesh.edofMatStructural].reshape(num_elems, 24), KETemplate) * sol[fe_solver.mesh.edofMatStructural].reshape(num_elems, 24)).sum(1)
            dJ_dxDesign = (-get_structural_material_model_sensitivity(x, material_model)) * EDesign * ce
            dJ_dEDesign = np.asarray((get_structural_material_model_scaling(x)) * ce)
            dJ_dz = (dJ_dEDesign * dE_dz).flatten()
            grad_compliance = np.concatenate((dJ_dxDesign, -dJ_dz))
            c[m, 0] = compliance/constraintLimit-1
            dc[m, :] = grad_compliance / constraintLimit
        elif constraintType == TO_QOI.VOLUME_FRACTION:
            volfracConstraint, volfracConstraint_gradient = compute_volumefraction_constraint_and_gradient(
                x, constraintLimit)
            grad_volfracConstraint = np.zeros_like(zeta)
            grad_volfracConstraint[0:num_elems] = volfracConstraint_gradient
            c[m, 0] = volfracConstraint
            dc[m, :] = grad_volfracConstraint
        elif constraintType == TO_QOI.MASS:
            elemVolume =  fe_solver.mesh.elem_size[0] * fe_solver.mesh.elem_size[1] * fe_solver.mesh.elem_size[2]
            totalMass =  (mass_density * x).sum()*elemVolume 
            grad_mass = np.concatenate((mass_density*elemVolume, (dMassDensity_dz*x*elemVolume).flatten()))
            c[m, 0] = totalMass/constraintLimit - 1.0
            dc[m, :] = grad_mass / constraintLimit
        elif constraintType == TO_QOI.COST:
            elemVolume =  fe_solver.mesh.elem_size[0] * fe_solver.mesh.elem_size[1] * fe_solver.mesh.elem_size[2]
            totalCost =  (mass_density* cost_per_unitmass * x).sum()*elemVolume 
            dTotalCost_dz = (dMassDensity_dz * cost_per_unitmass + mass_density * dCost_dz)*x*elemVolume
            dTotalCost_dx = mass_density * cost_per_unitmass * elemVolume
            grad_cost = np.concatenate((dTotalCost_dx, (dTotalCost_dz).flatten()))
            c[m, 0] = totalCost/constraintLimit - 1.0
            dc[m, :] = grad_cost / constraintLimit
        elif constraintType == TO_QOI.MAX_CRITICALITY:
            pnorm_criticality = np.sum(criticality ** pNormExponent) ** (1.0 / pNormExponent)
            maxCriticality = pnorm_criticality
            dpnorm_dcriticality = (criticality ** (pNormExponent - 1)) / (pnorm_criticality ** (pNormExponent - 1))
            dpnorm_dz = dCriticality_dz * dpnorm_dcriticality[np.newaxis, :]  # Broadcasting
            grad_crit_z = dpnorm_dz.flatten()
            grad_cons_criticality = np.zeros_like(zeta)
            grad_cons_criticality[num_elems:] = grad_crit_z / constraintLimit
            c[m, 0] = ((maxCriticality / constraintLimit) - 1.0)
            dc[m, :] = grad_cons_criticality
        elif constraintType == TO_QOI.MEAN_CRITICALITY:
            # Compute mean criticality
            mean_criticality = np.mean(criticality)
            grad_cons_mean_criticality = np.zeros_like(zeta)
            grad_cons_mean_criticality[num_elems:] = dCriticality_dz.flatten() / constraintLimit/len(criticality)
            c[m, 0] = ((mean_criticality / constraintLimit) - 1.0)
            dc[m, :] = grad_cons_mean_criticality
        elif constraintType == TO_QOI.STRESS_FAILURE_FACTOR:
            ff_pNorm, grad_ff_density,stress_ff_max = compute_pnorm_safety_factor_and_sensitivity(
                sol, x, fe_solver,EDesign,YDesign, KETemplate, MaterialModel.SIMP)
            safety_constraint = stress_ff_max/constraintLimit - 1.0
            c[m, 0] = safety_constraint
            grad_stress_ff = np.zeros_like(zeta)
            grad_stress_ff[:num_elems] = grad_ff_density/constraintLimit
            
            # 2. Compute latent variable part of gradient (manual approach using dY_dz)
            sigma_vm = fe_solver.vonMisesStress
            P = pNormExponent

            # Compute p-norm
            stress_ff_np = sigma_vm / YDesign
            sum_term = np.sum(stress_ff_np ** P)

            # d(ff_pNorm)/dY for each element
            outer = sum_term ** ((1.0 / P) - 1)
            d_ff_pNorm_dY = outer * (stress_ff_np ** (P - 1)) * (-sigma_vm / (YDesign ** 2))
            # Shape: (num_elems,)

            # dY_dz is shape (latentDim, num_elems)
            d_ff_pNorm_dz = (d_ff_pNorm_dY[:, np.newaxis] * dY_dz.T).flatten(order='F')
            # Shape: (num_elems * latentDim,)

            grad_stress_ff[num_elems:] = d_ff_pNorm_dz / constraintLimit
            dc[m, :] = grad_stress_ff
        else:
            raise NotImplementedError(f"Constraint {constraintType} is not implemented yet.")

    return c, dc