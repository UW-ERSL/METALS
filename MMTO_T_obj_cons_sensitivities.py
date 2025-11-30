from unicodedata import name
import numpy as np
from PyTOImports import *

# --- Support Functions ---
from MMTO_obj_cons_sensitivitiesOld import (
        compute_pnorm_safety_factor_and_sensitivity, 
        compute_pnorm_stress_and_sensitivity,
        compute_volumefraction_constraint_and_gradient
)

# --- Main Objective/Constraint Functions ---

def compute_mmto_objective_and_gradient(to_params, uvw, Temp, zeta, fe_solver_structural, KETemplate, matEncoder):
    """
    Compute objective value and its gradient for METALS LSR.
    Handles:
      1. Compliance objective (subject to mass)
      2. Compliance objective (subject to mass and cost)
      3. Mass objective (subject to yield strength and compliance)
    """
    objectiveType = to_params.Objective[0]
    optionalParam = to_params.Objective[1]
    num_elems = fe_solver_structural.mesh.num_elems
    x = zeta[0:num_elems]

    latentDim = matEncoder.vae_params.latentDim
    zPts = zeta[num_elems:].reshape((latentDim, -1)).T
    pSIMP = SIMP_STRUCTURAL_PENALTY
    pNormExponent = PNORM_EXPONENT
    if objectiveType == TO_QOI.COMPLIANCE:
        compliance = np.einsum('i, i -> ', fe_solver_structural.total_force, uvw)
        E, dE_dz = matEncoder.getValueOfAttributeAtZLocationAtTemperature("E",zPts, Temp, compute_gradients = True)
        ce = (np.dot(uvw[fe_solver_structural.mesh.edofMatStructural].reshape(num_elems, 24), KETemplate) * uvw[fe_solver_structural.mesh.edofMatStructural].reshape(num_elems, 24)).sum(1)
        dJ_dxDesign = (-pSIMP * x ** (pSIMP - 1)) * E * ce
        dJ_dE = (x ** pSIMP) * ce
        dJ_dz =( dJ_dE*dE_dz).flatten()
        grad_compliance = np.concatenate((dJ_dxDesign, -dJ_dz))
        return compliance, grad_compliance

    elif objectiveType == TO_QOI.PNORM_STRESS:
        E, dE_dz = matEncoder.getValueOfAttributeAtZLocationAtTemperature("E",zPts, Temp, compute_gradients = True)
        pNorm_stress, grad_stress_density, stress_pnorm_max = compute_pnorm_stress_and_sensitivity(
            uvw, x, fe_solver_structural, E, KETemplate, MaterialModel.SIMP)
        sigma_vm = fe_solver_structural.vonMisesStress
    
        outer = (np.sum(sigma_vm ** pNormExponent)) ** (1.0 / pNormExponent - 1)
        d_sigma_vm_dE = sigma_vm / E
        dE_dz = dE_dz.reshape(num_elems, latentDim)
        grad_vm_z = np.zeros(latentDim*num_elems)
        for d in range(latentDim):
            grad_vm_z[d*num_elems:(d+1)*num_elems] = (pNormExponent * (sigma_vm ** (pNormExponent - 1))) * (d_sigma_vm_dE * dE_dz[:,d])
        grad_vm_z = (1.0 / pNormExponent) * outer * grad_vm_z
        grad_pnorm_stress = np.zeros_like(zeta)
        grad_pnorm_stress[0:num_elems] = grad_stress_density
        grad_pnorm_stress[num_elems:] = grad_vm_z

    elif objectiveType == TO_QOI.MASS:
        mass_density, dMassDensity_dz = matEncoder.getValueOfAttributeAtZLocationAtTemperature("Density",zPts, Temp, compute_gradients = True)
        elemVolume =  fe_solver_structural.mesh.elem_size[0] * fe_solver_structural.mesh.elem_size[1] * fe_solver_structural.mesh.elem_size[2]
        totalMass =  (mass_density * x).sum()*elemVolume 
        grad_mass = np.concatenate((mass_density*elemVolume, (dMassDensity_dz*x*elemVolume).flatten()))
        return totalMass, grad_mass
    else:
        raise NotImplementedError(f"Objective {objectiveType} is not implemented yet.")


def compute_mmto_constraint_and_gradient(to_params, uvw, Temp, zeta, fe_solver_structural, KETemplate, matEncoder):
    """
    Compute constraint values and their gradients for METALS LSR.
    Handles:
      1. Compliance constraint
      2. Mass constraint
      3. Cost constraint
      4. Yield strength/safety factor constraint
    """
    nConstraints = len(to_params.Constraints)
    num_elems = fe_solver_structural.mesh.num_elems
    x = zeta[0:num_elems]
    latentDim = matEncoder.vae_params.latentDim
    c = np.zeros((nConstraints, 1))
    num_design_var = zeta.size
    dc = np.zeros((nConstraints, num_design_var))

    zPts = zeta[num_elems:].reshape((latentDim, -1)).T
    pSIMP = SIMP_STRUCTURAL_PENALTY
    pNormExponent = PNORM_EXPONENT
    for m in range(nConstraints):
        constraintType = to_params.Constraints[m][0]
        optionalParam = to_params.Constraints[m][1]
        constraintLimit = to_params.Constraints[m][2]

        if constraintType == TO_QOI.COMPLIANCE:
            compliance = np.einsum('i, i -> ', fe_solver_structural.total_force, uvw)
            E, dE_dz = matEncoder.getValueOfAttributeAtZLocationAtTemperature("E",zPts, Temp, compute_gradients = True)
            ce = (np.dot(uvw[fe_solver_structural.mesh.edofMatStructural].reshape(num_elems, 24), KETemplate) * uvw[fe_solver_structural.mesh.edofMatStructural].reshape(num_elems, 24)).sum(1)
            dJ_dxDesign = (-pSIMP * x ** (pSIMP - 1)) * E * ce
            dJ_dE = (x ** pSIMP) * ce
            dJ_dz =( dJ_dE*dE_dz).flatten()
            grad_compliance = np.concatenate((dJ_dxDesign, -dJ_dz))
            c[m, 0] = compliance/constraintLimit - 1.0
            dc[m, :] = grad_compliance / constraintLimit
        elif constraintType == TO_QOI.VOLUME_FRACTION:
            volfracConstraint, volfracConstraint_gradient = compute_volumefraction_constraint_and_gradient(
                x, constraintLimit)
            grad_volfracConstraint = np.zeros_like(zeta)
            grad_volfracConstraint[0:num_elems] = volfracConstraint_gradient
            c[m, 0] = volfracConstraint
            dc[m, :] = grad_volfracConstraint

        elif constraintType == TO_QOI.MASS:
            mass_density, dMassDensity_dz = matEncoder.getValueOfAttributeAtZLocationAtTemperature("Density",zPts, Temp, compute_gradients = True)
            elemVolume =  fe_solver_structural.mesh.elem_size[0] * fe_solver_structural.mesh.elem_size[1] * fe_solver_structural.mesh.elem_size[2]
            totalMass =  (mass_density * x).sum()*elemVolume 
            grad_mass = np.concatenate((mass_density*elemVolume, (dMassDensity_dz*x*elemVolume).flatten()))
            c[m, 0] = totalMass/constraintLimit - 1.0
            dc[m, :] = grad_mass / constraintLimit

        elif constraintType == TO_QOI.STRESS_FAILURE_FACTOR:
            #print("TMax = ", T.max())
            E, dE_dz = matEncoder.getValueOfAttributeAtZLocationAtTemperature("E",zPts, Temp, compute_gradients = True)
            Y, dY_dz = matEncoder.getValueOfAttributeAtZLocationAtTemperature("Y",zPts, Temp, compute_gradients = True)
            
            stress_ffpnorm, grad_stress_ffdensity, stress_ff_max = compute_pnorm_safety_factor_and_sensitivity(
                uvw, x, fe_solver_structural, E, Y, KETemplate, MaterialModel.SIMP)
  
            c[m, 0] = stress_ff_max/constraintLimit - 1.0
            grad_stress_ff = np.zeros_like(zeta)
            grad_stress_ff[:num_elems] = grad_stress_ffdensity/constraintLimit
            
            # 2. Compute latent variable part of gradient (manual approach using dY_dz)
            sigma_vm = fe_solver_structural.vonMisesStress
            P = pNormExponent

            # Compute p-norm
            stress_ff_np = sigma_vm / Y
            sum_term = np.sum(stress_ff_np ** P)

            # d(ff_pNorm)/dY for each element
            outer = sum_term ** ((1.0 / P) - 1)
            d_ff_pNorm_dY = outer * (stress_ff_np ** (P - 1)) * (-sigma_vm / (Y ** 2))
            # Shape: (num_elems,)

            # dY_dz is shape (latentDim, num_elems)
            d_ff_pNorm_dz = (d_ff_pNorm_dY[:, np.newaxis] * dY_dz.T).flatten(order='F')
            # Shape: (num_elems * latentDim,)

            grad_stress_ff[num_elems:] = d_ff_pNorm_dz / constraintLimit
            dc[m, :] = grad_stress_ff
        elif constraintType == TO_QOI.COST:
            mass_density, dMassDensity_dz = matEncoder.getValueOfAttributeAtZLocationAtTemperature("Density",zPts, Temp, compute_gradients = True)
            costperunitmass, dCost_dz = matEncoder.getValueOfAttributeAtZLocationAtTemperature("Cost",zPts, Temp, compute_gradients = True)

            x = zeta[0:fe_solver_structural.mesh.num_elems]
            elemVolume = fe_solver_structural.mesh.elem_size[0] * fe_solver_structural.mesh.elem_size[1] * fe_solver_structural.mesh.elem_size[2]
            totalCost = np.einsum('m,m,m->m', mass_density, costperunitmass, x).sum() * elemVolume
            grad_cons_cost = (dMassDensity_dz * costperunitmass + mass_density * dCost_dz) * x * elemVolume    
            c[m, 0] = (totalCost / constraintLimit - 1.0)
            dc[m, :] = grad_cons_cost/constraintLimit

        elif constraintType == TO_QOI.MEAN_CRITICALITY:
            criticality, dCriticality_dz = matEncoder.getValueOfAttributeAtZLocationAtTemperature("Criticality",zPts, Temp, compute_gradients = True)
            mean_criticality = np.mean(criticality)
            grad_cons_mean_criticality = np.zeros_like(zeta)
            grad_cons_mean_criticality[num_elems:] = dCriticality_dz.flatten() / constraintLimit/len(criticality)
            c[m, 0] = ((mean_criticality / constraintLimit) - 1.0)
            dc[m, :] = grad_cons_mean_criticality
        elif constraintType == TO_QOI.TEMPERATURE_FAILURE_FACTOR:
            temp_limit, dTempLimit_dz = matEncoder.getValueOfAttributeAtZLocationAtTemperature("Temp_Limit",zPts, Temp, compute_gradients = True)
        
            # Compute p-norm of temperature failure factor
            temp_ff = Temp / temp_limit
            temp_ff_pnorm = np.sum(temp_ff ** pNormExponent) ** (1.0 / pNormExponent)
            temp_ff_max = temp_ff_pnorm
            c[m, 0] = temp_ff_max / constraintLimit - 1.0

            # Compute gradient with respect to temp_limit
            outer = np.sum(temp_ff ** pNormExponent) ** ((1.0 / pNormExponent) - 1)
            d_temp_ff_pNorm_dTempLimit = outer * (temp_ff ** (pNormExponent - 1)) * (-Temp / (temp_limit ** 2))

            # Compute gradient with respect to latent variables
            d_temp_ff_pNorm_dz = (d_temp_ff_pNorm_dTempLimit[:, np.newaxis] * dTempLimit_dz.T).flatten(order='F')

            grad_temp_ff = np.zeros_like(zeta)
            grad_temp_ff[num_elems:] = d_temp_ff_pNorm_dz / constraintLimit
            dc[m, :] = grad_temp_ff
        elif constraintType == TO_QOI.PBR:
            pbr, dPbr_dz = matEncoder.getValueOfAttributeAtZLocationAtTemperature("PBR",zPts, Temp, compute_gradients = True)
            mean_pbr = np.mean(pbr)
            grad_cons_pbr = np.zeros_like(zeta)
            grad_cons_pbr[num_elems:] = dPbr_dz.flatten() / constraintLimit/len(pbr)
            c[m, 0] = ((mean_pbr / constraintLimit) - 1.0)
            dc[m, :] = grad_cons_pbr
        elif constraintType == TO_QOI.FATIGUE_FAILURE_FACTOR:
            E, dE_dz = matEncoder.getValueOfAttributeAtZLocationAtTemperature("E",zPts, Temp, compute_gradients = True)
            FatigueLimit, dFatigueLimit_dz = matEncoder.getValueOfAttributeAtZLocationAtTemperature("Fatigue_Limit",zPts, Temp, compute_gradients = True)
            
            stress_ffpnorm, grad_stress_ffdensity, stress_ff_max = compute_pnorm_safety_factor_and_sensitivity(
                uvw, x, fe_solver_structural, E, FatigueLimit, KETemplate, MaterialModel.SIMP)
  
            c[m, 0] = stress_ff_max/constraintLimit - 1.0
            grad_stress_ff = np.zeros_like(zeta)
            grad_stress_ff[:num_elems] = grad_stress_ffdensity/constraintLimit 

            # 2. Compute latent variable part of gradient (manual approach using dFatigueLimit_dz)
            sigma_vm = fe_solver_structural.vonMisesStress
            P = pNormExponent   
            # Compute p-norm
            stress_ff_np = sigma_vm / FatigueLimit
            sum_term = np.sum(stress_ff_np ** P)
            # d(ff_pNorm)/dFatigueLimit for each element
            outer = sum_term ** ((1.0 / P) - 1)
            d_ff_pNorm_dFatigueLimit = outer * (stress_ff_np ** (P - 1)) * (-sigma_vm / (FatigueLimit ** 2))
            # Shape: (num_elems,)
            # dFatigueLimit_dz is shape (latentDim, num_elems)
            d_ff_pNorm_dz = (d_ff_pNorm_dFatigueLimit[:, np.newaxis] * dFatigueLimit_dz.T).flatten(order='F')
            # Shape: (num_elems * latentDim,)
            grad_stress_ff[num_elems:] = d_ff_pNorm_dz / constraintLimit
            dc[m, :] = grad_stress_ff

        else:
            raise NotImplementedError(f"Constraint {constraintType} is not implemented yet.")

    return c, dc