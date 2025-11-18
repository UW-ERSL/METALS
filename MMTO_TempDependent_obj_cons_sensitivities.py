from unicodedata import name
import numpy as np
import torch
from PyTOImports import *
from InterpolationFunctions import hermiteInterpolation, hermiteInterpolation_torch
# --- Support Functions ---
from MMTO_obj_cons_sensitivities import compute_pnorm_safety_factor_and_sensitivity, compute_pnorm_stress_and_sensitivity, compute_volumefraction_constraint_and_gradient

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
    zetaTensor = torch.tensor(zeta).float()
    zetaTensor.requires_grad = True

    latentDim = matEncoder.vae_params.latentDim

    if objectiveType == TO_QOI.COMPLIANCE:
        ETensor = matEncoder.getMaterialPropertyAtTemperatureTorch("E", zetaTensor[num_elems:].view(latentDim, -1).T, Temp)
        compliance = np.einsum('i, i -> ', fe_solver_structural.total_force, uvw)
        ce = (np.dot(uvw[fe_solver_structural.mesh.edofMat].reshape(num_elems, 24), KETemplate) * uvw[fe_solver_structural.mesh.edofMat].reshape(num_elems, 24)).sum(1)
        pSIMP = 3.0
        dJ_dxDesign = (-pSIMP * x ** (pSIMP - 1)) * ETensor.detach().numpy() * ce
        dJ_dE = torch.tensor((x ** pSIMP) * ce)
        zetaTensor.grad = None
        ETensor.backward(dJ_dE, retain_graph=True)
        dJ_dzDesign = zetaTensor.grad[num_elems:].detach().numpy()
        grad_compliance = np.concatenate((dJ_dxDesign, -dJ_dzDesign.flatten()))
        return compliance, grad_compliance

    elif objectiveType == TO_QOI.VOLUME_FRACTION:
        pass

    elif objectiveType == TO_QOI.PNORM_STRESS:
        ETensor = matEncoder.getMaterialPropertyAtTemperatureTorch("E", zetaTensor[num_elems:].view(latentDim, -1).T, Temp)
        EDesign = ETensor.detach().numpy()
        vm_pnorm, grad_vm_density, max_vm = compute_pnorm_stress_and_sensitivity(
            uvw, x, fe_solver_structural, EDesign, KETemplate, MaterialModel.SIMP)

        sigma_vm = fe_solver_structural.vonMisesStress
        d_sigma_vm_dE = sigma_vm / EDesign
        pNormExponent = 6
        outer = (np.sum(sigma_vm ** pNormExponent)) ** (1.0 / pNormExponent - 1)

        # Backward for dE/dz 
        zetaTensor.grad = None
        ETensor.backward(torch.ones_like(ETensor), retain_graph=True)
        dE_dz = zetaTensor.grad[num_elems:].detach().numpy().reshape(num_elems, latentDim)

        grad_vm_z = np.zeros(latentDim * num_elems)
        for d in range(latentDim):
            d_sigma_dz = d_sigma_vm_dE * dE_dz[:, d]
            grad_vm_z[d*num_elems:(d+1)*num_elems] = (pNormExponent * (sigma_vm ** (pNormExponent - 1))) * d_sigma_dz

        grad_vm_z = (1.0 / pNormExponent) * outer * grad_vm_z
        grad_pnorm_stress = np.zeros_like(zeta)
        grad_pnorm_stress[0:num_elems] = grad_vm_density
        grad_pnorm_stress[num_elems:] = grad_vm_z / max_vm / pNormExponent  # KS: Check this scaling

        return vm_pnorm, grad_pnorm_stress

    elif objectiveType == TO_QOI.MASS:
        decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(latentDim, -1).T)
        mass_density = matEncoder.getMaterialProperties(decoded)['Density']
        pseudodensity = zetaTensor[0:num_elems]
        elemVolume = fe_solver_structural.mesh.elem_size[0] * fe_solver_structural.mesh.elem_size[1] * fe_solver_structural.mesh.elem_size[2]
        totalMass = torch.einsum('m,m->m', mass_density, pseudodensity).sum() * elemVolume
        totalMass.backward(retain_graph=True)
        obj_mass = totalMass.detach().numpy()
        grad_mass = zetaTensor.grad.detach().numpy()
        return obj_mass, grad_mass
    else:
        raise NotImplementedError(f"Objective {objectiveType} is not implemented yet.")


def compute_mmto_constraint_and_gradient(to_params, uvw, T, zeta, fe_solver_structural, KETemplate, matEncoder):
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
    zetaTensor = torch.tensor(zeta).float()
    zetaTensor.requires_grad = True

    latentDim = matEncoder.vae_params.latentDim

    decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(latentDim, -1).T)
    c = np.zeros((nConstraints, 1))
    num_design_var = zeta.size
    dc = np.zeros((nConstraints, num_design_var))
    for m in range(nConstraints):
        constraintType = to_params.Constraints[m][0]
        optionalParam = to_params.Constraints[m][1]
        constraintLimit = to_params.Constraints[m][2]

        if constraintType == TO_QOI.COMPLIANCE:
            ETensor = matEncoder.getMaterialPropertyAtTemperatureTorch("E", zetaTensor[num_elems:].view(latentDim, -1).T, T)
            compliance = np.einsum('i, i -> ', fe_solver_structural.total_force, uvw)
            complianceConstraint = (compliance / constraintLimit) - 1.0
            ce = (np.dot(uvw[fe_solver_structural.mesh.edofMat].reshape(num_elems, 24), KETemplate) * uvw[fe_solver_structural.mesh.edofMat].reshape(num_elems, 24)).sum(1)
            pSIMP = 3.0
            dJ_dxDesign = (-pSIMP * x ** (pSIMP - 1)) * ETensor.detach().numpy() * ce
            dJ_dE = torch.tensor((x ** pSIMP) * ce)
            zetaTensor.grad = None
            ETensor.backward(dJ_dE, retain_graph=True)
            dJ_dzDesign = zetaTensor.grad[num_elems:].detach().numpy()
            grad_compliance = np.concatenate((dJ_dxDesign, -dJ_dzDesign.flatten()))

            cons_compliance = complianceConstraint
            grad_cons_compliance = grad_compliance / constraintLimit
            c[m, 0] = cons_compliance
            dc[m, :] = grad_cons_compliance
        elif constraintType == TO_QOI.VOLUME_FRACTION:
            volfracConstraint, volfracConstraint_gradient = compute_volumefraction_constraint_and_gradient(
                x, constraintLimit)
            grad_volfracConstraint = np.zeros_like(zeta)
            grad_volfracConstraint[0:num_elems] = volfracConstraint_gradient
            c[m, 0] = volfracConstraint
            dc[m, :] = grad_volfracConstraint

        elif constraintType == TO_QOI.MASS:
            mass_density = matEncoder.getMaterialProperties(decoded)['Density']
            pseudodensity = zetaTensor[0:num_elems]
            elemVolume = fe_solver_structural.mesh.elem_size[0] * fe_solver_structural.mesh.elem_size[1] * fe_solver_structural.mesh.elem_size[2]
            totalMass = torch.einsum('m,m->m', mass_density, pseudodensity).sum() * elemVolume
            massConstraint = ((totalMass / constraintLimit) - 1.0)
            zetaTensor.grad = None
            massConstraint.backward(retain_graph=True)
            cons_mass = massConstraint.detach().numpy()
            grad_cons_mass = zetaTensor.grad.detach().numpy()
            c[m, 0] = cons_mass
            dc[m, :] = grad_cons_mass

        elif constraintType == TO_QOI.STRESS_SAFETY_FACTOR:
            ETensor = matEncoder.getMaterialPropertyAtTemperatureTorch("E", zetaTensor[num_elems:].view(latentDim, -1).T, T)
            YTensor = matEncoder.getMaterialPropertyAtTemperatureTorch("Y", zetaTensor[num_elems:].view(latentDim, -1).T, T)
            EDesign = ETensor.detach().numpy()
            YDesign = YTensor.detach().numpy()

            inv_sf_pnorm, grad_inv_sf_density, _ = compute_pnorm_safety_factor_and_sensitivity(
                uvw, x, fe_solver_structural, EDesign, YDesign, KETemplate, MaterialModel.SIMP)
            safety_factor = constraintLimit
            safety_constraint = inv_sf_pnorm - (1.0 / safety_factor)
            c[m, 0] = safety_constraint

            # 2. Compute latent variable part of gradient (chain rule)
            pNormExponent = 6
            sigma_vm = fe_solver_structural.vonMisesStress
            d_sigma_vm_dE = sigma_vm / EDesign
            Y = YDesign
            S = sigma_vm
            inv_sf_elem = S / Y
            outer = (np.sum(inv_sf_elem ** pNormExponent)) ** (1.0 / pNormExponent - 1)
            grad_inv_sf_z = np.zeros(latentDim * num_elems)
            # Backward for dE/dz and dY/dz
            zetaTensor.grad = None
            ETensor.backward(torch.ones_like(ETensor), retain_graph=True)
            dE_dz = zetaTensor.grad[num_elems:].detach().numpy().reshape(num_elems, latentDim)
            zetaTensor.grad = None
            YTensor.backward(torch.ones_like(YTensor), retain_graph=True)
            dY_dz = zetaTensor.grad[num_elems:].detach().numpy().reshape(num_elems, latentDim)
            zetaTensor.grad = None
            for d in range(latentDim):
                d_sigma_dz = d_sigma_vm_dE * dE_dz[:, d]
                d_inv_sf_dz = (d_sigma_dz * Y - dY_dz[:, d] * sigma_vm) / (Y ** 2)
                grad_inv_sf_z[d*num_elems:(d+1)*num_elems] = (pNormExponent * (inv_sf_elem ** (pNormExponent - 1))) * d_inv_sf_dz
            grad_inv_sf_z = (1.0 / pNormExponent) * outer * grad_inv_sf_z

            grad_inv_safety = np.zeros_like(zeta)
            grad_inv_safety[:num_elems] = grad_inv_sf_density
            grad_inv_safety[num_elems:] = grad_inv_sf_z
            dc[m, :] = grad_inv_safety

        elif constraintType == TO_QOI.COST:
            mass_density = matEncoder.getMaterialProperties(decoded)['Density']
            costperunitmass = matEncoder.getMaterialProperties(decoded)['Cost']
            pseudodensity = zetaTensor[0:fe_solver_structural.mesh.num_elems]
            elemVolume = fe_solver_structural.mesh.elem_size[0] * fe_solver_structural.mesh.elem_size[1] * fe_solver_structural.mesh.elem_size[2]
            totalCost = torch.einsum('m,m,m->m', mass_density, costperunitmass, pseudodensity).sum() * elemVolume
            costConstraint = ((totalCost / constraintLimit) - 1.0)
            zetaTensor.grad = None
            costConstraint.backward(retain_graph=True)
            cons_cost = costConstraint.detach().numpy()
            grad_cons_cost = zetaTensor.grad.detach().numpy()
            c[m, 0] = cons_cost
            dc[m, :] = grad_cons_cost

        elif constraintType == TO_QOI.MAX_CRITICALITY:
            criticality = matEncoder.getMaterialProperties(decoded)['Criticality']
            maxCriticality = torch.max(criticality)
            criticalityConstraint = ((maxCriticality / constraintLimit) - 1.0)
            zetaTensor.grad = None
            criticalityConstraint.backward(retain_graph=True)
            cons_criticality = criticalityConstraint.detach().numpy()
            grad_cons_criticality = zetaTensor.grad.detach().numpy()
            c[m, 0] = cons_criticality
            dc[m, :] = grad_cons_criticality

        elif constraintType == TO_QOI.MEAN_CRITICALITY:
            decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(latentDim, -1).T)
            criticality = matEncoder.getMaterialProperties(decoded)['Criticality']
            meanCriticality = torch.mean(criticality)
            criticalityConstraint = ((meanCriticality / constraintLimit) - 1.0)
            zetaTensor.grad = None
            criticalityConstraint.backward(retain_graph=True)
            cons_criticality = criticalityConstraint.detach().numpy()
            grad_cons_criticality = zetaTensor.grad.detach().numpy()
            c[m, 0] = cons_criticality
            dc[m, :] = grad_cons_criticality
        elif constraintType == TO_QOI.TEMPERATURE_SAFETY_FACTOR:
            TempLimit = matEncoder.getMaterialProperties(decoded)['Temp_Limit']
            pseudodensity = zetaTensor[0:num_elems]
            T_tensor = torch.tensor(T).float()
            inv_T_SF = pseudodensity*T_tensor / TempLimit
            # temp_constraint_elem = pseudodensity * TempLimit / T_tensor
            # temp_safety_factor = constraintLimit
            safety_constraint = inv_T_SF - (1.0 / constraintLimit)
            max_safety_constraint = torch.max(safety_constraint)
            zetaTensor.grad = None
            max_safety_constraint.backward(retain_graph=True)
            c[m, 0] = max_safety_constraint.detach().numpy()
            dc[m, :] = zetaTensor.grad.detach().numpy()
        else:
            raise NotImplementedError(f"Constraint {constraintType} is not implemented yet.")

    return c, dc