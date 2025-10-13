from unicodedata import name
import numpy as np
import torch
from PyTOImports import *
from InterpolationFunctions import hermiteInterpolation, hermiteInterpolation_torch
# --- Support Functions ---
from MMTO_obj_cons_sensitivities import compute_pnorm_safety_factor_and_sensitivity, d_relaxed_von_mises_dE, MaterialModel

# --- Main Objective/Constraint Functions ---

def compute_mmto_objective_and_gradient(to_params, uvw, T, zeta, fe_solver_structural, KETemplate, matEncoder):
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


    if objectiveType == TO_QOI.COMPLIANCE:
        ETensor = matEncoder.getMaterialPropertyAtTemperatureTorch("E", zetaTensor[num_elems:].view(2, -1).T, T)
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
       pass

    elif objectiveType == TO_QOI.MASS:
        decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(2,-1).T)
        mass_density = matEncoder.getMaterialProperties(decoded)['Density']
        #print("Mass density min:", mass_density.min().item(), "max:", mass_density.max().item())
        pseudodensity = zetaTensor[0:num_elems]
        elemVolume =  fe_solver_structural.mesh.elem_size[0] * fe_solver_structural.mesh.elem_size[1] * fe_solver_structural.mesh.elem_size[2]
        totalMass = torch.einsum('m,m->m', mass_density, pseudodensity).sum()*elemVolume 
        totalMass.backward(retain_graph=True)
        obj_mass = totalMass.detach().numpy()
        grad_mass = zetaTensor.grad.detach().numpy()
        return obj_mass, grad_mass
    else:
        raise NotImplementedError(f"Objective {objectiveType} is not implemented yet.")
    
   
def compute_mmto_constraint_and_gradient(to_params, uvw,T, zeta, fe_solver_structural, KETemplate, matEncoder):
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

    c = np.zeros((nConstraints, 1))
    num_design_var = zeta.size
    dc = np.zeros((nConstraints, num_design_var))
    for m in range(nConstraints):
        constraintType = to_params.Constraints[m][0]
        optionalParam = to_params.Constraints[m][1]
        constraintLimit = to_params.Constraints[m][2]

        if constraintType == TO_QOI.COMPLIANCE:
            pass

        elif constraintType == TO_QOI.MASS:
            decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(2,-1).T)
            mass_density = matEncoder.getMaterialProperties(decoded)['Density']
            #print("Mass density min:", mass_density.min().item(), "max:", mass_density.max().item())
            pseudodensity = zetaTensor[0:num_elems]
            elemVolume =  fe_solver_structural.mesh.elem_size[0] * fe_solver_structural.mesh.elem_size[1] * fe_solver_structural.mesh.elem_size[2]
            totalMass = torch.einsum('m,m->m', mass_density, pseudodensity).sum()*elemVolume 
            massConstraint = ((totalMass / constraintLimit) - 1.0)
            massConstraint.backward(retain_graph=True)
            cons_mass = massConstraint.detach().numpy()
            grad_cons_mass = zetaTensor.grad.detach().numpy()
            c[m, 0] = cons_mass
            dc[m, :] = grad_cons_mass
        elif constraintType == TO_QOI.STRESS_SAFETY_FACTOR:
            # decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(2,-1).T)
            # material_properties = matEncoder.getMaterialProperties(decoded)
            # Ed = material_properties['Youngs_modulus_constant_term']
            # Ec = material_properties['Youngs_modulus_linear_term']
            # Eb= material_properties['Youngs_modulus_quadratic_term']
            # Ea= material_properties['Youngs_modulus_cubic_term']
            # T_torch = torch.tensor(T, dtype=Ea.dtype, device=Ea.device)
            # E0 = to_params.E0 if hasattr(to_params, 'E0') else 100
            # T0 = to_params.T0 if hasattr(to_params, 'T0') else 500
            # EDesigntensor = (
            #     Ea * T_torch**3 * E0 / T0**3 +
            #     Eb * T_torch**2 * E0 / T0**2 +
            #     Ec * T_torch * E0 / T0 +
            #     Ed * E0
            # )
            # Yd = material_properties['Yield_strength_constant_term']
            # Yc = material_properties['Yield_strength_linear_term']
            # Yb= material_properties['Yield_strength_quadratic_term']
            # Ya= material_properties['Yield_strength_cubic_term']
            # Y0 = to_params.Y0 if hasattr(to_params, 'Y0') else 100
            # T0 = to_params.T0 if hasattr(to_params, 'T0') else 500
            # YDesigntensor = (
            #     Ya * T_torch**3 * Y0 / T0**3 +
            #     Yb * T_torch**2 * Y0 / T0**2 +
            #     Yc * T_torch * Y0 / T0 +
            #     Yd * Y0
            # )
            # EDesign= EDesigntensor.detach().numpy()
            # YDesign= YDesigntensor.detach().numpy()
            # DensDesign = mass_density.detach().numpy()
            
            # vm_max = np.max(fe_solver_structural.vonMisesStress)
            # vm_min = np.min(fe_solver_structural.vonMisesStress)
            # print("Max von Mises stress:", vm_max)
            # print("Min von Mises stress:", vm_min)
            # inv_sf_pnorm, grad_inv_sf_density = compute_pnorm_safety_factor_and_sensitivity(
            #     uvw, x, fe_solver_structural,EDesign,YDesign, KETemplate, MaterialModel.SIMP,
            #     p=to_params.PNormExponent
            # )
            # print("P-norm of inv safety factor:", inv_sf_pnorm)
            # print("Grad inv sf density min:", grad_inv_sf_density.min(), "max:", grad_inv_sf_density.max())
            # # Safety factor constraint value
            # safety_factor = constraintLimit
            # safety_constraint = inv_sf_pnorm - (1.0 / safety_factor)
            # c[m, 0] = safety_constraint

        
            # # 2. Compute latent variable part of gradient (chain rule)
            # p = to_params.PNormExponent
            # d_sigma_vm_dE = np.zeros(num_elems)
            # for e in range(num_elems):
            #     # Divide by decoded youngs modulus for that element
            #     d_sigma_vm_dE[e] = d_relaxed_von_mises_dE(
            #         fe_solver_structural.stressComponents[e], x[e].item(), q=1) / EDesigntensor[e].item()
            # # Get per-element von Mises and yield strength
            # sigma_vm = np.zeros(num_elems)
            # for e in range(num_elems):
            #     stress = fe_solver_structural.stressComponents[e]
            #     sxx, syy, szz, syz, sxz, sxy = stress
            #     sigma_vm[e] = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) +
            #         3 * (syz ** 2 + sxz ** 2 + sxy ** 2)) * (zetaTensor[0:num_elems][e].item() ** 1)
            # Y = np.array([mat.yield_strength for mat in fe_solver_structural.mat_prop])
            # S = sigma_vm
            # inv_sf_elem = S / Y
            # sum_p = np.sum(inv_sf_elem ** p)
            # outer = (sum_p) ** (1.0 / p - 1)
            # grad_z = np.zeros(2*num_elems)
            # # Backward for dE/dz and dY/dz
            # zetaTensor.grad = None
            # EDesigntensor.backward(torch.ones_like(EDesigntensor), retain_graph=True)
            # dE_dz = zetaTensor.grad[num_elems:].detach().numpy().reshape(num_elems, -1)
            # zetaTensor.grad = None
            # YDesigntensor.backward(torch.ones_like(YDesigntensor), retain_graph=True)
            # dY_dz = zetaTensor.grad[num_elems:].detach().numpy().reshape(num_elems, -1)
            # zetaTensor.grad = None
            # for e in range(num_elems):
            #     d_sigma_dz = d_sigma_vm_dE[e] * dE_dz[e]
            #     dYdz = dY_dz[e]
            #     bracket = (d_sigma_dz * Y[e] - dYdz * S[e]) / (Y[e] ** 2) 
            #     grad_z[0:num_elems] += p * (inv_sf_elem[e] ** (p - 1)) * bracket[0]
            #     grad_z[num_elems:] += p * (inv_sf_elem[e] ** (p - 1)) * bracket[1]
            # grad_z = (1.0 / p) * outer * grad_z

            # # 3. Assemble full gradient for constraint
            # grad_safety = np.zeros_like(zeta)
            # grad_safety[:num_elems] = grad_inv_sf_density
            # grad_safety[num_elems:] = grad_z
            # dc[m, :] = grad_safety
            pass

        elif constraintType == TO_QOI.COST:
            # Cost constraint: sum of density * mass_density * cost * element volume
            decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(2,-1).T)
            mass_density = matEncoder.getMaterialProperties(decoded)['Density']
            costperunitmass = matEncoder.getMaterialProperties(decoded)['Cost']
            pseudodensity = zetaTensor[0:fe_solver_structural.mesh.num_elems]
            elemVolume = fe_solver_structural.mesh.elem_size[0] * fe_solver_structural.mesh.elem_size[1] * fe_solver_structural.mesh.elem_size[2]
            totalCost = torch.einsum('m,m,m->m', mass_density, costperunitmass, pseudodensity).sum()*elemVolume
            costConstraint = ((totalCost /constraintLimit) - 1.0)
            costConstraint.backward(retain_graph=True)
            cons_cost = costConstraint.detach().numpy()
            grad_cons_cost = zetaTensor.grad.detach().numpy()
            c[m, 0] = cons_cost
            dc[m, :] = grad_cons_cost
        elif constraintType == TO_QOI.CRITICALITY:
            decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(2,-1).T)
            criticality = matEncoder.getMaterialProperties(decoded)['Criticality']
            mass_density = matEncoder.getMaterialProperties(decoded)['Density']
            pseudodensity = zetaTensor[0:fe_solver_structural.mesh.num_elems]
            elemVolume =  fe_solver_structural.mesh.elem_size[0] * fe_solver_structural.mesh.elem_size[1] * fe_solver_structural.mesh.elem_size[2]
            totalmass = torch.einsum('m,m->m', mass_density, pseudodensity).sum()*elemVolume
            avgCriticality = torch.einsum('m,m,m->m',mass_density, criticality, pseudodensity).sum()*elemVolume/totalmass
            criticalityConstraint = ((avgCriticality / constraintLimit) - 1.0)
            criticalityConstraint.backward(retain_graph=True)
            cons_criticality = criticalityConstraint.detach().numpy()
            grad_cons_criticality = zetaTensor.grad.detach().numpy()
            c[m, 0] = cons_criticality
            dc[m, :] = grad_cons_criticality

        else:
            raise NotImplementedError(f"Constraint {constraintType} is not implemented yet.")

    return c, dc