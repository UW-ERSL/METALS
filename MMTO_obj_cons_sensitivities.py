import numpy as np
import torch
from PyTOImports import *
# --- Support Functions ---

def compute_pnorm_stress_and_sensitivity(sol: np.ndarray, x, fe_solver, EDesign, KETemplate, material_model):
    """
    MMTO-compatible: Compute von Mises stress and sensitivity with respect to x for p-norm stress.
    """
    mesh = fe_solver.mesh
    nelems = mesh.num_elems

    qStress = 0.5  # STRESS relaxation factor
    pSIMP = 3    # SIMP penalization
    pNormExponent = 6  # p-norm exponent

    # Get element-wise Poisson's ratio
    if isinstance(fe_solver.mat_prop, list):
        nu = np.array([fe_solver.mat_prop[i].poissons_ratio for i in range(nelems)])
    else:
        nu = fe_solver.mat_prop.poissons_ratio

    # Build element-wise constitutive matrices
    D_list = []
    for Ei, nui in zip(EDesign, nu):
        D = hex_element_stiffness.isotropic_constitutive_matrix(Ei, nui)
        D_list.append(D)
    D = np.stack(D_list)

    # B matrix setup
    gradN = (1 / 8) * np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1]
    ])
    for i in range(3):
        gradN[i, :] = 2*gradN[i,:] / fe_solver.mesh.elem_size[i]
    B = np.zeros((6, 24))
    Bi = np.zeros((6, 3, 8))
    Bi[0, 0, :] = gradN[0, :]
    Bi[1, 1, :] = gradN[1, :]
    Bi[2, 2, :] = gradN[2, :]
    Bi[3, 0, :] = gradN[1, :]
    Bi[3, 1, :] = gradN[0, :]
    Bi[4, 0, :] = gradN[2, :]
    Bi[4, 2, :] = gradN[0, :]
    Bi[5, 1, :] = gradN[2, :]
    Bi[5, 2, :] = gradN[1, :]
    idx = np.arange(8)
    B[:, (3 * idx)[:, None] + np.arange(3)] = Bi.transpose(0, 2, 1)

    vm_elems = fe_solver.vonMisesStress
    vm_pnorm = fe_solver.pNormStress
    
    # Compute dpn_dvms = (sum(vm^p))^(1/p - 1)
    dpn_dvms = (np.sum(vm_elems ** pNormExponent)) ** (1/pNormExponent - 1)
    
    # Pre-compute DvmDs for all elements
    DvmDs_all = np.zeros((nelems, 6))
    for e in range(nelems):
        stress_elem = fe_solver.stressComponents[e]
        sigma11, sigma22, sigma33, sigma12, sigma13, sigma23 = stress_elem
        
        # DvmDs - derivative of von Mises w.r.t. stress components
        DvmDs_all[e, 0] = 1/(2*vm_elems[e]) * (2*sigma11 - sigma22 - sigma33)
        DvmDs_all[e, 1] = 1/(2*vm_elems[e]) * (2*sigma22 - sigma11 - sigma33)
        DvmDs_all[e, 2] = 1/(2*vm_elems[e]) * (2*sigma33 - sigma11 - sigma22)
        DvmDs_all[e, 3] = 3/vm_elems[e] * sigma12
        DvmDs_all[e, 4] = 3/vm_elems[e] * sigma13
        DvmDs_all[e, 5] = 3/vm_elems[e] * sigma23
    
    # Compute T1 (direct sensitivity)
    beta = np.zeros(nelems)
    x = np.maximum(x, 1e-12) # avoid division by zero
    for e in range(nelems):
        edof = mesh.edofMatStructural[e]
        u_e = sol[edof]
        beta[e] = qStress * (x[e]**(qStress-1)) * (vm_elems[e]**(pNormExponent-1)) * DvmDs_all[e] @ D[e] @ B @ u_e
    
    T1 = dpn_dvms * beta
    
    # Compute adjoint right-hand side using pre-computed DvmDs
    g = np.zeros(fe_solver.bc.num_dofs)
    for e in range(nelems):
        edof = mesh.edofMatStructural[e]
        g_e = (x[e]**qStress) * dpn_dvms * B.T @ D[e].T @ DvmDs_all[e] * (vm_elems[e]**(pNormExponent-1))
        g[edof] += g_e
    
    # Solve adjoint equation
    adjointSol = linear_solvers.solve(fe_solver.stiff_mtrx,
                                       g,
                                       fe_solver.solver,
                                       fe_solver.bc,
                                       dsolver=fe_solver.dsolver,
                                       **fe_solver.kwargs)
    
    # Compute T2 (indirect sensitivity via adjoint)
    dofMat = fe_solver.mesh.edofMatStructural
    nRows = KETemplate.shape[0]
    ce = (np.dot(adjointSol[dofMat].reshape(nelems, nRows), KETemplate) * sol[dofMat].reshape(nelems, nRows)).sum(1)*EDesign
    
    T2 = -pSIMP * (x**(pSIMP-1)) * ce  # Note the negative sign from MATLAB
    
    vm_pnorm_sensitivity = T1 + T2
    max_vm = np.max(vm_elems)
    
    return vm_pnorm, vm_pnorm_sensitivity, max_vm

def compute_pnorm_safety_factor_and_sensitivity(sol: np.ndarray, x, fe_solver, EDesign,YDesign, KETemplate, material_model):
    """
    MMTO-compatible: Compute p-norm of inverse safety factor and its sensitivity with respect to x.
    """
    pNormExponent = 6  # p-norm exponent
    # Compute inverse safety factor per element
    sigma_vm = fe_solver.vonMisesStress
    Y = YDesign
    inv_sf_elems = sigma_vm / Y
    
    inv_sf_pnorm = np.sum(inv_sf_elems ** pNormExponent) ** (1.0 / pNormExponent)

    # d (inv_sf_pnorm) / d (x) = d (inv_sf_pnorm) / d (inv_sf) * d (inv_sf) / d (x)

    # Compute dpn_dinv_sf = (sum(inv_sf^p))^(1/p - 1)
    dpn_dinv_sf = (np.sum(inv_sf_elems ** pNormExponent)) ** (1.0 / pNormExponent - 1)

    # d(inv_sf) / d (x) = d (sigma_vm / Y) / d (x) = (1/Y) * d (sigma_vm) / d (x)
    # d (sigma_vm) / d (x) is computed similarly to p-norm stress sensitivity, exccept we scale by (1/Y) and we don' take pNorm

    mesh = fe_solver.mesh
    nelems = mesh.num_elems

    qStress = 0.5  # STRESS relaxation factor
    pSIMP = 3    # SIMP penalization

    # Get element-wise Poisson's ratio
    if isinstance(fe_solver.mat_prop, list):
        nu = np.array([fe_solver.mat_prop[i].poissons_ratio for i in range(nelems)])
    else:
        nu = fe_solver.mat_prop.poissons_ratio

    # Build element-wise constitutive matrices
    D_list = []
    for Ei, nui in zip(EDesign, nu):
        D = hex_element_stiffness.isotropic_constitutive_matrix(Ei, nui)
        D_list.append(D)
    D = np.stack(D_list)

    # B matrix setup
    gradN = (1 / 8) * np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1]
    ])
    for i in range(3):
        gradN[i, :] = 2*gradN[i,:] / fe_solver.mesh.elem_size[i]
    B = np.zeros((6, 24))
    Bi = np.zeros((6, 3, 8))
    Bi[0, 0, :] = gradN[0, :]
    Bi[1, 1, :] = gradN[1, :]
    Bi[2, 2, :] = gradN[2, :]
    Bi[3, 0, :] = gradN[1, :]
    Bi[3, 1, :] = gradN[0, :]
    Bi[4, 0, :] = gradN[2, :]
    Bi[4, 2, :] = gradN[0, :]
    Bi[5, 1, :] = gradN[2, :]
    Bi[5, 2, :] = gradN[1, :]
    idx = np.arange(8)
    B[:, (3 * idx)[:, None] + np.arange(3)] = Bi.transpose(0, 2, 1)

    
    # Pre-compute DinvSfDs for all elements
    DinvSfDs_all = np.zeros((nelems, 6))
    for e in range(nelems):
        stress_elem = fe_solver.stressComponents[e]
        sigma11, sigma22, sigma33, sigma12, sigma13, sigma23 = stress_elem
        vm = sigma_vm[e]
        Y_e = Y[e]
        # Derivative of inv_sf w.r.t. stress components
        DvmDs = np.zeros(6)
        DvmDs[0] = 1/(2*vm) * (2*sigma11 - sigma22 - sigma33)
        DvmDs[1] = 1/(2*vm) * (2*sigma22 - sigma11 - sigma33)
        DvmDs[2] = 1/(2*vm) * (2*sigma33 - sigma11 - sigma22)
        DvmDs[3] = 3/vm * sigma12
        DvmDs[4] = 3/vm * sigma13
        DvmDs[5] = 3/vm * sigma23
        DinvSfDs_all[e, :] = DvmDs / Y_e

    # Compute T1 (direct sensitivity)
    beta = np.zeros(nelems)
    x = np.maximum(x, 1e-12) # avoid division by zero
    for e in range(nelems):
        edof = mesh.edofMatStructural[e]
        u_e = sol[edof]
        beta[e] = qStress * (x[e]**(qStress-1)) * (inv_sf_elems[e]**(pNormExponent-1)) * DinvSfDs_all[e] @ D[e] @ B @ u_e

    T1 = dpn_dinv_sf * beta

    # Compute adjoint right-hand side using pre-computed DinvSfDs
    g = np.zeros(fe_solver.bc.num_dofs)
    for e in range(nelems):
        edof = mesh.edofMatStructural[e]
        g_e = (x[e]**qStress) * dpn_dinv_sf * B.T @ D[e].T @ DinvSfDs_all[e] * (inv_sf_elems[e]**(pNormExponent-1))
        g[edof] += g_e

    # Solve adjoint equation
    adjointSol = linear_solvers.solve(fe_solver.stiff_mtrx,
                                    g,
                                    fe_solver.solver,
                                    fe_solver.bc,
                                    dsolver=fe_solver.dsolver,
                                    **fe_solver.kwargs)

    # Compute T2 (indirect sensitivity via adjoint)
    dofMat = fe_solver.mesh.edofMatStructural
    nRows = KETemplate.shape[0]
    ce = (np.dot(adjointSol[dofMat].reshape(nelems, nRows), KETemplate) * sol[dofMat].reshape(nelems, nRows)).sum(1)*EDesign

    T2 = -pSIMP * (x**(pSIMP-1)) * ce  # Note the negative sign from MATLAB

    inv_sf_pnorm_sensitivity = T1 + T2
    max_inv_sf = np.max(inv_sf_elems)

    return inv_sf_pnorm, inv_sf_pnorm_sensitivity, max_inv_sf

def compute_volumefraction_constraint_and_gradient(x: np.ndarray, volfracUpper: float) -> tuple:
    volFracConstraint = ((np.mean(x)/volfracUpper) - 1.0)
    volFracConstraint_gradient = np.ones_like(x) / volfracUpper/ x.size
    return volFracConstraint, volFracConstraint_gradient

# --- Main Objective/Constraint Functions ---
def compute_mmto_objective_and_gradient(to_params, sol, zeta, fe_solver, KETemplate, matEncoder):
    """
    Compute objective value and its gradient for METALS LSR.
    Handles:
      1. Compliance objective (subject to mass)
      2. Compliance objective (subject to mass and cost)
      3. Mass objective (subject to yield strength and compliance)
    """
    objectiveType = to_params.Objective[0]
    optionalParam = to_params.Objective[1]
    num_elems = fe_solver.mesh.num_elems
    x = zeta[0:fe_solver.mesh.num_elems]
    zetaTensor = torch.tensor(zeta).float()
    zetaTensor.requires_grad = True

    latentDim = matEncoder.vae_params.latentDim

    if objectiveType == TO_QOI.COMPLIANCE:
        decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(latentDim,-1).T)
        material_properties = matEncoder.getMaterialProperties(decoded)
        youngsModulus = material_properties['Youngs_Modulus']
        EDesign = youngsModulus.detach().numpy()
        compliance = np.einsum('i, i -> ', fe_solver.total_force, sol)
        ce = (np.dot(sol[fe_solver.mesh.edofMatStructural].reshape(num_elems, 24), KETemplate) * sol[fe_solver.mesh.edofMatStructural].reshape(num_elems, 24)).sum(1)
        penal = 3.0
        dJ_dxDesign = (-penal * x ** (penal - 1)) * EDesign * ce
        dJ_dEDesign = np.asarray((x ** penal) * ce)
        dJ_dEDesign_tensor = torch.tensor(dJ_dEDesign)
        zetaTensor.grad = None
        youngsModulus.backward(dJ_dEDesign_tensor)
        dJ_dzeta = zetaTensor.grad.detach().numpy()
        grad_compliance = np.concatenate((dJ_dxDesign, -dJ_dzeta[num_elems:].flatten()))
        return compliance, grad_compliance
    
    elif objectiveType == TO_QOI.PNORM_STRESS:
        decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(latentDim,-1).T)
        material_properties = matEncoder.getMaterialProperties(decoded)
        ETensor = material_properties['Youngs_Modulus']
        EDesign = ETensor.detach().numpy()
        
        vm_pnorm, grad_vm_density, max_vm = compute_pnorm_stress_and_sensitivity(
            sol, x, fe_solver, EDesign, KETemplate, MaterialModel.SIMP)
        
        sigma_vm = fe_solver.vonMisesStress
        d_sigma_vm_dE = sigma_vm / EDesign
        pNormExponent = 6
        outer = (np.sum(sigma_vm ** pNormExponent)) ** (1.0 / pNormExponent - 1)

        # Backward for dE/dz 
        zetaTensor.grad = None
        ETensor.backward(torch.ones_like(ETensor), retain_graph=True)
        dE_dz = zetaTensor.grad[num_elems:].detach().numpy().reshape(num_elems, latentDim)

        grad_vm_z = np.zeros(latentDim*num_elems)
        for d in range(latentDim):
            grad_vm_z[d*num_elems:(d+1)*num_elems] = (pNormExponent * (sigma_vm ** (pNormExponent - 1))) * (d_sigma_vm_dE * dE_dz[:,d])
        grad_vm_z = (1.0 / pNormExponent) * outer * grad_vm_z
        grad_pnorm_stress = np.zeros_like(zeta)
        grad_pnorm_stress[0:num_elems] = grad_vm_density
        grad_pnorm_stress[num_elems:] = grad_vm_z

        return vm_pnorm, grad_pnorm_stress
    
    elif objectiveType == TO_QOI.MASS:
        decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(latentDim,-1).T)
        mass_density = matEncoder.getMaterialProperties(decoded)['Density']
        pseudodensity = zetaTensor[0:num_elems]
        elemVolume =  fe_solver.mesh.elem_size[0] * fe_solver.mesh.elem_size[1] * fe_solver.mesh.elem_size[2]
        zetaTensor.grad = None
        totalMass = torch.einsum('m,m->m', mass_density, pseudodensity).sum()*elemVolume 
        totalMass.backward(retain_graph=True)
        obj_mass = totalMass.detach().numpy()
        grad_mass = zetaTensor.grad.detach().numpy()
        return obj_mass, grad_mass
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
    zetaTensor = torch.tensor(zeta).float()
    zetaTensor.requires_grad = True

    latentDim = matEncoder.vae_params.latentDim

    c = np.zeros((nConstraints, 1))
    num_design_var = zeta.size
    dc = np.zeros((nConstraints, num_design_var))
    for m in range(nConstraints):
        constraintType = to_params.Constraints[m][0]
        optionalParam = to_params.Constraints[m][1]
        constraintLimit = to_params.Constraints[m][2]

        if constraintType == TO_QOI.COMPLIANCE:
            decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(latentDim,-1).T)
            material_properties = matEncoder.getMaterialProperties(decoded)
            youngsModulus = material_properties['Youngs_Modulus']
            EDesign = youngsModulus.detach().numpy()
            compliance = np.einsum('i, i -> ', fe_solver.total_force, sol)
            ce = (np.dot(sol[fe_solver.mesh.edofMatStructural].reshape(num_elems, 24), KETemplate) * sol[fe_solver.mesh.edofMatStructural].reshape(num_elems, 24)).sum(1)
            penal = 3.0
            dJ_dxDesign = (-penal * x ** (penal - 1)) * EDesign * ce
            dJ_dEDesign = np.asarray((x ** penal) * ce)
            dJ_dEDesign_tensor = torch.tensor(dJ_dEDesign)
            zetaTensor.grad = None
            youngsModulus.backward(dJ_dEDesign_tensor)
            dJ_dzeta = zetaTensor.grad.detach().numpy()
            grad_compliance = np.concatenate((dJ_dxDesign, -dJ_dzeta[num_elems:].flatten()))
            complianceConstraint = ((compliance / constraintLimit) - 1.0)
            grad_complianceConstraint = grad_compliance / constraintLimit
            c[m, 0] = complianceConstraint
            dc[m, :] = grad_complianceConstraint

        elif constraintType == TO_QOI.VOLUME_FRACTION:
            volfracConstraint, volfracConstraint_gradient = compute_volumefraction_constraint_and_gradient(
                x, constraintLimit)
            grad_volfracConstraint = np.zeros_like(zeta)
            grad_volfracConstraint[0:num_elems] = volfracConstraint_gradient
            c[m, 0] = volfracConstraint
            dc[m, :] = grad_volfracConstraint
            
        elif constraintType == TO_QOI.MASS:
            decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(latentDim,-1).T)
            mass_density = matEncoder.getMaterialProperties(decoded)['Density']
            pseudodensity = zetaTensor[0:num_elems]
            elemVolume = fe_solver.mesh.elem_size[0] * fe_solver.mesh.elem_size[1] * fe_solver.mesh.elem_size[2]
            totalMass = torch.einsum('m,m->m', mass_density, pseudodensity).sum()*elemVolume 
            zetaTensor.grad = None
            massConstraint = ((totalMass / constraintLimit) - 1.0)
            massConstraint.backward(retain_graph=True)
            cons_mass = massConstraint.detach().numpy()
            grad_cons_mass = zetaTensor.grad.detach().numpy()
            c[m, 0] = cons_mass
            dc[m, :] = grad_cons_mass
        
        elif constraintType == TO_QOI.COST:
            decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(latentDim,-1).T)
            mass_density = matEncoder.getMaterialProperties(decoded)['Density']
            costperunitmass = matEncoder.getMaterialProperties(decoded)['Cost']
            pseudodensity = zetaTensor[0:fe_solver.mesh.num_elems]
            elemVolume = fe_solver.mesh.elem_size[0] * fe_solver.mesh.elem_size[1] * fe_solver.mesh.elem_size[2]
            totalCost = torch.einsum('m,m,m->m', mass_density, costperunitmass, pseudodensity).sum()*elemVolume
            costConstraint = ((totalCost /constraintLimit) - 1.0)
            zetaTensor.grad = None
            costConstraint.backward(retain_graph=True)
            cons_cost = costConstraint.detach().numpy()
            grad_cons_cost = zetaTensor.grad.detach().numpy()
            c[m, 0] = cons_cost
            dc[m, :] = grad_cons_cost
        elif constraintType == TO_QOI.MAX_CRITICALITY:
            decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(latentDim,-1).T)
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
            decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(latentDim,-1).T)
            criticality = matEncoder.getMaterialProperties(decoded)['Criticality']
            meanCriticality = torch.mean(criticality)
            criticalityConstraint = ((meanCriticality / constraintLimit) - 1.0)
            zetaTensor.grad = None
            criticalityConstraint.backward(retain_graph=True)
            cons_criticality = criticalityConstraint.detach().numpy()
            grad_cons_criticality = zetaTensor.grad.detach().numpy()
            c[m, 0] = cons_criticality
            dc[m, :] = grad_cons_criticality
        elif constraintType == TO_QOI.STRESS_FAILURE_FACTOR:
            
            decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(latentDim,-1).T)
            material_properties = matEncoder.getMaterialProperties(decoded)
            ETensor = material_properties['Youngs_Modulus']
            YTensor = material_properties['Yield_Strength']
            YDesign= YTensor.detach().numpy()
            EDesign = ETensor.detach().numpy()

            inv_sf_pnorm, grad_ff_density,stress_ff_max = compute_pnorm_safety_factor_and_sensitivity(
                sol, x, fe_solver,EDesign,YDesign, KETemplate, MaterialModel.SIMP)
            safety_constraint = stress_ff_max/constraintLimit - 1.0
            c[m, 0] = safety_constraint
            grad_stress_ff = np.zeros_like(zeta)
            grad_stress_ff[:num_elems] = grad_ff_density/constraintLimit

            # 2. Compute latent variable part of gradient (simple approach)
            # we assume that latent variables have a more direct influence on yield strength than on von Mises stress via E
            sigma_vm = fe_solver.vonMisesStress
            pNormExponent = 6
            stress_ff = torch.tensor(sigma_vm) / YTensor
            # we can use max directly, but to keep consistent with p-norm approach used above
            ff_pNorm= torch.sum(stress_ff ** pNormExponent) ** (1.0 / pNormExponent)
            zetaTensor.grad = None
            ff_pNorm.backward(retain_graph=True)
            grad_stress_ff[num_elems:] = zetaTensor.grad[num_elems:].detach().numpy()/constraintLimit
            
            dc[m, :] = grad_stress_ff
        else:
            raise NotImplementedError(f"Constraint {constraintType} is not implemented yet.")

    return c, dc