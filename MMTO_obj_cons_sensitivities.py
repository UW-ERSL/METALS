import numpy as np
import torch
from PyTOImports import *
# --- Support Functions ---

def compute_pnorm_stress_and_sensitivity(sol: np.ndarray, x, fe_solver, EDesign, KETemplate, material_model):
    """
    Compute von Mises stress and sensitivity with respect to x for p-norm stress.
    """
    # "An efficient 146-line 3D sensitivity analysis code of 
    # stress based topology optimization written in MATLAB"
    # Optimization and Engineering (2022) 23:1733–1757
    # The sensitivity of pnorm von mises stress with respect to x has 2 terms: T1 and T2
    # T1 arises due to the stress relaxation: x**STRESS_RELAXATION
    # T2 arises indirectly via the solution sensitivity via the adjoint
    mesh = fe_solver.mesh
    nelems = mesh.num_elems

    qStress = 2  # STRESS factor for sensitivity
    pSIMP = 3

    if isinstance(fe_solver.mat_prop, list):
        E = EDesign
        nu = np.array([fe_solver.mat_prop[i].poissons_ratio for i in range(nelems)])
        D_list = []
        for Ei, nui in zip(E, nu):
            D = hex_element_stiffness.isotropic_constitutive_matrix(Ei, nui)
            D_list.append(D)
        D = np.stack(D_list)
    else:
        E = fe_solver.mat_prop.youngs_modulus
        nu = fe_solver.mat_prop.poissons_ratio
        D = hex_element_stiffness.isotropic_constitutive_matrix(E, nu)

    gradN = (1 / 8) * np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1]
    ])
    # Define the B matrix (strain-displacement matrix) for a hexahedral element at the center (xi=0, eta=0, zeta=0)
    B = np.zeros((6, 24))
    # Vectorized construction of B matrix for all 8 nodes at once
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
    # Vectorized assignment to B
    idx = np.arange(8)
    B[:, (3 * idx)[:, None] + np.arange(3)] = Bi.transpose(0, 2, 1)
    # F can be per-element for multi-material
    if isinstance(E, np.ndarray):
        F_stack = np.array([D[e] @ B for e in range(nelems)])
    else:
        F = D @ B
    g_elem = np.zeros((nelems, 24))
    vm_elems = np.zeros(nelems)
    T1 = np.zeros(nelems)
    T2 = np.zeros(nelems)

    for e in range(nelems):
        vm_elems[e] = fe_solver.vonMisesStress[e]
        T1[e] = pSIMP * (x[e] ** (pSIMP - 1)) * vm_elems[e]

        stress_elem = fe_solver.stressComponents[e]
        sigma11, sigma22, sigma33, sigma12, sigma13, sigma23 = stress_elem
        if isinstance(E, np.ndarray):
            F = F_stack[e]
        g_e = ((sigma11 - sigma22) * (F[0] - F[1]) +
               (sigma11 - sigma33) * (F[0] - F[2]) +
               (sigma22 - sigma33) * (F[1] - F[2]) +
               6 * sigma12 * F[3] + 6 * sigma13 * F[4] + 6 * sigma23 * F[5]) / np.sqrt(2)
        g_elem[e] = pSIMP * qStress * vm_elems[e] ** (pSIMP * qStress - 2) * g_e

    max_vm = np.max(vm_elems)
    # Note that we are using the relaxed von Mises below
    pNormExponent = 6
    vm_pnorm = np.sum(vm_elems ** pNormExponent) ** (1 / pNormExponent)
    T1 *= (1 / pNormExponent) * (np.sum(vm_elems ** pNormExponent) ** (1 / pNormExponent - 1))

    # Now compute the rhs of adjoint eqn 
    g = np.zeros(fe_solver.bc.num_dofs)
    for e in range(nelems):  # assemble  g vector
        edof = mesh.edofMat[e]
        g[edof] += g_elem[e]
    g *= -(1 / pNormExponent) * (np.sum(vm_elems ** pNormExponent) ** (1 / pNormExponent - 1))

    # Solve the adjoint	
    adjointSol = linear_solvers.solve(
        fe_solver.stiff_mtrx,
        g,
        fe_solver.solver,
        fe_solver.bc,
        dsolver=fe_solver.dsolver,
        **fe_solver.kwargs
    )

    dofMat = fe_solver.mesh.edofMat
    num_elems = fe_solver.mesh.num_elems
    nRows = KETemplate.shape[0]
    ce = (np.dot(adjointSol[dofMat].reshape(num_elems, nRows), KETemplate) * sol[dofMat].reshape(num_elems, nRows)).sum(1)*EDesign

    T2 = get_structural_material_model_sensitivity(x, material_model) * ce
    vm_pnorm_sensitivity = T1 + T2
    return vm_pnorm, vm_pnorm_sensitivity, max_vm

def compute_pnorm_safety_factor_and_sensitivity(sol: np.ndarray, x, fe_solver, EDesign,YDesign, KETemplate, material_model):
    """
    Compute von Mises stress and sensitivity with respect to x for p-norm stress.
    """
    # "An efficient 146-line 3D sensitivity analysis code of 
    # stress based topology optimization written in MATLAB"
    # Optimization and Engineering (2022) 23:1733–1757
    # The sensitivity of pnorm von mises stress with respect to x has 2 terms: T1 and T2
    # T1 arises due to the stress relaxation: x**STRESS_RELAXATION
    # T2 arises indirectly via the solution sensitivity via the adjoint
    mesh = fe_solver.mesh
    nelems = mesh.num_elems

    qStress = 2  # STRESS factor for sensitivity
    pSIMP = 3


    if isinstance(fe_solver.mat_prop, list):
        Y = YDesign
        E = EDesign
        nu = np.array([fe_solver.mat_prop[i].poissons_ratio for i in range(nelems)])
        D_list = []
        for Ei, nui in zip(E, nu):
            D = hex_element_stiffness.isotropic_constitutive_matrix(Ei, nui)
            D_list.append(D)
        D = np.stack(D_list)
    else:
        E = fe_solver.mat_prop.youngs_modulus
        nu = fe_solver.mat_prop.poissons_ratio
        D = hex_element_stiffness.isotropic_constitutive_matrix(E, nu)
    gradN = (1 / 8) * np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1]
    ])
    # Define the B matrix (strain-displacement matrix) for a hexahedral element at the center (xi=0, eta=0, zeta=0)
    B = np.zeros((6, 24))
    # Vectorized construction of B matrix for all 8 nodes at once
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
    # Vectorized assignment to B
    idx = np.arange(8)
    B[:, (3 * idx)[:, None] + np.arange(3)] = Bi.transpose(0, 2, 1)
    if isinstance(E, np.ndarray):
        F_stack = np.array([D[e] @ B for e in range(nelems)])
    else:
        F = D @ B
    g_elem = np.zeros((nelems, 24))
    inv_sf_elems = np.zeros(nelems)
    T1 = np.zeros(nelems)
    T2 = np.zeros(nelems)

    for e in range(nelems):
        inv_sf_elems[e] = fe_solver.vonMisesStress[e]/Y[e]
        T1[e] = pSIMP * (x[e] ** (pSIMP - 1)) * inv_sf_elems[e]

        stress_elem = fe_solver.stressComponents[e]
        sigma11, sigma22, sigma33, sigma12, sigma13, sigma23 = stress_elem
        if isinstance(E, np.ndarray):
            F = F_stack[e]
        g_e = ((sigma11 - sigma22) * (F[0] - F[1]) +
               (sigma11 - sigma33) * (F[0] - F[2]) +
               (sigma22 - sigma33) * (F[1] - F[2]) +
               6 * sigma12 * F[3] + 6 * sigma13 * F[4] + 6 * sigma23 * F[5]) / np.sqrt(2)
        g_elem[e] = pSIMP * qStress * inv_sf_elems[e] ** (pSIMP * qStress - 2) * g_e

    max_inv_sf = np.max(inv_sf_elems)
    # Note that we are using the relaxed von Mises below
    pNormExponent = 6
    inv_sf_pnorm = np.sum(inv_sf_elems ** pNormExponent) ** (1 / pNormExponent)
    T1 *= (1 / pNormExponent) * (np.sum(inv_sf_elems ** pNormExponent) ** (1 / pNormExponent - 1))

    # Now compute the rhs of adjoint eqn 
    g = np.zeros(fe_solver.bc.num_dofs)
    for e in range(nelems):  # assemble  g vector
        edof = mesh.edofMat[e]
        g[edof] += g_elem[e]
    g *= -(1 / pNormExponent) * (np.sum(inv_sf_elems ** pNormExponent) ** (1 / pNormExponent - 1))

    # Solve the adjoint	
    adjointSol = linear_solvers.solve(
        fe_solver.stiff_mtrx,
        g,
        fe_solver.solver,
        fe_solver.bc,
        dsolver=fe_solver.dsolver,
        **fe_solver.kwargs
    )

    dofMat = fe_solver.mesh.edofMat
    num_elems = fe_solver.mesh.num_elems
    nRows = KETemplate.shape[0]
    ce = (np.dot(adjointSol[dofMat].reshape(num_elems, nRows), KETemplate) * sol[dofMat].reshape(num_elems, nRows)).sum(1)*EDesign

    T2 = get_structural_material_model_sensitivity(x, material_model) * ce
    inv_sf_pnorm_sensitivity = T1 + T2

    return inv_sf_pnorm, inv_sf_pnorm_sensitivity, max_inv_sf

def compute_volume_constraint_and_gradient(x: np.ndarray, volfracUpper: float) -> tuple:
    volConstraint = ((np.mean(x)/volfracUpper) - 1.0)
    volConstraint_gradient = np.ones_like(x) / volfracUpper/ x.size
    return volConstraint, volConstraint_gradient

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
        ce = (np.dot(sol[fe_solver.mesh.edofMat].reshape(num_elems, 24), KETemplate) * sol[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
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
        grad_pnorm_stress[num_elems:] = 0.1*grad_vm_z

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
            ce = (np.dot(sol[fe_solver.mesh.edofMat].reshape(num_elems, 24), KETemplate) * sol[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
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
        elif constraintType == TO_QOI.STRESS_SAFETY_FACTOR:
            decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(latentDim,-1).T)
            material_properties = matEncoder.getMaterialProperties(decoded)
            ETensor = material_properties['Youngs_Modulus']
            YTensor = material_properties['Yield_Strength']
            YDesign= YTensor.detach().numpy()
            EDesign = ETensor.detach().numpy()
            inv_sf_pnorm, grad_inv_sf_density,_ = compute_pnorm_safety_factor_and_sensitivity(
                sol, x, fe_solver,EDesign,YDesign, KETemplate, MaterialModel.SIMP)
            safety_factor = constraintLimit
            safety_constraint = inv_sf_pnorm - (1.0 / safety_factor)
            c[m, 0] = safety_constraint

            # 2. Compute latent variable part of gradient (chain rule)
            pNormExponent = 6
            sigma_vm = fe_solver.vonMisesStress
            d_sigma_vm_dE = sigma_vm / EDesign
            Y = YDesign
            S = sigma_vm
            inv_sf_elem = S / Y
            outer = (np.sum(sigma_vm ** pNormExponent)) ** (1.0 / pNormExponent - 1)
            grad_inv_sf_z = np.zeros(latentDim*num_elems)
            # Backward for dE/dz and dY/dz
            zetaTensor.grad = None
            ETensor.backward(torch.ones_like(ETensor), retain_graph=True)
            dE_dz = zetaTensor.grad[num_elems:].detach().numpy().reshape(num_elems, latentDim)
            zetaTensor.grad = None
            YTensor.backward(torch.ones_like(YTensor), retain_graph=True)
            dY_dz = zetaTensor.grad[num_elems:].detach().numpy().reshape(num_elems, latentDim)
            zetaTensor.grad = None
            for d in range(latentDim):
                d_sigma_dz = d_sigma_vm_dE * dE_dz[:,d]
                d_inv_sf_dz = (d_sigma_dz * Y - (sigma_vm / (Y ** 2)) * dY_dz[:,d])
                grad_inv_sf_z[d*num_elems:(d+1)*num_elems] = (pNormExponent * (inv_sf_elem ** (pNormExponent - 1))) * d_inv_sf_dz
            grad_inv_sf_z = (1.0 / pNormExponent) * outer * grad_inv_sf_z

            grad_inv_safety = np.zeros_like(zeta)
            grad_inv_safety[:num_elems] = grad_inv_sf_density
            grad_inv_safety[num_elems:] = grad_inv_sf_z
            dc[m, :] = grad_inv_safety

        else:
            raise NotImplementedError(f"Constraint {constraintType} is not implemented yet.")

    return c, dc