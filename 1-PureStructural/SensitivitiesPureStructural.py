import numpy as np
import sys
import os
common_path = os.path.join(os.path.dirname(__file__), '..', '0-Common')
sys.path.insert(0, os.path.abspath(common_path))

from PyTOImports import (get_pNorm_exponent, hex_element_stiffness, get_stress_relaxation_factor_sensitivity, # type: ignore
        get_stress_relaxation_correction, linear_solvers, get_structural_material_model_sensitivity, TO_QOI,
        MaterialModel,get_structural_material_model_scaling) # type: ignore

# --- Support Functions ---
def compute_pnorm_stress_and_sensitivity(sol: np.ndarray, x, fe_solver, EDesign, KETemplate, material_model):
    """
    MMTO-compatible: Compute von Mises stress and sensitivity with respect to x for p-norm stress.
    """
    mesh = fe_solver.mesh
    nelems = mesh.num_elems

    pNormExponent = get_pNorm_exponent()
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
        beta[e] = get_stress_relaxation_factor_sensitivity(x[e]) * (vm_elems[e]**(pNormExponent-1)) * DvmDs_all[e] @ D[e] @ B @ u_e
    
    T1 = dpn_dvms * beta
    
    # Compute adjoint right-hand side using pre-computed DvmDs
    g = np.zeros(fe_solver.bc.num_dofs)
    for e in range(nelems):
        edof = mesh.edofMatStructural[e]
        g_e = get_stress_relaxation_correction(x[e]) * dpn_dvms * B.T @ D[e].T @ DvmDs_all[e] * (vm_elems[e]**(pNormExponent-1))
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
    
    T2 = -get_structural_material_model_sensitivity(x,material_model) * ce  # Note the negative sign from MATLAB
    
    vm_pnorm_sensitivity = T1 + T2
    max_vm = np.max(vm_elems)
    
    return vm_pnorm, vm_pnorm_sensitivity, max_vm

def compute_pnorm_safety_factor_and_sensitivity(sol: np.ndarray, x, fe_solver, EDesign,YDesign, KETemplate, material_model):
    """
    MMTO-compatible: Compute p-norm of inverse safety factor and its sensitivity with respect to x.
    """
    pNormExponent = get_pNorm_exponent()  # p-norm exponent
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
        beta[e] = get_stress_relaxation_factor_sensitivity(x[e]) * (inv_sf_elems[e]**(pNormExponent-1)) * DinvSfDs_all[e] @ D[e] @ B @ u_e

    T1 = dpn_dinv_sf * beta

    # Compute adjoint right-hand side using pre-computed DinvSfDs
    g = np.zeros(fe_solver.bc.num_dofs)
    for e in range(nelems):
        edof = mesh.edofMatStructural[e]
        g_e = get_stress_relaxation_correction(x[e]) * dpn_dinv_sf * B.T @ D[e].T @ DinvSfDs_all[e] * (inv_sf_elems[e]**(pNormExponent-1))
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

    T2 = -get_structural_material_model_sensitivity(x, material_model) * ce  # Note the negative sign from MATLAB

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

    material_model = to_params.materialModel
    latentDim = matEncoder.vae_params.latentDim
    zPts = zeta[num_elems:].reshape((latentDim, -1)).T
    material_properties, gradients = matEncoder.getMaterialPropertiesAtLatentPoints(zPts, compute_gradients=True)

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

   
    pNormExponent = get_pNorm_exponent()
    if objectiveType == TO_QOI.COMPLIANCE:
        compliance = np.einsum('i, i -> ', fe_solver.total_force, sol)
        ce = (np.dot(sol[fe_solver.mesh.edofMatStructural].reshape(num_elems, 24), KETemplate) * sol[fe_solver.mesh.edofMatStructural].reshape(num_elems, 24)).sum(1)
        dJ_dxDesign = (-get_structural_material_model_sensitivity(x, material_model)) * EDesign * ce
        dJ_dEDesign = np.asarray((get_structural_material_model_scaling(x, material_model)) * ce)
        dJ_dzeta = (dJ_dEDesign * dE_dz).flatten()
        grad_compliance = np.concatenate((dJ_dxDesign, -dJ_dzeta))
        return compliance, grad_compliance
    elif objectiveType == TO_QOI.PNORM_STRESS:
        vm_pnorm, grad_vm_density, max_vm = compute_pnorm_stress_and_sensitivity(
            sol, x, fe_solver, EDesign, KETemplate, MaterialModel.SIMP)
        sigma_vm = fe_solver.vonMisesStress
    
        outer = (np.sum(sigma_vm ** pNormExponent)) ** (1.0 / pNormExponent - 1)
        d_sigma_vm_dE = sigma_vm / EDesign
        dE_dz = dE_dz.reshape(num_elems, latentDim)
        grad_vm_z = np.zeros(latentDim*num_elems)
        for d in range(latentDim):
            grad_vm_z[d*num_elems:(d+1)*num_elems] = (pNormExponent * (sigma_vm ** (pNormExponent - 1))) * (d_sigma_vm_dE * dE_dz[:,d])
        grad_vm_z = (1.0 / pNormExponent) * outer * grad_vm_z
        grad_pnorm_stress = np.zeros_like(zeta)
        grad_pnorm_stress[0:num_elems] = grad_vm_density
        grad_pnorm_stress[num_elems:] = grad_vm_z
        return vm_pnorm, grad_pnorm_stress
    elif objectiveType == TO_QOI.MASS:
        elemVolume =  fe_solver.mesh.elem_size[0] * fe_solver.mesh.elem_size[1] * fe_solver.mesh.elem_size[2]
        totalMass = np.einsum('m,m->m', mass_density, x).sum()*elemVolume 
        grad_mass = np.concatenate((mass_density*elemVolume, (dMassDensity_dz*x*elemVolume).flatten()))
        return totalMass, grad_mass
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