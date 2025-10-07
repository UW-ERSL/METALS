from unicodedata import name
import numpy as np
import torch
from PyTOImports import *
from HermiteFunction import hermiteInterpolation, hermiteInterpolation_torch
# --- Support Functions ---

def compute_pnorm_safety_factor_and_sensitivity(sol: np.ndarray, x, fe_solver, EDesign,YDesign, KETemplate, material_model, p):
    """
    Compute p-norm of (von Mises stress / yield strength) and its sensitivity for multi-material case.
    """
    mesh = fe_solver.mesh
    nelems = mesh.num_elems
    q = 1  # STRESS_RELAXATION factor

    # Handle multi-material: get yield strength for each element
    if isinstance(fe_solver.mat_prop, list):
        yield_strengths = YDesign
        E = EDesign
        nu = np.array([fe_solver.mat_prop[i].poissons_ratio for i in range(nelems)])
        D_list = []
        for Ei, nui in zip(E, nu):
            D = hex_element_stiffness.isotropic_constitutive_matrix(Ei, nui)
            D_list.append(D)
        D_stack = np.stack(D_list)
    else:
        yield_strengths = np.full(nelems, fe_solver.mat_prop.yield_strength)
        E = fe_solver.mat_prop.youngs_modulus
        nu = fe_solver.mat_prop.poissons_ratio
        D = hex_element_stiffness.isotropic_constitutive_matrix(E, nu)

    gradN = (1 / 8) * np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1]
    ])
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
    # F can be per-element for multi-material
    if isinstance(E, np.ndarray):
        F_stack = np.array([D_stack[e] @ B for e in range(nelems)])
    else:
        F = D @ B

    g_elem = np.zeros((nelems, 24))
    inv_sf_elems = np.zeros(nelems)
    T1 = np.zeros(nelems)
    T2 = np.zeros(nelems)

    for e in range(nelems):
        # Stress for T1 (no relaxation)
        stress_elem = fe_solver.stressComponents[e]
        sigma11, sigma22, sigma33, sigma12, sigma13, sigma23 = stress_elem
        vm = np.sqrt(0.5 * ((sigma11 - sigma22) ** 2 + (sigma22 - sigma33) ** 2 + (sigma33 - sigma11) ** 2)
            + 3 * (sigma12 ** 2 + sigma13 ** 2 + sigma23 ** 2)) 
        inv_sf = vm / yield_strengths[e]
        T1[e] = p * q * (x[e] ** (p * q - 1)) * inv_sf

        # Stress for T2 (with relaxation)
        stress_elem_relaxed = (x[e] ** q) * fe_solver.stressComponents[e]
        sigma11, sigma22, sigma33, sigma12, sigma13, sigma23 = stress_elem_relaxed
        vm_relaxed = np.sqrt( 0.5 * ((sigma11 - sigma22) ** 2 + (sigma22 - sigma33) ** 2 + (sigma33 - sigma11) ** 2)
            + 3 * (sigma12 ** 2 + sigma13 ** 2 + sigma23 ** 2))
        inv_sf_elems[e] = vm_relaxed / yield_strengths[e]

        if isinstance(E, np.ndarray):
            F = F_stack[e]
        # Sensitivity of von Mises stress w.r.t. displacement
        g_e = ((sigma11 - sigma22) * (F[0] - F[1])
            + (sigma11 - sigma33) * (F[0] - F[2])
            + (sigma22 - sigma33) * (F[1] - F[2])
            + 6 * sigma12 * F[3]
            + 6 * sigma13 * F[4]
            + 6 * sigma23 * F[5]) / np.sqrt(2)
        g_elem[e] = p * (inv_sf_elems[e] ** (p - 2)) * g_e 

    inv_sf_pnorm = np.sum(inv_sf_elems ** p) ** (1 / p)
    T1 *= (1 / p) * (np.sum(inv_sf_elems ** p) ** (1 / p - 1))

    # Assemble adjoint RHS
    g = np.zeros(fe_solver.bc.num_dofs)
    for e in range(nelems):
        edof = mesh.edofMat[e]
        g[edof] += g_elem[e]
    g *= -(1 / p) * (np.sum(inv_sf_elems ** p) ** (1 / p - 1))

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
    ce = ( np.dot(adjointSol[dofMat].reshape(num_elems, nRows), KETemplate)
        * sol[dofMat].reshape(num_elems, nRows) ).sum(1)*EDesign

    T2 = get_structural_material_model_sensitivity(x, material_model) * ce / yield_strengths 
    inv_sf_pnorm_sensitivity = T1  + T2

    return inv_sf_pnorm, inv_sf_pnorm_sensitivity

def d_relaxed_von_mises_dE(stress, x, q=1):
    """
    Compute derivative of relaxed von Mises stress with respect to Young's modulus E for a single element.
    stress: (6,) array-like [sxx, syy, szz, syz, sxz, sxy] (unrelaxed)
    x: density variable for the element
    q: stress relaxation exponent (default 1)
    Returns: scalar d(sigma_vm_relaxed)/dE
    """
    sxx, syy, szz, syz, sxz, sxy = stress
    # Relaxed stress
    factor = x**q
    sxx_r = factor * sxx
    syy_r = factor * syy
    szz_r = factor * szz
    syz_r = factor * syz
    sxz_r = factor * sxz
    sxy_r = factor * sxy
    sigma_vm_relaxed = np.sqrt(
        0.5 * ((sxx_r - syy_r) ** 2 + (syy_r - szz_r) ** 2 + (szz_r - sxx_r) ** 2) +
        3 * (syz_r ** 2 + sxz_r ** 2 + sxy_r ** 2)
    )
    if sigma_vm_relaxed == 0:
        return 0.0
    # Partial derivatives w.r.t. each relaxed stress component
    d_vm_dsxx = (2 * sxx_r - syy_r - szz_r) / (2 * sigma_vm_relaxed)
    d_vm_dsyy = (2 * syy_r - sxx_r - szz_r) / (2 * sigma_vm_relaxed)
    d_vm_dszz = (2 * szz_r - sxx_r - syy_r) / (2 * sigma_vm_relaxed)
    d_vm_dsyz = 3 * syz_r / sigma_vm_relaxed
    d_vm_dsxz = 3 * sxz_r / sigma_vm_relaxed
    d_vm_dsxy = 3 * sxy_r / sigma_vm_relaxed
    # Chain rule: d(sigma_vm_relaxed)/dE = sum_i d(sigma_vm_relaxed)/d(sigma_i) * d(sigma_i)/dE
    # For linear elasticity, stress is proportional to E, so d(sigma_i)/dE = stress_i / E
    d_vm_dE = (
        d_vm_dsxx * sxx +
        d_vm_dsyy * syy +
        d_vm_dszz * szz +
        d_vm_dsyz * syz +
        d_vm_dsxz * sxz +
        d_vm_dsxy * sxy
    ) * factor
    return d_vm_dE


def compute_volume_constraint_and_gradient(x: np.ndarray, volfracUpper: float) -> tuple:
    volConstraint = ((np.mean(x)/volfracUpper) - 1.0)
    volConstraint_gradient = np.ones_like(x) / volfracUpper/ x.size
    return volConstraint, volConstraint_gradient


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
            decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(2,-1).T)
            material_properties = matEncoder.getMaterialProperties(decoded)
            Ed = material_properties['Youngs_modulus_constant_term']
            Ec = material_properties['Youngs_modulus_linear_term']
            Eb= material_properties['Youngs_modulus_quadratic_term']
            Ea= material_properties['Youngs_modulus_cubic_term']
            T_torch = torch.tensor(T, dtype=Ea.dtype, device=Ea.device)
            E0 = to_params.E0 if hasattr(to_params, 'E0') else 100
            T0 = to_params.T0 if hasattr(to_params, 'T0') else 500
            EDesignTensor = (
                Ea * T_torch**3 * E0 / T0**3 +
                Eb * T_torch**2 * E0 / T0**2 +
                Ec * T_torch * E0 / T0 +
                Ed * E0
            )
            EDesign= EDesignTensor.detach().numpy()
            compliance = np.einsum('i, i -> ', fe_solver_structural.total_force, uvw)
            ce = (np.dot(uvw[fe_solver_structural.mesh.edofMat].reshape(num_elems, 24), KETemplate) * uvw[fe_solver_structural.mesh.edofMat].reshape(num_elems, 24)).sum(1)
            penal = 3.0
            dJ_dxDesign = (-penal * x ** (penal - 1)) * EDesign * ce
            dJ_dE = torch.tensor((x ** penal) * ce, dtype=Ea.dtype, device=Ea.device)
            E = EDesign
            zetaTensor.grad = None
            E.backward(dJ_dE, retain_graph=True)
            dJ_dzDesign = zetaTensor.grad[num_elems:].detach().numpy()
            grad_compliance = np.concatenate((dJ_dxDesign, -dJ_dzDesign.flatten()))
            complianceConstraint = ((compliance / constraintLimit) - 1.0)
            grad_complianceConstraint = grad_compliance / constraintLimit
            c[m, 0] = complianceConstraint
            dc[m, :] = grad_complianceConstraint

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
            decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(2,-1).T)
            material_properties = matEncoder.getMaterialProperties(decoded)
            Ed = material_properties['Youngs_modulus_constant_term']
            Ec = material_properties['Youngs_modulus_linear_term']
            Eb= material_properties['Youngs_modulus_quadratic_term']
            Ea= material_properties['Youngs_modulus_cubic_term']
            T_torch = torch.tensor(T, dtype=Ea.dtype, device=Ea.device)
            E0 = to_params.E0 if hasattr(to_params, 'E0') else 100
            T0 = to_params.T0 if hasattr(to_params, 'T0') else 500
            EDesigntensor = (
                Ea * T_torch**3 * E0 / T0**3 +
                Eb * T_torch**2 * E0 / T0**2 +
                Ec * T_torch * E0 / T0 +
                Ed * E0
            )
            Yd = material_properties['Yield_strength_constant_term']
            Yc = material_properties['Yield_strength_linear_term']
            Yb= material_properties['Yield_strength_quadratic_term']
            Ya= material_properties['Yield_strength_cubic_term']
            Y0 = to_params.Y0 if hasattr(to_params, 'Y0') else 100
            T0 = to_params.T0 if hasattr(to_params, 'T0') else 500
            YDesigntensor = (
                Ya * T_torch**3 * Y0 / T0**3 +
                Yb * T_torch**2 * Y0 / T0**2 +
                Yc * T_torch * Y0 / T0 +
                Yd * Y0
            )
            EDesign= EDesigntensor.detach().numpy()
            YDesign= YDesigntensor.detach().numpy()
            DensDesign = mass_density.detach().numpy()
            
            vm_max = np.max(fe_solver_structural.vonMisesStress)
            vm_min = np.min(fe_solver_structural.vonMisesStress)
            print("Max von Mises stress:", vm_max)
            print("Min von Mises stress:", vm_min)
            inv_sf_pnorm, grad_inv_sf_density = compute_pnorm_safety_factor_and_sensitivity(
                uvw, x, fe_solver_structural,EDesign,YDesign, KETemplate, MaterialModel.SIMP,
                p=to_params.PNormExponent
            )
            print("P-norm of inv safety factor:", inv_sf_pnorm)
            print("Grad inv sf density min:", grad_inv_sf_density.min(), "max:", grad_inv_sf_density.max())
            # Safety factor constraint value
            safety_factor = constraintLimit
            safety_constraint = inv_sf_pnorm - (1.0 / safety_factor)
            c[m, 0] = safety_constraint

        
            # 2. Compute latent variable part of gradient (chain rule)
            p = to_params.PNormExponent
            d_sigma_vm_dE = np.zeros(num_elems)
            for e in range(num_elems):
                # Divide by decoded youngs modulus for that element
                d_sigma_vm_dE[e] = d_relaxed_von_mises_dE(
                    fe_solver_structural.stressComponents[e], x[e].item(), q=1) / EDesigntensor[e].item()
            # Get per-element von Mises and yield strength
            sigma_vm = np.zeros(num_elems)
            for e in range(num_elems):
                stress = fe_solver_structural.stressComponents[e]
                sxx, syy, szz, syz, sxz, sxy = stress
                sigma_vm[e] = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) +
                    3 * (syz ** 2 + sxz ** 2 + sxy ** 2)) * (zetaTensor[0:num_elems][e].item() ** 1)
            Y = np.array([mat.yield_strength for mat in fe_solver_structural.mat_prop])
            S = sigma_vm
            inv_sf_elem = S / Y
            sum_p = np.sum(inv_sf_elem ** p)
            outer = (sum_p) ** (1.0 / p - 1)
            grad_z = np.zeros(2*num_elems)
            # Backward for dE/dz and dY/dz
            zetaTensor.grad = None
            EDesigntensor.backward(torch.ones_like(EDesigntensor), retain_graph=True)
            dE_dz = zetaTensor.grad[num_elems:].detach().numpy().reshape(num_elems, -1)
            zetaTensor.grad = None
            YDesigntensor.backward(torch.ones_like(YDesigntensor), retain_graph=True)
            dY_dz = zetaTensor.grad[num_elems:].detach().numpy().reshape(num_elems, -1)
            zetaTensor.grad = None
            for e in range(num_elems):
                d_sigma_dz = d_sigma_vm_dE[e] * dE_dz[e]
                dYdz = dY_dz[e]
                bracket = (d_sigma_dz * Y[e] - dYdz * S[e]) / (Y[e] ** 2) 
                grad_z[0:num_elems] += p * (inv_sf_elem[e] ** (p - 1)) * bracket[0]
                grad_z[num_elems:] += p * (inv_sf_elem[e] ** (p - 1)) * bracket[1]
            grad_z = (1.0 / p) * outer * grad_z

            # 3. Assemble full gradient for constraint
            grad_safety = np.zeros_like(zeta)
            grad_safety[:num_elems] = grad_inv_sf_density
            grad_safety[num_elems:] = grad_z
            dc[m, :] = grad_safety

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

        else:
            raise NotImplementedError(f"Constraint {constraintType} is not implemented yet.")

    return c, dc