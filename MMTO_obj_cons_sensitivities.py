import numpy as np
import torch
from PyTOImports import *
# --- Support Functions ---

def compute_pnorm_safety_factor_and_sensitivity(sol: np.ndarray, x, fe_solver, KE, material_model, p):
    """
    Compute p-norm of (von Mises stress / yield strength) and its sensitivity for multi-material case.
    """
    mesh = fe_solver.mesh
    nelems = mesh.num_elems
    q = 1  # STRESS_RELAXATION factor

    # Handle multi-material: get yield strength for each element
    if isinstance(fe_solver.mat_prop, list):
        yield_strengths = np.array([fe_solver.mat_prop[i].yield_strength for i in range(nelems)])
        E = np.array([fe_solver.mat_prop[i].youngs_modulus for i in range(nelems)])
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
    nRows = KE.shape[0]
    ce = ( np.dot(adjointSol[dofMat].reshape(num_elems, nRows), KE)
        * sol[dofMat].reshape(num_elems, nRows)).sum(1)

    T2 = get_structural_material_model_sensitivity(x, material_model) * ce / yield_strengths #KS: Should divide by yield_strengths?
    inv_sf_pnorm_sensitivity = T1  + T2

    return inv_sf_pnorm, inv_sf_pnorm_sensitivity

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


    if objectiveType == TO_QOI.COMPLIANCE:
        decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(2,-1).T)
        material_properties = matEncoder.getMaterialProperties(decoded)
        youngsModulus = material_properties['Youngs_Modulus']
        EDesign = youngsModulus.detach().numpy()
        compliance = np.einsum('i, i -> ', fe_solver.total_force, sol)
        ce = (np.dot(sol[fe_solver.mesh.edofMat].reshape(num_elems, 24), KETemplate) * sol[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
        penal = 3.0
        dJ_dxDesign = (-penal * x ** (penal - 1)) * EDesign * ce
        dJ_dEDesign = np.asarray((x ** penal) * ce)
        dJ_dEDesign_tensor = torch.tensor(dJ_dEDesign)
        youngsModulus.backward(dJ_dEDesign_tensor)
        dJ_dzeta = zetaTensor.grad.detach().numpy()
        grad_compliance = np.concatenate((dJ_dxDesign, -dJ_dzeta[num_elems:].flatten()))
        return compliance, grad_compliance
    
    elif objectiveType == TO_QOI.VOLUME_FRACTION:
        pass

    elif objectiveType == TO_QOI.PNORM_STRESS:
       pass

    elif objectiveType == TO_QOI.MASS:
        # Mass objective: sum of density * mass_density * element volume
        pass
    else:
        raise NotImplementedError(f"Objective {objectiveType} is not implemented yet.")
    
   
def compute_mmto_constraint_and_gradient(to_params, sol, zeta, fe_solver, KE, matEncoder):
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
            elemVolume =  fe_solver.mesh.elem_size[0] * fe_solver.mesh.elem_size[1] * fe_solver.mesh.elem_size[2]
            totalMass = torch.einsum('m,m->m', mass_density, pseudodensity).sum()*elemVolume 
            massConstraint = ((totalMass / constraintLimit) - 1.0)
            massConstraint.backward(retain_graph=True)
            cons_mass = massConstraint.detach().numpy()
            grad_cons_mass = zetaTensor.grad.detach().numpy()
            c[m, 0] = cons_mass
            dc[m, :] = grad_cons_mass
        elif constraintType == TO_QOI.PNORM_STRESS or constraintType == TO_QOI.STRESS_SAFETY_FACTOR:
           pass

        elif constraintType == TO_QOI.COST:
            # Cost constraint: sum of density * mass_density * cost * element volume
            decoded = matEncoder.vaeNet.decoder(zetaTensor[num_elems:].view(2,-1).T)
            mass_density = matEncoder.getMaterialProperties(decoded)['Density']
            costperunitmass = matEncoder.getMaterialProperties(decoded)['Cost']
            pseudodensity = zetaTensor[0:fe_solver.mesh.num_elems]
            elemVolume = fe_solver.mesh.elem_size[0] * fe_solver.mesh.elem_size[1] * fe_solver.mesh.elem_size[2]
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