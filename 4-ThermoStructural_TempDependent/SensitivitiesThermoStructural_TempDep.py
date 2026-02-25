import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../0-Common')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../1-PureStructural')))

from PyTOImports import bound_cond, linear_solvers, TO_QOI, get_pNorm_exponent, MaterialModel  # type: ignore

from SensitivitiesPureStructural import (  # type: ignore
    compute_pnorm_safety_factor_and_sensitivity,
    compute_pnorm_stress_and_sensitivity,
    compute_volumefraction_constraint_and_gradient
)

from topopt_material_model import (
    get_thermal_material_model_scaling,
    get_structural_material_model_scaling,
    get_structural_material_model_sensitivity,
    get_thermal_material_model_sensitivity
)  # type: ignore


# -----------------------------
# Helpers
# -----------------------------
def _compute_elem_average_temperature(fe_solver_thermal, temperature):
    """
    temperature: nodal temperature array (num_nodes,)
    returns Telem: (nelem,)
    """
    T_e = temperature[fe_solver_thermal.mesh.edofMatThermal]  # (nelem, 8)
    return T_e.mean(axis=1)


# -----------------------------
# TEMP-DEPENDENT THERMAL ADJOINT
# -----------------------------
def solve_thermal_adjoint_multimaterial_tempdep(
    x, displacement, temperature, Telem,
    fe_thermal_solver, fe_structural_solver,
    E, alpha, dE_dT, dAlpha_dT,
    KSTemplate_unitE,
    material_model
):
    """
    Thermal adjoint for STRAIN ENERGY objective:
        J = u^T K_S u

    TEMP-DEPENDENT properties E(Tbar), alpha(Tbar).

    We use Telem = average element temperature (Tbar).
    Properties are evaluated at Telem, and dE_dT, dAlpha_dT are w.r.t. physical temperature.

    Thermal adjoint equation:
        K_T * lambda_T = (∂R_S/∂T)^T * lambda_S
    with lambda_S = 2u for J = u^T K u.

    Compared to temp-independent RHS:
        rhs_e = -2*sS*E*alpha*H^T u_e

    We add extra terms due to:
        g(T)=E(T)*alpha(T)   and   K_S depends on E(T)

    Final RHS per element (8-vector):
        rhs_e = -2*sS * [ g * H^T u_e  +  (1/8)*g_T*(u_e^T H (Tdiff))*1 ]
               -      sS * (1/8)*E_T*(u_e^T K0 u_e)*1

    where:
        g_T = d(E*alpha)/dT = alpha*dE_dT + E*dAlpha_dT
        K0 = KSTemplate_unitE (unit-E stiffness template for that element)
        1 is a vector of ones length 8 distributing d/dTelem to nodal temps
    """
    nelem = fe_thermal_solver.mesh.num_elems
    num_thermal_dofs = fe_thermal_solver.mesh.num_nodes

    rhs = np.zeros(num_thermal_dofs)

    dx, dy, dz = fe_structural_solver.mesh.elem_size
    Tref = fe_thermal_solver.thermoElasticReferenceTemperature

    # Poisson ratio per element
    if isinstance(fe_structural_solver.mat_prop, list):
        nu = np.array([fe_structural_solver.mat_prop[i].poissons_ratio for i in range(nelem)])
    else:
        nu = np.full(nelem, fe_structural_solver.mat_prop.poissons_ratio)

    ones8 = np.ones(8, dtype=float)

    for e in range(nelem):
        edof_s = fe_structural_solver.mesh.edofMatStructural[e, :]
        edof_t = fe_thermal_solver.mesh.edofMatThermal[e, :]

        u_e = displacement[edof_s]     # (24,)
        T_e = temperature[edof_t]      # (8,)
        Tdiff = T_e - Tref

        H_e = fe_thermal_solver.getHMatrix(dx, dy, dz, nu[e])  # (24,8)

        sS = get_structural_material_model_scaling(x[e], material_model)

        g = E[e] * alpha[e]
        gT = alpha[e] * dE_dT[e] + E[e] * dAlpha_dT[e]  # d(E*alpha)/dT at Telem[e]

        # scalar: u^T H (T - Tref)
        uH_T = float(u_e.T @ (H_e @ Tdiff))

        # scalar: u^T K0 u   (K0 is unit-E)
        uK0u = float(u_e.T @ (KSTemplate_unitE @ u_e))

        # Base (temp-independent) term:
        rhs_e = -2.0 * sS * (g * (H_e.T @ u_e))

        # NEW: g(Telem) dependence distributed equally to 8 nodes:
        rhs_e += -2.0 * sS * ( (gT * (1.0/8.0) * uH_T) * ones8 )

        # NEW: stiffness E(Telem) dependence distributed equally:
        rhs_e += -(sS * (1.0/8.0) * dE_dT[e] * uK0u) * ones8

        rhs[edof_t] += rhs_e

    # Solve K_T * lambda_T = rhs
    K_T = fe_thermal_solver.stiff_mtrx
    bcAdjoint = bound_cond.BC(
        force=np.zeros_like(fe_thermal_solver.bc.force),
        fixed_dofs=fe_thermal_solver.bc.fixed_dofs,
        dirichlet_values=np.zeros_like(fe_thermal_solver.bc.dirichlet_values)
    )

    lambda_T = linear_solvers.solve(
        K_T, rhs, fe_thermal_solver.solver, bcAdjoint, **fe_thermal_solver.kwargs
    )
    return lambda_T


# -----------------------------
# DENSITY GRADIENT (TEMP-DEPENDENT)
# -----------------------------
def compute_thermoelastic_compliance_and_gradient_density_multimaterial_tempdep(
    x, temperature, Telem, displacement, to_params,
    fe_thermal_solver, fe_structural_solver,
    E, alpha, K,
    dE_dT, dAlpha_dT,
    KSTemplate_unitE, KTTemplate_unitK
):
    """
    Temperature-dependent thermoelastic strain-energy objective:
      J = u^T K u

    Terms 1-3 are structurally same as your temp-independent version
    except E, alpha, K are evaluated at Telem.

    The only change: lambda_T must be computed with the temp-dependent RHS.
    """
    material_model = to_params.materialModel

    J = displacement.T @ fe_structural_solver.stiff_mtrx @ displacement

    nelem = fe_structural_solver.mesh.num_elems
    dJ_dx = np.zeros(nelem)

    lambda_T = solve_thermal_adjoint_multimaterial_tempdep(
        x=x,
        displacement=displacement,
        temperature=temperature,
        Telem=Telem,
        fe_thermal_solver=fe_thermal_solver,
        fe_structural_solver=fe_structural_solver,
        E=E,
        alpha=alpha,
        dE_dT=dE_dT,
        dAlpha_dT=dAlpha_dT,
        KSTemplate_unitE=KSTemplate_unitE,
        material_model=material_model
    )

    # Poisson ratio per element
    if isinstance(fe_structural_solver.mat_prop, list):
        nu = np.array([fe_structural_solver.mat_prop[i].poissons_ratio for i in range(nelem)])
    else:
        nu = np.full(nelem, fe_structural_solver.mat_prop.poissons_ratio)

    dx, dy, dz = fe_structural_solver.mesh.elem_size
    Tref = fe_thermal_solver.thermoElasticReferenceTemperature

    for e in range(nelem):
        edof_s = fe_structural_solver.mesh.edofMatStructural[e, :]
        edof_t = fe_thermal_solver.mesh.edofMatThermal[e, :]

        u_e = displacement[edof_s]
        T_e = temperature[edof_t]
        lamT_e = lambda_T[edof_t]

        H_e = fe_thermal_solver.getHMatrix(dx, dy, dz, nu[e])
        Tdiff = T_e - Tref

        # Term 1: stiffness via x
        term1 = -get_structural_material_model_sensitivity(x[e], material_model) * (u_e.T @ (E[e] * KSTemplate_unitE) @ u_e)

        # Term 2: thermal force via x
        term2 = 2.0 * get_structural_material_model_sensitivity(x[e], material_model) * E[e] * alpha[e] * (u_e.T @ H_e @ Tdiff)

        # Term 3: thermal conduction adjoint via x (matches your original structure)
        term3 = get_thermal_material_model_sensitivity(x[e], material_model) * (lamT_e.T @ (K[e] * KTTemplate_unitK) @ T_e)

        dJ_dx[e] = term1 + term2 + term3

    return J, dJ_dx, lambda_T


# -----------------------------
# LATENT GRADIENT (TEMP-DEPENDENT)
# -----------------------------
def compute_thermoelastic_compliance_and_gradient_latent_multimaterial_tempdep(
    zeta, temperature, Telem, displacement, to_params,
    fe_thermal_solver, fe_structural_solver,
    E, alpha, K,
    dE_dz, dAlpha_dz, dK_dz,
    dE_dT, dAlpha_dT,
    KSTemplate_unitE, KTTemplate_unitK,
    material_model,
    lambda_T=None
):
    """
    Temperature-dependent thermoelastic latent gradients for strain-energy objective.
    Uses your verified term1-term4 structure, with E(Telem), alpha(Telem), K(Telem).

    IMPORTANT:
      - dE_dz, dAlpha_dz, dK_dz must be computed at the same Telem.
      - lambda_T is computed with temp-dependent adjoint RHS.
    """
    nelem = fe_structural_solver.mesh.num_elems
    x = zeta[:nelem]
    latent_dim = dE_dz.shape[0]

    J = displacement.T @ fe_structural_solver.stiff_mtrx @ displacement

    if lambda_T is None:
        lambda_T = solve_thermal_adjoint_multimaterial_tempdep(
            x=x,
            displacement=displacement,
            temperature=temperature,
            Telem=Telem,
            fe_thermal_solver=fe_thermal_solver,
            fe_structural_solver=fe_structural_solver,
            E=E,
            alpha=alpha,
            dE_dT=dE_dT,
            dAlpha_dT=dAlpha_dT,
            KSTemplate_unitE=KSTemplate_unitE,
            material_model=material_model
        )

    dJ_dz = np.zeros((nelem, latent_dim))

    # Poisson ratio per element
    if isinstance(fe_structural_solver.mat_prop, list):
        nu = np.array([fe_structural_solver.mat_prop[i].poissons_ratio for i in range(nelem)])
    else:
        nu = np.full(nelem, fe_structural_solver.mat_prop.poissons_ratio)

    dx, dy, dz = fe_structural_solver.mesh.elem_size
    Tref = fe_thermal_solver.thermoElasticReferenceTemperature

    for e in range(nelem):
        edof_s = fe_structural_solver.mesh.edofMatStructural[e, :]
        edof_t = fe_thermal_solver.mesh.edofMatThermal[e, :]

        u_e = displacement[edof_s]
        T_e = temperature[edof_t]
        lamT_e = lambda_T[edof_t]

        Tdiff = T_e - Tref
        H_e = fe_thermal_solver.getHMatrix(dx, dy, dz, nu[e])

        dE_dz_e = dE_dz[:, e]
        dA_dz_e = dAlpha_dz[:, e]
        dK_dz_e = dK_dz[:, e]

        sS = get_structural_material_model_scaling(x[e], material_model)
        sT = get_thermal_material_model_scaling(x[e], material_model)

        # Term 1: stiffness via E(z)
        term1 = -sS * dE_dz_e * (u_e.T @ KSTemplate_unitE @ u_e)

        # Term 2: thermal force via E(z)
        term2 = 2.0 * sS * dE_dz_e * alpha[e] * (u_e.T @ H_e @ Tdiff)

        # Term 3: thermal force via alpha(z)
        term3 = 2.0 * sS * E[e] * dA_dz_e * (u_e.T @ H_e @ Tdiff)

        # Term 4: thermal conduction via K(z)
        term4 = sT * dK_dz_e * (lamT_e.T @ (KTTemplate_unitK @ T_e))

        dJ_dz[e, :] = term1 + term2 + term3 + term4

    return J, dJ_dz.T.flatten(), lambda_T


# -----------------------------
# MAIN objective/gradient wrapper (TEMP-DEPENDENT)
# -----------------------------
def compute_mmto_objective_and_gradient_tempdep(
    to_params,
    displacement,
    temperature,
    Telem,
    zeta,
    fe_solver_structural,
    KETemplate_unitE,
    KTTemplate_unitK,
    matEncoder,
    fe_solver_thermal
):
    """
    TEMP-DEPENDENT version of compute_mmto_objective_and_gradient.

    Uses Telem to evaluate E(T), Alpha(T), K(T) and their derivatives.

    Expects new Excel with columns:
      Density, E0..E3, Y0..Y3, Alpha0..Alpha3, K0..K3
    """
    objectiveType = to_params.Objective[0]
    num_elems = fe_solver_structural.mesh.num_elems
    x = zeta[0:num_elems]
    latentDim = matEncoder.vae_params.latentDim
    zPts = zeta[num_elems:].reshape((latentDim, -1)).T  # (nelem, latentDim)

    material_model = to_params.materialModel
    pNormExponent = get_pNorm_exponent()

    # --- temp-dependent properties + grads (at Telem) ---
    E, dE_dz, dE_dT = matEncoder.getAttribute_value_dz_dT_atTemperature("E", zPts, Telem)
    alpha, dA_dz, dA_dT = matEncoder.getAttribute_value_dz_dT_atTemperature("Alpha", zPts, Telem)
    K, dK_dz, dK_dT = matEncoder.getAttribute_value_dz_dT_atTemperature("K", zPts, Telem)
    # DEBUG toggle: disable temperature-derivative terms in adjoint
    # dE_dT[:] = 0.0
    # dA_dT[:] = 0.0
    if objectiveType == TO_QOI.COMPLIANCE:
        J, dJdx, lambda_T = compute_thermoelastic_compliance_and_gradient_density_multimaterial_tempdep(
            x=x,
            temperature=temperature,
            Telem=Telem,
            displacement=displacement,
            to_params=to_params,
            fe_thermal_solver=fe_solver_thermal,
            fe_structural_solver=fe_solver_structural,
            E=E,
            alpha=alpha,
            K=K,
            dE_dT=dE_dT,
            dAlpha_dT=dA_dT,
            KSTemplate_unitE=KETemplate_unitE,
            KTTemplate_unitK=KTTemplate_unitK
        )

        _, dJdz, _ = compute_thermoelastic_compliance_and_gradient_latent_multimaterial_tempdep(
            zeta=zeta,
            temperature=temperature,
            Telem=Telem,
            displacement=displacement,
            to_params=to_params,
            fe_thermal_solver=fe_solver_thermal,
            fe_structural_solver=fe_solver_structural,
            E=E,
            alpha=alpha,
            K=K,
            dE_dz=dE_dz,
            dAlpha_dz=dA_dz,
            dK_dz=dK_dz,
            dE_dT=dE_dT,
            dAlpha_dT=dA_dT,
            KSTemplate_unitE=KETemplate_unitE,
            KTTemplate_unitK=KTTemplate_unitK,
            material_model=material_model,
            lambda_T=lambda_T
        )

        grad_obj = np.concatenate([dJdx, dJdz])
        return J, grad_obj

    # You said you only need compliance + mass/vol now, so we keep other objective types out of this file.
    raise NotImplementedError(f"Temp-dependent objective {objectiveType} not implemented here.")


# -----------------------------
# CONSTRAINTS (keep only VOLUME/MASS robustly)
# -----------------------------
def compute_mmto_constraint_and_gradient(
    to_params, sol, zeta, fe_solver, KETemplate_unitE, matEncoder
):
    """
    TEMP-DEP safe constraint evaluator.

    Implements:
      - VOLUME_FRACTION
      - MASS (uses density as temp-independent attribute from decoder)

    Other constraints are left unimplemented here because your new Excel
    no longer has 'Youngs_Modulus', 'Yield_Strength', etc. as direct columns.
    """
    nConstraints = len(to_params.Constraints)
    num_elems = fe_solver.mesh.num_elems
    x = zeta[0:num_elems]
    latentDim = matEncoder.vae_params.latentDim
    zPts = zeta[num_elems:].reshape((latentDim, -1)).T

    # Get decoded properties + gradients for non-temp-dependent attributes (Density, Cost, etc. if present)
    material_properties, gradients = matEncoder.getMaterialPropertiesAtLatentPoints(zPts, compute_gradients=True)

    mass_density = None
    dMassDensity_dz = None
    if 'Density' in material_properties:
        mass_density = material_properties['Density'].detach().cpu().numpy()
        dMassDensity_dz = gradients['Density'].detach().cpu().numpy()  # (nelem, latentDim)

    c = np.zeros((nConstraints, 1))
    dc = np.zeros((nConstraints, zeta.size))

    for m in range(nConstraints):
        constraintType = to_params.Constraints[m][0]
        constraintLimit = to_params.Constraints[m][2]

        if constraintType == TO_QOI.VOLUME_FRACTION:
            volfracConstraint, volfracConstraint_gradient = compute_volumefraction_constraint_and_gradient(x, constraintLimit)
            grad = np.zeros_like(zeta)
            grad[0:num_elems] = volfracConstraint_gradient
            c[m, 0] = volfracConstraint
            dc[m, :] = grad

        elif constraintType == TO_QOI.MASS:
            if mass_density is None:
                raise RuntimeError("Density not present in the materials Excel but MASS constraint requested.")

            elemVolume = fe_solver.mesh.elem_size[0] * fe_solver.mesh.elem_size[1] * fe_solver.mesh.elem_size[2]
            totalMass = (mass_density * x).sum() * elemVolume

            grad_mass = np.zeros_like(zeta)
            grad_mass[0:num_elems] = mass_density * elemVolume
            # d rho / dz (nelem, latentDim) -> flatten Fortran order
            grad_mass[num_elems:] = (dMassDensity_dz.T * x).flatten(order='F') * elemVolume

            c[m, 0] = totalMass / constraintLimit - 1.0
            dc[m, :] = grad_mass / constraintLimit

        else:
            raise NotImplementedError(
                f"Constraint {constraintType} not implemented in temp-dependent module. "
                "Use VOLUME_FRACTION or MASS for now."
            )

    return c, dc