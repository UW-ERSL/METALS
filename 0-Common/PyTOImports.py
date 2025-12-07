import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../PyTO/src')))
import stl_reader, hex_mesher,topopt_filters, bound_cond, mat_lib, hex_structural_fea, hex_element_stiffness, hex_thermal_fea, linear_solvers, deflation # type: ignore[reportMissingImports]

from topopt_common import createFilters, TO_QOI, TOParams, find_elements_with_fixedDOF # type: ignore[reportMissingImports]
from mmaWrapper import  runMMA # type: ignore[reportMissingImports]
from topopt_material_model import MaterialModel, get_stress_relaxation_correction, get_stress_relaxation_factor_sensitivity, get_pNorm_exponent, get_structural_material_model_sensitivity, get_structural_material_model_scaling, get_thermal_material_model_sensitivity, get_thermal_material_model_scaling    # type: ignore[reportMissingImports]
from topopt_material_model import initialize_SIMP_STRUCTURAL_PENALTY, initialize_SIMP_THERMAL_PENALTY # type: ignore[reportMissingImports]
from topopt_material_model import   increment_SIMP_THERMAL_PENALTY, increment_SIMP_STRUCTURAL_PENALTY # type: ignore[reportMissingImports]