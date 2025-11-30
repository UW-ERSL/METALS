import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../PyTO/src')))
import stl_reader, hex_mesher,topopt_filters, bound_cond, mat_lib, hex_structural_fea, hex_element_stiffness, hex_thermal_fea, linear_solvers, deflation # type: ignore[reportMissingImports]
from topopt_common import SIMP_STRUCTURAL_PENALTY,SIMP_THERMAL_PENALTY,PNORM_EXPONENT, SIMP_STRESS_RELAXATION, TOParams, TO_QOI, find_elements_with_fixedDOF,createFilters, find_elements_with_forces # type: ignore[reportMissingImports]
from mmaWrapper import  runMMA # type: ignore[reportMissingImports]
from topopt_material_model import MaterialModel, get_structural_material_model_sensitivity, get_structural_material_model_scaling # type: ignore[reportMissingImports]