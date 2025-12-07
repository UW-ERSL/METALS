import enum
import sys
import os
import numpy as np
# Add the 0-Common folder to Python path
common_path = os.path.join(os.path.dirname(__file__), '..', '0-Common')
sys.path.insert(0, os.path.abspath(common_path))

# Now you can import normally
from ThermoStructuralFEAExamples import ThermoStructuralFEAExamples, getThermoStructuralFEAProblem # type: ignore
from PyTOImports import  TOParams, TO_QOI, find_elements_with_fixedDOF # type: ignore
from dataclasses import dataclass, field

@dataclass(slots=True) # avoid accidental modification
class VAEParams: #default VAE parameters
    klFactor: float = 5e-6
    learningRate: float = 2e-6
    numEpochs: int = 150000 # max epochs
    vae_hiddenDim: int = 500
    latentDim: int = 2 # default latent dimension
    maxAttributeErrorPercent: float = 0.001 # termination criteria for VAE training
# The actual implementations are in topopt_structural_benchmarks.py and topopt_thermal_benchmarks.py
class MMTOThermostructuralExamples(enum.Enum):
	BiClamp = enum.auto()
	MBBBeam = enum.auto()


def getMMTOThermostructuralProblem(to_problem: MMTOThermostructuralExamples, **kwargs):
	to_params = TOParams()
	vae_params = VAEParams()
	if to_problem == MMTOThermostructuralExamples.BiClamp:
		print("Creating Thermo-structural BiClamp problem...")
		thermostructural_problem = ThermoStructuralFEAExamples.BiClamp 
		kwargs['structural_load'] = 1e5
		kwargs['TWall'] = 30 # 23 is the reference temperature
		to_params.Comment = "Thermo-structural BiClamp example"
		to_params.XSymmetry = True
		to_params.ExtrudeZ = True
		to_params.RelativeFilterRadius = 1.5
		to_params.nDOFDesired = 25000
		to_params.Objective = (TO_QOI.COMPLIANCE, None)
		to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.25)]
		to_params.MaterialsExcelFile = './2-ThermoStructural/MaterialDataThermoStructural/3Materials.xlsx'
	elif to_problem == MMTOThermostructuralExamples.MBBBeam:
		# See paper: "Compliance‑based topology optimization of structural components
		# subjected to thermo‑mechanical loading", by Ooms, et al., 2023
		print("Creating Thermo-structural MBB Beam problem...")
		thermostructural_problem = ThermoStructuralFEAExamples.MBBBeam 
		kwargs['structural_load'] = 1e4
		kwargs['Ta'] = 23  # Ambient temperature
		kwargs['Tf'] = 223 # Base temperature
		to_params.Comment = "Thermo-structural MBB Beam example"
		to_params.ExtrudeZ = True
		to_params.nDOFDesired = 25000
		to_params.Objective = (TO_QOI.COMPLIANCE, None)
		to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.4)]
		to_params.MaterialsExcelFile = './2-ThermoStructural/MaterialDataThermoStructural/3Materials.xlsx'
	else:
		raise ValueError("Invalid Thermo-structural Topology Optimization problem specified.")
	mesh, mat_prop, bcStructural,bcThermal, elem_body_force = getThermoStructuralFEAProblem(thermostructural_problem, **kwargs)

	return mesh, mat_prop, bcStructural,bcThermal, elem_body_force, to_params,vae_params