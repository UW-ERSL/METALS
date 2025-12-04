import enum
from Thermostructural_examples import *
from PyTOImports import  *
from dataclasses import dataclass, field
material_colors = { # these colors can be changed as desired for each example
            0: "#080878",  # medium blue
            1: '#004d00',  # deep forest green
            2: '#800000',  # dark maroon
            3: '#e6194b', # vibrant red
            4: '#b8860b',  # dark goldenrod
            5: '#1b1b1b',  # charcoal black
            6: "#49c878", # vibrant magenta
            7: "#28a3dc",  # bright sky blue
            8: '#6a5acd',  # slate blue
            9: "#a35f5f", # brownish
            10: '#9932cc', # dark orchid
            11: '#228b22',  # forest green
            12: '#ffb6c1',  # light pink
            13: "#e76f5d", # vibrant green
            14: '#ffe119', # vibrant yellow
            15: '#008080',  # teal
            16: '#f58231', # vibrant orange
            17: '#911eb4', # vibrant purple
            18: '#42d4f4', # vibrant cyan
            19: "#505053",  # dim gray
        }
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
		thermostructural_problem = ThermoStructuralExamples.BiClamp 
		kwargs['structural_load'] = 1e5
		kwargs['TWall'] = 24 # 23 is the reference temperature
		to_params.Comment = "Thermo-structural BiClamp example"
		to_params.XSymmetry = True
		to_params.ExtrudeZ = True
		to_params.RelativeFilterRadius = 1.5
		to_params.nDOFDesired = 25000
		to_params.Objective = (TO_QOI.COMPLIANCE, None)
		to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.25)]
		to_params.MaterialsExcelFile = './DataConstantTemperature/3MaterialsCTE.xlsx'
	elif to_problem == MMTOThermostructuralExamples.MBBBeam:
		# See paper: "Compliance‑based topology optimization of structural components
		# subjected to thermo‑mechanical loading", by Ooms, et al., 2023
		print("Creating Thermo-structural MBB Beam problem...")
		thermostructural_problem = ThermoStructuralExamples.MBBBeam 
		kwargs['structural_load'] = 10000
		kwargs['Ta'] = 23  # Ambient temperature
		kwargs['Tf'] = 73 # Base temperature
		to_params.Comment = "Thermo-structural MBB Beam example"
		to_params.ExtrudeZ = True
		to_params.nDOFDesired = 25000
		to_params.Objective = (TO_QOI.COMPLIANCE, None)
		to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.4)]
		to_params.MaterialsExcelFile = './DataConstantTemperature/3MaterialsCTE.xlsx'
	else:
		raise ValueError("Invalid Thermo-structural Topology Optimization problem specified.")
	mesh, mat_prop, bcStructural,bcThermal, elem_body_force = getThermoStructuralProblem(thermostructural_problem, **kwargs)

	return mesh, mat_prop, bcStructural,bcThermal, elem_body_force, to_params,vae_params