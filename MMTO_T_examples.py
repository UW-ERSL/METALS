import enum
from Structural_examples import *
from Thermal_examples import *
from PyTOImports import  *

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

class VAEParams:
    klFactor = 5e-6
    learningRate = 2e-6
    numEpochs = 150000 # max epochs
    vae_hiddenDim = 500
    latentDim = 2 # hard coded for now
    maxAttributeErrorPercent = 0.001 # termination criteria for VAE training

class MMTOTempDependentExamples(enum.Enum):
    LBracket_Compliance_Mass = enum.auto()
    LBracket_Compliance_MassCost = enum.auto()
    LBracket_Stress_MassVolumeTemp = enum.auto()
    LBracket_Stress_MassCompliance = enum.auto()
    LBracket_Stress_MultipleConstraints = enum.auto()
    LBracket_Mass_StressFF_Compliance = enum.auto()    
    LBracket_Mass_MultipleConstraints = enum.auto()    


    BliskSection_Compliance_MassCost = enum.auto()
    BliskSection_Mass_StressFF = enum.auto()
    BliskSection_Mass_MultipleConstraints = enum.auto()
    BliskSection_Stress_MassComplianceCriticality = enum.auto()
    BliskSection_Compliance_Mass = enum.auto()
    BliskSection_Test = enum.auto()


def getMMTOTempDependentProblem(to_problem: MMTOTempDependentExamples,nDOFDesired = None, **kwargs):
    """Get the structural topology optimization problem based on the specified example.

    Args:
        problem: The example problem to solve.
        **kwargs: Additional arguments to pass to the problem.

    Returns:
        StructuralTOProblem: The structural topology optimization problem.
    """
    
    to_params = TOParams()
    vae_params = VAEParams()
    if to_problem == MMTOTempDependentExamples.LBracket_Compliance_Mass:
        structural_problem=MMTOStructuralExamples.LBracket
        thermal_problem=MMTOThermalExamples.LBracketThermal
        kwargs['topload'] = 1e4 
        kwargs['midload'] = 0
        to_params.Comment  = "Thermal + Structural TO Problem"
        to_params.MaterialsExcelFile = './DataVaryingTemperature/3Materials.xlsx'
        to_params.Objective=(TO_QOI.COMPLIANCE, None)
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 10000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints=[(TO_QOI.MASS, None, 25)]
        vae_params.latentDim = 6
        vae_params.learningRate = 2e-5
        vae_params.vae_hiddenDim = 500
        vae_params.numEpochs = 200000
    elif to_problem == MMTOTempDependentExamples.LBracket_Compliance_MassCost:
        structural_problem=MMTOStructuralExamples.LBracket
        thermal_problem=MMTOThermalExamples.LBracketThermal
        kwargs['topload'] = 1e4 
        kwargs['midload'] = 0
        to_params.Comment  = "Thermal + Structural TO Problem"
        to_params.MaterialsExcelFile = './DataVaryingTemperature/3Materials.xlsx'
        to_params.Objective=(TO_QOI.COMPLIANCE, None)
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 10000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints=[(TO_QOI.MASS, None, 25), (TO_QOI.COST, None, 200)]
        vae_params.latentDim = 6
        vae_params.learningRate = 2e-5
        vae_params.vae_hiddenDim = 500
        vae_params.numEpochs = 200000        
    elif to_problem == MMTOTempDependentExamples.LBracket_Stress_MassVolumeTemp:
        structural_problem=MMTOStructuralExamples.LBracket
        thermal_problem=MMTOThermalExamples.LBracketThermal
        kwargs['topload'] = 1e4 
        kwargs['midload'] = 0
        to_params.Comment  = "Thermal + Structural TO Problem"
        to_params.MaterialsExcelFile = './DataVaryingTemperature/LSR_20251119_all_materials_2.xlsx'
        to_params.Objective=(TO_QOI.PNORM_STRESS, None)
        to_params.ExtrudeZ = True
        to_params.RelativeFilterRadius = 1.5
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints=[ (TO_QOI.VOLUME_FRACTION, None, 0.4), (TO_QOI.MASS, None, 30),
                               (TO_QOI.TEMPERATURE_FAILURE_FACTOR, None, 1)]
        vae_params.latentDim = 6
        vae_params.learningRate = 2e-5
        vae_params.vae_hiddenDim = 500
        vae_params.numEpochs = 200000
    elif to_problem == MMTOTempDependentExamples.LBracket_Mass_StressFF_Compliance:
        structural_problem=MMTOStructuralExamples.LBracket
        thermal_problem=MMTOThermalExamples.LBracketThermal
        Force = 500
        kwargs['topload'] = Force
        kwargs['midload'] = 0
        to_params.Comment  = "Thermal + Structural TO Problem"
        to_params.MaterialsExcelFile = './DataVaryingTemperature/3Materials.xlsx'
        to_params.Objective=(TO_QOI.MASS, None)
        to_params.ExtrudeZ = True
        to_params.Eliminate_Hanging_Elements = False
        to_params.nDOFDesired =25000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints=[ (TO_QOI.STRESS_FAILURE_FACTOR, None, 1),
                               (TO_QOI.COMPLIANCE, None, 0.0001*Force)]
        vae_params.latentDim = 6
        vae_params.learningRate = 2e-5
        vae_params.vae_hiddenDim = 500
        vae_params.numEpochs = 200000
    elif to_problem == MMTOTempDependentExamples.LBracket_Mass_MultipleConstraints:
        structural_problem=MMTOStructuralExamples.LBracket
        thermal_problem=MMTOThermalExamples.LBracketThermal
        Force = 5000
        kwargs['topload'] = Force
        kwargs['midload'] = 0

        to_params.Comment  = "Thermal + Structural TO Problem"
        to_params.MaterialsExcelFile = './DataVaryingTemperature/LSR_20251119_all_materials.xlsx'
        to_params.Objective=(TO_QOI.MASS, None)
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints=[ (TO_QOI.STRESS_FAILURE_FACTOR, None, 0.8),
                               (TO_QOI.FATIGUE_FAILURE_FACTOR, None, 1),
                               (TO_QOI.TEMPERATURE_FAILURE_FACTOR, None, 1),
                               (TO_QOI.MEAN_CRITICALITY, None, 1),
                               (TO_QOI.PBR, None, 1.8),
                               (TO_QOI.COMPLIANCE, None, 0.0005*Force)]
        vae_params.latentDim = 6
        vae_params.learningRate = 2e-5
        vae_params.vae_hiddenDim = 500
        vae_params.numEpochs = 200000

    elif to_problem == MMTOTempDependentExamples.LBracket_Stress_MultipleConstraints:
        structural_problem=MMTOStructuralExamples.LBracket
        thermal_problem=MMTOThermalExamples.LBracketThermal
        kwargs['topload'] = 1e4 
        kwargs['midload'] = 0
        to_params.Comment  = "Thermal + Structural TO Problem"
        to_params.MaterialsExcelFile = './DataVaryingTemperature/LSR_20251119_all_materials_2.xlsx'
        to_params.Objective=(TO_QOI.PNORM_STRESS, None)
        to_params.ExtrudeZ = True
        to_params.RelativeFilterRadius = 1.5
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints=[ (TO_QOI.VOLUME_FRACTION, None, 0.4), (TO_QOI.MASS, None, 30),
                               (TO_QOI.TEMPERATURE_FAILURE_FACTOR, None, 1),
                               (TO_QOI.FATIGUE_FAILURE_FACTOR, None, 0.5),
                               (TO_QOI.PBR, None, 1.6)]
        
        vae_params.latentDim = 6
        vae_params.learningRate = 2e-5
        vae_params.vae_hiddenDim = 500
        vae_params.numEpochs = 200000

    elif to_problem == MMTOTempDependentExamples.BliskSection_Compliance_MassCost:
        structural_problem = MMTOStructuralExamples.BliskSection
        thermal_problem=MMTOThermalExamples.BliskSection
        to_params.Comment  = "Large DOF"
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = False
        to_params.nDOFDesired = 100000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.MASS, None,  100), (TO_QOI.COST, None, 100)]
        to_params.MaterialsExcelFile = './DataVaryingTemperature/6MaterialsTempDependent.xlsx'

        # for large number of materials and attributes, we need to train the VAE longer
        vae_params.learningRate = 2e-5
        vae_params.vae_hiddenDim = 500
        vae_params.numEpochs = 200000
        vae_params.latentDim = 6
    elif to_problem == MMTOTempDependentExamples.BliskSection_Compliance_Mass:
        structural_problem = MMTOStructuralExamples.BliskSection
        thermal_problem=MMTOThermalExamples.BliskSection
        to_params.Comment  = "Large DOF"
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = False
        to_params.nDOFDesired = 10000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.MASS, None,  20)]
        to_params.MaterialsExcelFile = './DataVaryingTemperature/METALSDemoMaterials.xlsx'

        # for large number of materials and attributes, we need to train the VAE longer
        vae_params.learningRate = 2e-5
        vae_params.vae_hiddenDim = 500
        vae_params.numEpochs = 200000
        vae_params.latentDim = 6

    elif to_problem == MMTOTempDependentExamples.BliskSection_Stress_MassComplianceCriticality:
        structural_problem = MMTOStructuralExamples.BliskSection
        thermal_problem=MMTOThermalExamples.BliskSection
        to_params.Comment  = "Large DOF"
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = False
        to_params.nDOFDesired = 100000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.PNORM_STRESS, None)
        to_params.Constraints = [(TO_QOI.MASS, None,  40), (TO_QOI.COMPLIANCE, None, 100), (TO_QOI.MEAN_CRITICALITY, None, 3.5)]
        to_params.MaterialsExcelFile = './DataVaryingTemperature/6MaterialsTempDependent.xlsx'

        # for large number of materials and attributes, we need to train the VAE longer
        vae_params.learningRate = 2e-5
        vae_params.vae_hiddenDim = 500
        vae_params.numEpochs = 200000
        vae_params.latentDim = 6

    elif to_problem == MMTOTempDependentExamples.BliskSection_Mass_MultipleConstraints:
        structural_problem = MMTOStructuralExamples.BliskSection
        thermal_problem=MMTOThermalExamples.BliskSection
        to_params.Comment  = "Large DOF"
        kwargs['rpm'] = 1000
        kwargs['radialForce'] = 0
        kwargs['downwardForce'] = 1500
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = False
        to_params.nDOFDesired = 100000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.MASS, None)
        to_params.Constraints=[ (TO_QOI.STRESS_FAILURE_FACTOR, None, 1.0),
                               (TO_QOI.FATIGUE_FAILURE_FACTOR, None, 0.5),
                               (TO_QOI.MEAN_CRITICALITY, None, 0.7),
                               (TO_QOI.PBR, None, 1.8),
                               (TO_QOI.COMPLIANCE, None, 20)]
        to_params.MaterialsExcelFile = './DataVaryingTemperature/LSR_20251119_all_materials.xlsx'

        vae_params.latentDim = 6
        vae_params.learningRate = 2e-5
        vae_params.vae_hiddenDim = 500
        vae_params.numEpochs = 200000
    elif to_problem == MMTOTempDependentExamples.BliskSection_Test:
        structural_problem = MMTOStructuralExamples.BliskSection
        thermal_problem=MMTOThermalExamples.BliskSection
        to_params.Comment  = "Large DOF"
        kwargs['rpm'] = 1000
        kwargs['radialForce'] = 0
        kwargs['downwardForce'] = 200
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = False
        to_params.nDOFDesired = 10000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.MASS, None)
        to_params.Constraints=[ (TO_QOI.STRESS_FAILURE_FACTOR, None, 1.0),
                               (TO_QOI.FATIGUE_FAILURE_FACTOR, None, 0.5),
                               (TO_QOI.MEAN_CRITICALITY, None, 0.7),
                               (TO_QOI.PBR, None, 1.8),
                               (TO_QOI.COMPLIANCE, None, 50)]
        to_params.MaterialsExcelFile = './DataVaryingTemperature/LSR_20251119_all_materials-2.xlsx'

        vae_params.latentDim = 6
        vae_params.learningRate = 2e-5
        vae_params.vae_hiddenDim = 500
        vae_params.numEpochs = 200000

    else:
        raise ValueError(f"Unknown problem: {to_problem}")
    
    mesh, mat_prop, bc, elem_body_force = getMMTOStructuralProblem(structural_problem, 
                                                                   nDOFDesired = to_params.nDOFDesired, **kwargs)

    if 'thermal_problem' in locals() and thermal_problem is not None:
        mesh_thermal, mat_prop_thermal, bc_thermal = getMMTOThermalProblem(thermal_problem, 
                                                                           nDOFDesired=to_params.nDOFDesired)
        
    else:
        mesh_thermal, mat_prop_thermal, bc_thermal = None, None, None
    

    # Add  elements to keep
    to_params.ElemsToKeep  = None # default value

    # Here we add additional parameters specific to the optimization problem
    if (to_params.KeepFixedElems):
        to_params.ElemsToKeep = find_elements_with_fixedDOF(mesh, bc,nDOFPerNode=3)


    if 'Blisk' in to_problem.name:
        centerPt = [0,0,0]
        axis = [0,0,1]
        outerRadius1 = 0.558
        outerRadius2 = 1
        bladeElements = mesh.get_elems_within_annular_region(centerPt,axis,outerRadius1,outerRadius2)
        to_params.ElemsToKeep = np.union1d(to_params.ElemsToKeep, bladeElements)

    return mesh, mesh_thermal, mat_prop, mat_prop_thermal, bc, bc_thermal, elem_body_force, to_params, vae_params
    
