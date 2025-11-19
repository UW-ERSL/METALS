import enum
from MMTO_structural_examples import *
from MMTO_thermal_examples import *
from PyTOImports import  *

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
    LBracket_Stress_MassVolume = enum.auto()
    LBracket_Stress_MassCompliance = enum.auto()


    BliskSection_Compliance_MassCost = enum.auto()
    BliskSection_Stress_MassComplianceCriticality = enum.auto()
    BliskSection_Compliance_Mass = enum.auto()

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
        to_params.MaterialsExcelFile = './DataVaryingTemperature/3MaterialsTempDependent.xlsx'
        to_params.Objective=(TO_QOI.COMPLIANCE, None)
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 10000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints=[(TO_QOI.MASS, None, 150)]
        vae_params.latentDim = 2
        vae_params.learningRate = 2e-5
        vae_params.vae_hiddenDim = 750
        vae_params.numEpochs = 200000
    elif to_problem == MMTOTempDependentExamples.LBracket_Compliance_MassCost:
        structural_problem=MMTOStructuralExamples.LBracket
        thermal_problem=MMTOThermalExamples.LBracketThermal
        kwargs['topload'] = 1e4 
        kwargs['midload'] = 0
        to_params.Comment  = "Thermal + Structural TO Problem"
        to_params.MaterialsExcelFile = './DataVaryingTemperature/3MaterialsTempDependent.xlsx'
        to_params.Objective=(TO_QOI.COMPLIANCE, None)
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 10000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints=[(TO_QOI.MASS, None, 50), (TO_QOI.COST, None, 200)]
        vae_params.latentDim = 2
        vae_params.learningRate = 2e-5
        vae_params.vae_hiddenDim = 750
        vae_params.numEpochs = 200000        
    elif to_problem == MMTOTempDependentExamples.LBracket_Stress_MassVolume:
        structural_problem=MMTOStructuralExamples.LBracket
        thermal_problem=MMTOThermalExamples.LBracketThermal
        kwargs['topload'] = 1e4 
        kwargs['midload'] = 0
        to_params.Comment  = "Thermal + Structural TO Problem"
        to_params.MaterialsExcelFile = './DataVaryingTemperature/LSR_20251118_KS.xlsx'
        to_params.Objective=(TO_QOI.PNORM_STRESS, None)
        to_params.ExtrudeZ = True
        to_params.RelativeFilterRadius = 1.5
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints=[ (TO_QOI.VOLUME_FRACTION, None, 0.3)]

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
        vae_params.latentDim = 3
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
        vae_params.vae_hiddenDim = 750
        vae_params.numEpochs = 200000
        vae_params.latentDim = 2
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
        vae_params.vae_hiddenDim = 750
        vae_params.numEpochs = 200000
        vae_params.latentDim = 3

    else:
        raise ValueError(f"Unknown problem: {to_problem}")
    
    mesh, mat_prop, bc, elem_body_force = getMMTOStructuralProblem(structural_problem, nDOFDesired = to_params.nDOFDesired, **kwargs)

    if 'thermal_problem' in locals() and thermal_problem is not None:
        mesh_thermal, mat_prop_thermal, bc_thermal = getMMTOThermalProblem(thermal_problem, nDOFDesired=to_params.nDOFDesired, **kwargs)
        
    else:
        mesh_thermal, mat_prop_thermal, bc_thermal = None, None, None
    

    # Add  elements to keep
    to_params.ElemsToKeep  = None # default value

    # Here we add additional parameters specific to the optimization problem
    if (to_params.KeepFixedElems):
        to_params.ElemsToKeep = find_elements_with_fixedDOF(mesh, bc,nDOFPerNode=3)


    if to_problem == MMTOTempDependentExamples.BliskSection_Compliance_MassCost or \
        to_problem == MMTOTempDependentExamples.BliskSection_Stress_MassComplianceCriticality or \
        to_problem == MMTOTempDependentExamples.BliskSection_Compliance_Mass:
        centerPt = [0,0,0]
        axis = [0,0,1]
        outerRadius1 = 0.558
        outerRadius2 = 1
        bladeElements = mesh.get_elems_within_annular_region(centerPt,axis,outerRadius1,outerRadius2)
        to_params.ElemsToKeep = np.union1d(to_params.ElemsToKeep, bladeElements)

    return mesh, mesh_thermal, mat_prop, mat_prop_thermal, bc, bc_thermal, elem_body_force, to_params, vae_params
    
