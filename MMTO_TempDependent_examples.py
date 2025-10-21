import enum
from MMTO_structural_examples import *
from MMTO_thermal_examples import *
from PyTOImports import  *

class VAEParams:
    klFactor = 5e-6
    learningRate = 2e-4
    numEpochs = 100000
    vae_hiddenDim = 500
    latentDim = 2
    maxAttributeErrorPercent = 0.0001

class MMTOTempDependentExamples(enum.Enum):
    LBracket_Compliance_Mass = enum.auto()
    LBracket_Compliance_MassCost = enum.auto()
    LBracket_Compliance_MassCriticality = enum.auto()
    LBracket_Pnormstress_ComplianceMass = enum.auto()
    LBracket_Mass_ComplianceSafetyFactor = enum.auto()


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
        kwargs['topload'] = 1000 
        kwargs['midload'] = 0
        to_params.Comment  = "Thermal + Structural TO Problem"
        to_params.MaterialsExcelFile = './DataVaryingTemperature/Data3MaterialsCriticalityConductivity.xlsx'
        to_params.Objective=(TO_QOI.COMPLIANCE, None)
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 10000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints=[(TO_QOI.MASS, None, 60)]
        vae_params.klFactor=5e-6
        vae_params.learningRate=2e-6
        vae_params.numEpochs= 150000
        vae_params.vae_hiddenDim=500
        vae_params.latentDim=2 # don't change this. Only 2D latent space is supported 
    elif to_problem == MMTOTempDependentExamples.LBracket_Compliance_MassCost:
        structural_problem=MMTOStructuralExamples.LBracket
        thermal_problem=MMTOThermalExamples.LBracketThermal
        kwargs['topload'] = 1000 
        kwargs['midload'] = 0
        to_params.Comment  = "Thermal + Structural TO Problem"
        to_params.MaterialsExcelFile = './DataVaryingTemperature/Data3Materials.xlsx'
        to_params.Objective=(TO_QOI.COMPLIANCE, None)
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 10000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints=[(TO_QOI.MASS, None, 50), (TO_QOI.COST, None, 200)]
        vae_params.klFactor=5e-6
        vae_params.learningRate=2e-6
        vae_params.numEpochs= 100000
        vae_params.vae_hiddenDim=500
        vae_params.latentDim=2 # don't change this. Only 2D latent space is supported 
    elif to_problem == MMTOTempDependentExamples.LBracket_Compliance_MassCriticality:
        structural_problem=MMTOStructuralExamples.LBracket
        thermal_problem=MMTOThermalExamples.LBracketThermal
        kwargs['topload'] = 1000 
        kwargs['midload'] = 0
        to_params.Comment  = "Thermal + Structural TO Problem"
        to_params.MaterialsExcelFile = './DataVaryingTemperature/Data3MaterialsCriticalityConductivity.xlsx'
        to_params.Objective=(TO_QOI.COMPLIANCE, None)
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 10000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints=[(TO_QOI.MASS, None, 50), (TO_QOI.CRITICALITY, None, 0.2)]
        vae_params.klFactor=5e-6
        vae_params.learningRate=2e-6
        vae_params.numEpochs= 100000
        vae_params.vae_hiddenDim=500
        vae_params.latentDim=2 # don't change this. Only 2D latent space is supported 
    elif to_problem == MMTOTempDependentExamples.LBracket_Pnormstress_ComplianceMass:
        structural_problem=MMTOStructuralExamples.LBracket
        thermal_problem=MMTOThermalExamples.LBracketThermal
        kwargs['topload'] = 1000 
        kwargs['midload'] = 0
        to_params.Comment  = "Thermal + Structural TO Problem"
        to_params.MaterialsExcelFile = './DataVaryingTemperature/Data3Materials.xlsx'
        to_params.Objective=(TO_QOI.PNORM_STRESS, None)
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 20000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints=[(TO_QOI.MASS, None, 50), (TO_QOI.COMPLIANCE, None, 3)]
        vae_params.klFactor=5e-6
        vae_params.learningRate=2e-6
        vae_params.numEpochs= 100000
        vae_params.vae_hiddenDim=500
        vae_params.latentDim=2 # don't change this. Only 2D latent space is supported
    elif to_problem == MMTOTempDependentExamples.LBracket_Mass_ComplianceSafetyFactor:
        structural_problem=MMTOStructuralExamples.LBracket
        thermal_problem=MMTOThermalExamples.LBracketThermal
        kwargs['topload'] = 5e-4   
        kwargs['midload'] = 0
        to_params.Comment  = "Thermal + Structural TO Problem"
        to_params.MaterialsExcelFile =  './DataVaryingTemperature/Data3MaterialsCriticalityConductivity.xlsx'
        to_params.Objective=(TO_QOI.MASS, None) 
        to_params.ExtrudeZ = True
        to_params.T0=500
        to_params.E0=1
        to_params.Y0=1
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints=[ (TO_QOI.STRESS_SAFETY_FACTOR, None,2), (TO_QOI.COMPLIANCE, None, 9e-4)] 
        vae_params.klFactor=5e-6
        vae_params.learningRate=2e-6
        vae_params.numEpochs= 100000
        vae_params.vae_hiddenDim=500
        vae_params.latentDim=2 # don't change this. Only 2D latent space is supported 
    

    else:
        raise ValueError(f"Unknown problem: {to_problem}")
    
    mesh, mat_prop, bc, elem_body_force = getMMTOStructuralProblem(structural_problem, nDOFDesired = to_params.nDOFDesired, **kwargs)

    if 'thermal_problem' in locals() and thermal_problem is not None:
        print("Setting up thermal + structural TO problem")
        mesh_thermal, mat_prop_thermal, bc_thermal = getMMTOThermalProblem(thermal_problem, nDOFDesired=to_params.nDOFDesired, **kwargs)
        return mesh, mesh_thermal, mat_prop, mat_prop_thermal, bc, bc_thermal, elem_body_force, to_params, vae_params
    else:
        mesh_thermal, mat_prop_thermal, bc_thermal = None, None, None
    

    # Add  elements to keep
    to_params.ElemsToKeep  = None # default value

    # Here we add additional parameters specific to the optimization problem
    if (to_params.KeepFixedElems):
        to_params.ElemsToKeep = find_elements_with_fixedDOF(mesh, bc,nDOFPerNode=3)


    if to_problem == MMTOTempDependentExamples.BliskSectionComplianceMassCost or \
        to_problem == MMTOTempDependentExamples.BliskSectionMassComplianceCostSafetyFactor:
        centerPt = [0,0,0]
        axis = [0,0,1]
        outerRadius1 = 0.558
        outerRadius2 = 1
        bladeElements = mesh.get_elems_within_annular_region(centerPt,axis,outerRadius1,outerRadius2)
        to_params.ElemsToKeep = np.union1d(to_params.ElemsToKeep, bladeElements)
        print(f"Number of elements in the blade region = {len(bladeElements)}")


    return mesh, mesh_thermal,mat_prop, bc, elem_body_force, to_params, vae_params

