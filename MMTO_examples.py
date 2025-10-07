import enum
from MMTO_structural_examples import *
from MMTO_thermal_examples import *
from PyTOImports import  *

class VAEParams:
    klFactor = 5e-6
    learningRate = 2e-4
    numEpochs = 20000
    vae_hiddenDim = 500
    latentDim = 2

class MMTOExamples(enum.Enum):
    EdgeCantilever = enum.auto()  # Another variant of edge cantilever
    BridgeComplianceMassCost = enum.auto()
    LBracketMidLoadComplianceMassCost = enum.auto()
    LBracketMidLoadStressSafetyFactor = enum.auto()

    BliskSectionComplianceMassCost = enum.auto()
    BliskSectionMassComplianceCostSafetyFactor = enum.auto()

    LBracketThermal = enum.auto()
    LBracketStressThermal = enum.auto()

def getMMTOProblem(to_problem: MMTOExamples,nDOFDesired = None, **kwargs):
    """Get the structural topology optimization problem based on the specified example.

    Args:
        problem: The example problem to solve.
        **kwargs: Additional arguments to pass to the problem.

    Returns:
        StructuralTOProblem: The structural topology optimization problem.
    """
    
    to_params = TOParams()
    vae_params = VAEParams()
    if to_problem == MMTOExamples.EdgeCantilever:
        structural_problem = MMTOStructuralExamples.EdgeCantilever
        to_params.Comment = "Classic TO Problem"
        to_params.YSymmetry = True
        to_params.nDOFDesired = nDOFDesired
        to_params.Constraints = [(TO_QOI.MASS, None, 15)]  # kg
   
    elif to_problem == MMTOExamples.BridgeComplianceMassCost:
        structural_problem = MMTOStructuralExamples.Bridge
        to_params.Comment  = "Benchmark 2.5D with Mass and Cost Constraint"
        to_params.XSymmetry = True 
        to_params.EXtrudeZ = True
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.MaterialsExcelFile = './DataConstantTemperature/BridgeMaterials.xlsx'
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.MASS, None, 0.4*5000), (TO_QOI.COST, None, 0.3*5000)]
        vae_params.klFactor=5e-6
        vae_params.learningRate=2e-6
        vae_params.numEpochs= 100000
        vae_params.vae_hiddenDim=500
        vae_params.latentDim=2 # don't change this. Only 2D latent space is supported 

    elif to_problem == MMTOExamples.LBracketMidLoadComplianceMassCost:
        structural_problem = MMTOStructuralExamples.LBracket
        kwargs['topload'] = 0
        kwargs['midload'] = 1e4
        to_params.Comment  = "Stress Safety Factor"
        to_params.MaterialsExcelFile = './DataConstantTemperature/LBracketMaterialsSI.xlsx'
        to_params.Objective = (TO_QOI.COMPLIANCE, None) 
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 10000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints = [(TO_QOI.MASS, None, 100),(TO_QOI.COST, None, 200)] 
        vae_params.klFactor=5e-6
        vae_params.learningRate=2e-6
        vae_params.numEpochs= 100000
        vae_params.vae_hiddenDim=500
        vae_params.latentDim=2 # don't change this. Only 2D latent space is supported

    elif to_problem == MMTOExamples.LBracketMidLoadStressSafetyFactor:
        structural_problem = MMTOStructuralExamples.LBracket
        kwargs['topload'] = 0
        kwargs['midload'] = 5e4
        to_params.Comment  = "Stress Safety Factor"
        to_params.MaterialsExcelFile = './DataConstantTemperature/LBracketMaterialsSI.xlsx'
        to_params.Objective = (TO_QOI.MASS, None) 
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 10000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints = [ (TO_QOI.STRESS_SAFETY_FACTOR, None,200), (TO_QOI.COMPLIANCE, None, 400)] 
        vae_params.klFactor=5e-6
        vae_params.learningRate=2e-6
        vae_params.numEpochs= 100000
        vae_params.vae_hiddenDim=500
        vae_params.latentDim=2 # don't change this. Only 2D latent space is supported 

    elif to_problem == MMTOExamples.BliskSectionComplianceMassCost:
        structural_problem = MMTOStructuralExamples.BliskSection
        to_params.Comment  = "Large DOF"
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = False
        to_params.nDOFDesired = 100000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.MASS, None,  7), (TO_QOI.COST, None, 25)]
        to_params.MaterialsExcelFile = './DataConstantTemperature/TeledyneMaterialsSI.xlsx'
        vae_params.klFactor=5e-5
        vae_params.learningRate=2e-4
        vae_params.numEpochs=100000
        vae_params.vae_hiddenDim=500
        vae_params.latentDim=2
    elif to_problem == MMTOExamples.BliskSectionMassComplianceCostSafetyFactor:
        structural_problem = MMTOStructuralExamples.BliskSection
        to_params.Comment  = "Large DOF"
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = False
        to_params.nDOFDesired = 100000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.MASS, None)
        to_params.Constraints = [(TO_QOI.COMPLIANCE, None,  7), 
                                 (TO_QOI.COST, None, 10), 
                                 (TO_QOI.STRESS_SAFETY_FACTOR, None, 2.5)]
        to_params.MaterialsExcelFile = './DataConstantTemperature/TeledyneMaterialsSI.xlsx'
        vae_params.klFactor=5e-5
        vae_params.learningRate=2e-4
        vae_params.numEpochs=100000
        vae_params.vae_hiddenDim=500
        vae_params.latentDim=2
    elif to_problem == MMTOExamples.LBracketThermal:
        structural_problem=MMTOStructuralExamples.LBracket
        thermal_problem=MMTOThermalExamples.LBracketThermal
        kwargs['topload'] = 5e4 
        kwargs['midload'] = 0
        to_params.Comment  = "Thermal + Structural TO Problem"
        to_params.MaterialsExcelFile = './DataVaryingTemperature/LBracketMaterialsThermal.xlsx'
        to_params.Objective=(TO_QOI.COMPLIANCE, None)
        to_params.ExtrudeZ = True
        to_params.T0=500
        to_params.E0=1
        to_params.Y0=1
        to_params.nDOFDesired = 20000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints=[(TO_QOI.MASS, None, 60)]
        vae_params.klFactor=5e-6
        vae_params.learningRate=2e-6
        vae_params.numEpochs= 100000
        vae_params.vae_hiddenDim=500
        vae_params.latentDim=2 # don't change this. Only 2D latent space is supported 

    elif to_problem == MMTOExamples.LBracketStressThermal:
        structural_problem=MMTOStructuralExamples.LBracket
        thermal_problem=MMTOThermalExamples.LBracketThermal
        kwargs['topload'] = 5e-4   
        kwargs['midload'] = 0
        to_params.Comment  = "Thermal + Structural TO Problem"
        to_params.MaterialsExcelFile =  './MaterialDataTemperatureDependent/LBracketMaterialsThermal.xlsx'
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


    # Add  elements to keep
    to_params.ElemsToKeep  = None # default value

    # Here we add additional parameters specific to the optimization problem
    if (to_params.KeepFixedElems):
        to_params.ElemsToKeep = find_elements_with_fixedDOF(mesh, bc,nDOFPerNode=3)


    if to_problem == MMTOExamples.BliskSectionComplianceMassCost or \
        to_problem == MMTOExamples.BliskSectionMassComplianceCostSafetyFactor:
        centerPt = [0,0,0]
        axis = [0,0,1]
        outerRadius1 = 0.558
        outerRadius2 = 1
        bladeElements = mesh.get_elems_within_annular_region(centerPt,axis,outerRadius1,outerRadius2)
        to_params.ElemsToKeep = np.union1d(to_params.ElemsToKeep, bladeElements)
        print(f"Number of elements in the blade region = {len(bladeElements)}")
    return mesh, mat_prop, bc, elem_body_force, to_params, vae_params