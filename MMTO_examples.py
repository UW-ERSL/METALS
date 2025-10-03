import enum
from MMTO_structural_examples import *
from PyTOImports import  *

class VAEParams:
    klFactor = 5e-6
    learningRate = 2e-4
    numEpochs = 20000
    vae_hiddenDim = 500
    latentDim = 2

class METALSTOExamples(enum.Enum):
    EdgeCantilever = enum.auto()  # Another variant of edge cantilever
    BliskWithBladeMass = enum.auto()
    BliskSectionWithSymmetry = enum.auto()
    Bridge = enum.auto()
    LBracketMidLoadStressSafetyFactor = enum.auto()


def getMETALSTOProblem(to_problem: METALSTOExamples,nDOFDesired = None, **kwargs):
    """Get the structural topology optimization problem based on the specified example.

    Args:
        problem: The example problem to solve.
        **kwargs: Additional arguments to pass to the problem.

    Returns:
        StructuralTOProblem: The structural topology optimization problem.
    """
    
    to_params = TOParams()
    vae_params = VAEParams()
    if to_problem == METALSTOExamples.EdgeCantilever:
        structural_problem = METALSStructuralExamples.EdgeCantilever
        to_params.Comment = "Classic TO Problem"
        to_params.YSymmetry = True
        to_params.nDOFDesired = nDOFDesired
        to_params.Constraints = [(TO_QOI.MASS, None, 15)]  # kg
    elif to_problem == METALSTOExamples.BliskSectionWithSymmetry:
        structural_problem = METALSStructuralExamples.BliskSectionWithSymmetry
        to_params.Comment  = "Large DOF"
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = False
        to_params.nDOFDesired = 100000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.MASS, None,  0.035), (TO_QOI.COST, None, 0.5)]
        to_params.MaterialsExcelFile = './data/TeledyneMaterials.xlsx'
        vae_params.klFactor=5e-5
        vae_params.learningRate=2e-4
        vae_params.numEpochs=100000
        vae_params.vae_hiddenDim=500
        vae_params.latentDim=2
    elif to_problem == METALSTOExamples.Bridge:
        structural_problem = METALSStructuralExamples.Bridge
        to_params.Comment  = "Benchmark 2.5D with Mass and Cost Constraint"
        to_params.XSymmetry = True 
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.MaterialsExcelFile = './data/BridgeMaterials.xlsx'
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.MASS, None, 0.4*5000), (TO_QOI.COST, None, 0.3*5000)]
        vae_params.klFactor=5e-6
        vae_params.learningRate=2e-4
        vae_params.numEpochs=20000
        vae_params.vae_hiddenDim=500
        vae_params.latentDim=2
    elif to_problem == METALSTOExamples.LBracketMidLoadStressSafetyFactor:
        structural_problem = METALSStructuralExamples.LBracket
        kwargs['topload'] = 5e-4
        kwargs['midload'] = 0
        to_params.Comment  = "Stress Safety Factor"
        to_params.Objective = (TO_QOI.MASS, None) 
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints = [ (TO_QOI.STRESS_SAFETY_FACTOR, None,2), (TO_QOI.COMPLIANCE, None, 9e-4)] 
    else:
        raise ValueError(f"Unknown problem: {to_problem}")
    
    mesh, mat_prop, bc, elem_body_force = getMETALSStructuralProblem(structural_problem, nDOFDesired = to_params.nDOFDesired, **kwargs)

    # Add  elements to keep
    to_params.ElemsToKeep  = None # default value


    # Here we add additional parameters specific to the optimization problem
    if (to_params.KeepFixedElems):
        to_params.ElemsToKeep = find_elements_with_fixedDOF(mesh, bc,nDOFPerNode=3)


    if to_problem == METALSTOExamples.BliskWithBladeMass:
        # Get the elements to keep for the blade
        centerPt = [0,0,0]
        axis = [0,0,1]
        outerRadius1 = 0.057
        outerRadius2 = 0.08
        bladeElements = mesh.get_elems_within_annular_region(centerPt,axis,outerRadius1,outerRadius2)
        to_params.ElemsToKeep = np.union1d(to_params.ElemsToKeep, bladeElements)
    if to_problem == METALSTOExamples.BliskSectionWithSymmetry:
        centerPt = [0,0,0]
        axis = [0,0,1]
        outerRadius1 = 0.0558
        outerRadius2 = 0.1
        bladeElements = mesh.get_elems_within_annular_region(centerPt,axis,outerRadius1,outerRadius2)
        to_params.ElemsToKeep = np.union1d(to_params.ElemsToKeep, bladeElements)
    return mesh, mat_prop, bc, elem_body_force, to_params, vae_params