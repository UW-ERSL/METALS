import enum
from METALS_structural_examples import *
from LSRImports import *

class METALSTOExamples(enum.Enum):
	EdgeCantilever = enum.auto()  # Another variant of edge cantilever
	BliskWithBladeMass = enum.auto()
	BliskSectionWithSymmetry = enum.auto()
	BridgeMMTO = enum.auto()
def getMETALSTOProblem(to_problem: METALSTOExamples,nDOFDesired = None, **kwargs):
    """Get the structural topology optimization problem based on the specified example.

    Args:
        problem: The example problem to solve.
        **kwargs: Additional arguments to pass to the problem.

    Returns:
        StructuralTOProblem: The structural topology optimization problem.
    """
    
    to_params = TOParams()
    if to_problem == METALSTOExamples.EdgeCantilever:
        structural_problem = METALSStructuralExamples.EdgeCantilever
        to_params.Comment = "Classic TO Problem"
        to_params.YSymmetry = True
        to_params.nDOFDesired = nDOFDesired
        to_params.Constraints = [(TO_QOI.MASS, None, 15)]  # kg
    elif to_problem == METALSTOExamples.BliskWithBladeMass:
        structural_problem = METALSStructuralExamples.BliskWithBladeMass
        to_params.Comment  = "Large DOF"
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = True
        to_params.nDOFDesired = nDOFDesired
        to_params.Constraints = [(TO_QOI.MASS, None, 0.01)]  # kg
    elif to_problem == METALSTOExamples.BliskSectionWithSymmetry:
        structural_problem = METALSStructuralExamples.BliskSectionWithSymmetry
        to_params.Comment  = "Large DOF"
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = False
        to_params.nDOFDesired = 100000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints = [(TO_QOI.MASS, None, 0.035)]  # kg
    elif to_problem == METALSTOExamples.BridgeMMTO:
        structural_problem = METALSStructuralExamples.BridgeMMTO
        to_params.Comment  = "Benchmark 2.5D"
        to_params.XSymmetry = True 
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints = [(TO_QOI.MASS, None, 0.4*5000)]
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
    return mesh, mat_prop, bc, elem_body_force, to_params