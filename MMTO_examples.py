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

class MMTOExamples(enum.Enum):
    Bridge_Compliance_MassCost = enum.auto()

    LBracketTopLoad_Compliance_Mass = enum.auto()
    LBracketTopLoad_Compliance_MassCost = enum.auto()
    LBracketTopLoad_Compliance_MassCriticality = enum.auto()
    LBracketTopLoad_Stress_Compliance = enum.auto()
    LBracketTopLoad_Mass_StressSafetyFactorCompliance = enum.auto()

    BliskSection_Compliance_MassCost = enum.auto()
    BliskSection_Mass_ComplianceCriticality = enum.auto()


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
    if to_problem == MMTOExamples.Bridge_Compliance_MassCost:
        structural_problem = MMTOStructuralExamples.Bridge
        to_params.Comment  = "Benchmark 2.5D with Mass and Cost Constraint"
        to_params.XSymmetry = True 
        to_params.EXtrudeZ = True
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.MaterialsExcelFile = './DataConstantTemperature/Bridge3Materials.xlsx'
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.MASS, None, 0.4*5000), (TO_QOI.COST, None, 0.3*5000)]
        #to_params.RelativeFilterRadius = 1.5
        vae_params.klFactor = 5e-6
        vae_params.learningRate = 2e-5
        vae_params.numEpochs = 100000
        vae_params.vae_hiddenDim = 500
      
    elif to_problem == MMTOExamples.LBracketTopLoad_Compliance_Mass:
        structural_problem = MMTOStructuralExamples.LBracket
        kwargs['topload'] = 1e4
        kwargs['midload'] = 0
        to_params.Comment  = "Stress Safety Factor"
        to_params.MaterialsExcelFile = './DataConstantTemperature/3Materials.xlsx'
        to_params.Objective = (TO_QOI.COMPLIANCE, None) 
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 10000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints = [(TO_QOI.MASS, None, 60)] 
        vae_params.klFactor=5e-6
        vae_params.learningRate=2e-6
        vae_params.numEpochs= 100000
        vae_params.vae_hiddenDim=500

    elif to_problem == MMTOExamples.LBracketTopLoad_Compliance_MassCost:
        structural_problem = MMTOStructuralExamples.LBracket
        kwargs['topload'] = 1e4
        kwargs['midload'] = 0
        to_params.Comment  = "Stress Safety Factor"
        to_params.MaterialsExcelFile = './DataConstantTemperature/3Materials.xlsx'
        to_params.Objective = (TO_QOI.COMPLIANCE, None) 
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 10000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints = [(TO_QOI.MASS, None, 60),(TO_QOI.COST, None, 100)] 
        vae_params.klFactor=5e-6
        vae_params.learningRate=2e-6
        vae_params.numEpochs= 100000
        vae_params.vae_hiddenDim=500

    elif to_problem == MMTOExamples.LBracketTopLoad_Compliance_MassCriticality:
        structural_problem = MMTOStructuralExamples.LBracket
        kwargs['topload'] = 1e4
        kwargs['midload'] = 0
        to_params.Comment  = "Stress Safety Factor"
        to_params.MaterialsExcelFile = './DataConstantTemperature/3Materials.xlsx'
        to_params.Objective = (TO_QOI.COMPLIANCE, None) 
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 10000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints = [(TO_QOI.MASS, None, 160),(TO_QOI.CRITICALITY, None, 0.5)] 
        vae_params.klFactor=5e-6
        vae_params.learningRate=2e-6
        vae_params.numEpochs= 150000
        vae_params.vae_hiddenDim=500   

    elif to_problem == MMTOExamples.LBracketTopLoad_Stress_Compliance:
        structural_problem = MMTOStructuralExamples.LBracket
        kwargs['topload'] = 1e4
        kwargs['midload'] = 0
        to_params.Comment  = "Stress Safety Factor"
        to_params.MaterialsExcelFile = './DataConstantTemperature/3Materials.xlsx'
        to_params.Objective = (TO_QOI.PNORM_STRESS, None) 
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 10000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints = [ (TO_QOI.COMPLIANCE, None, 300)] 
        vae_params.klFactor=5e-6
        vae_params.learningRate=2e-6
        vae_params.numEpochs= 150000
        vae_params.vae_hiddenDim=500
    
    elif to_problem == MMTOExamples.LBracketTopLoad_Mass_StressSafetyFactorCompliance:
        structural_problem = MMTOStructuralExamples.LBracket
        kwargs['topload'] = 1e4
        kwargs['midload'] = 0
        to_params.Comment  = "Stress Safety Factor"
        to_params.MaterialsExcelFile = './DataConstantTemperature/3Materials.xlsx'
        to_params.Objective = (TO_QOI.MASS, None) 
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 10000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints = [ (TO_QOI.STRESS_SAFETY_FACTOR, None,200), (TO_QOI.COMPLIANCE, None, 400)] 
        vae_params.klFactor=5e-6
        vae_params.learningRate=2e-6
        vae_params.numEpochs= 150000
        vae_params.vae_hiddenDim=500
       
    elif to_problem == MMTOExamples.BliskSection_Compliance_MassCost:
        structural_problem = MMTOStructuralExamples.BliskSection
        to_params.Comment  = "Large DOF"
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = False
        to_params.nDOFDesired = 100000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.MASS, None,  30), (TO_QOI.COST, None, 25)]
        to_params.MaterialsExcelFile = './DataConstantTemperature/3Materials.xlsx'
        vae_params.klFactor=5e-5
        vae_params.learningRate=8e-6
        vae_params.numEpochs=150000
        vae_params.vae_hiddenDim=500
        
    elif to_problem == MMTOExamples.BliskSection_Mass_ComplianceCriticality:
        structural_problem = MMTOStructuralExamples.BliskSection
        to_params.Comment  = "Large DOF"
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = False
        to_params.nDOFDesired = 100000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.MASS, None)
        to_params.Constraints = [(TO_QOI.COMPLIANCE, None,10), 
                                 (TO_QOI.CRITICALITY, None,0.5)]
        to_params.MaterialsExcelFile = './DataConstantTemperature/3Materials.xlsx'
        vae_params.klFactor=5e-5
        vae_params.learningRate=2e-4
        vae_params.numEpochs=150000
        vae_params.vae_hiddenDim=500
        

    else:
        raise ValueError(f"Unknown problem: {to_problem}")
    
    mesh, mat_prop, bc, elem_body_force = getMMTOStructuralProblem(structural_problem, nDOFDesired = to_params.nDOFDesired, **kwargs)


    # Add  elements to keep
    to_params.ElemsToKeep  = None # default value

    # Here we add additional parameters specific to the optimization problem
    if (to_params.KeepFixedElems):
        to_params.ElemsToKeep = find_elements_with_fixedDOF(mesh, bc,nDOFPerNode=3)


    if to_problem == MMTOExamples.BliskSection_Compliance_MassCost or \
        to_problem == MMTOExamples.BliskSection_Mass_ComplianceCriticality:
        centerPt = [0,0,0]
        axis = [0,0,1]
        outerRadius1 = 0.558
        outerRadius2 = 1
        bladeElements = mesh.get_elems_within_annular_region(centerPt,axis,outerRadius1,outerRadius2)
        to_params.ElemsToKeep = np.union1d(to_params.ElemsToKeep, bladeElements)
        print(f"Number of elements in the blade region = {len(bladeElements)}")


    return mesh,mat_prop, bc, elem_body_force, to_params, vae_params

