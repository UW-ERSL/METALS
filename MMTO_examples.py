import enum
from MMTO_structural_examples import *
from MMTO_thermal_examples import *
from PyTOImports import  *
from dataclasses import dataclass, field

@dataclass(slots=True) # avoid accidental modification
class VAEParams: #default VAE parameters
    klFactor: float = 5e-6
    learningRate: float = 2e-6
    numEpochs: int = 150000 # max epochs
    vae_hiddenDim: int = 500
    latentDim: int = 2 # hard coded for now
    maxAttributeErrorPercent: float = 0.001 # termination criteria for VAE training

class MMTOExamples(enum.Enum):
    Bridge_Compliance_MassCost = enum.auto()
    Bridge_Compliance_MassCost_Saitou = enum.auto()
    BridgeHalf_Compliance_MassCost = enum.auto()

    CantileverBenchmark_Compliance_Mass = enum.auto()
    CantileverBenchmark_Compliance_VolumeFraction = enum.auto()

    LBracketTopLoad_Compliance_Mass = enum.auto()
    LBracketTopLoad_Compliance_MassCost = enum.auto()
    LBracketTopLoad_Compliance_MassCriticality = enum.auto()
    LBracketTopLoad_Stress_VolumeFraction_Mass = enum.auto()
    LBracketTopLoad_Mass_StressFF = enum.auto()

    EdgeCantilever_Compliance_MassCost = enum.auto()
    
    BliskSection_Compliance_MassCost = enum.auto()
    BliskSection_Compliance_MassCriticality = enum.auto()
    BliskSection_Stress_Mass = enum.auto()


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
        to_params.ExtrudeZ = True
        to_params.RelativeFilterRadius = 1.5
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.MaterialsExcelFile = './DataConstantTemperature/3MaterialsBridge.xlsx'
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.MASS, None, 0.4*20000), (TO_QOI.COST, None, 0.3*20000)]
        vae_params.latentDim = 2
    elif to_problem == MMTOExamples.Bridge_Compliance_MassCost_Saitou:
        structural_problem = MMTOStructuralExamples.BridgeSaitou
        to_params.Comment  = "Benchmark 2.5D with Mass and Cost Constraint - Saitou Bridge Model"
        to_params.XSymmetry = True 
        to_params.ExtrudeZ = True
        to_params.RelativeFilterRadius = 1.5
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.MaterialsExcelFile = './DataConstantTemperature/3MaterialsBridge.xlsx'
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.MASS, None, 0.4*5000), (TO_QOI.COST, None, 0.3*5000)]
        vae_params.latentDim = 2
    elif to_problem == MMTOExamples.BridgeHalf_Compliance_MassCost:
        structural_problem = MMTOStructuralExamples.BridgeHalf
        to_params.Comment  = "Benchmark 2.5D with Mass and Cost Constraint - Half Model"
        to_params.XSymmetry = True 
        to_params.ExtrudeZ = True
        to_params.RelativeFilterRadius = 1.5
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.MaterialsExcelFile = './DataConstantTemperature/3MaterialsBridgev2.xlsx'
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.MASS, None, 0.4*10000), (TO_QOI.COST, None, 0.3*10000)]
        vae_params.latentDim = 2
    elif to_problem == MMTOExamples.CantileverBenchmark_Compliance_Mass:
        structural_problem = MMTOStructuralExamples.CantileverBenchmark
        to_params.Comment  = "Cantilever Benchmark with Mass and Cost Constraint"
        to_params.YSymmetry = True
        to_params.RelativeFilterRadius = 1.5
        to_params.nDOFDesired = 30000 if nDOFDesired is None else nDOFDesired
        to_params.MaterialsExcelFile = './DataConstantTemperature/5MaterialsCantilever.xlsx'
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.MASS, None, 0.3*9600)]
    elif to_problem == MMTOExamples.CantileverBenchmark_Compliance_VolumeFraction:
        structural_problem = MMTOStructuralExamples.CantileverBenchmark
        to_params.Comment  = "Cantilever Benchmark with Volume Fraction Constraint"
        to_params.YSymmetry = True
        to_params.RelativeFilterRadius = 1.5
        to_params.nDOFDesired = 30000 if nDOFDesired is None else nDOFDesired
        to_params.MaterialsExcelFile = './DataConstantTemperature/5MaterialsCantilever.xlsx'
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None,  1)]
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
        vae_params.latentDim = 3
    elif to_problem == MMTOExamples.LBracketTopLoad_Compliance_MassCost:
        structural_problem = MMTOStructuralExamples.LBracket
        kwargs['topload'] = 1e4
        kwargs['midload'] = 0
        to_params.Comment  = "Stress Safety Factor"
        to_params.RelativeFilterRadius = 1.5
        to_params.MaterialsExcelFile = './DataConstantTemperature/3Materials.xlsx'
        to_params.Objective = (TO_QOI.COMPLIANCE, None) 
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 10000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints = [(TO_QOI.MASS, None, 60),(TO_QOI.COST, None, 100)] 
        vae_params.latentDim = 3
    elif to_problem == MMTOExamples.LBracketTopLoad_Compliance_MassCriticality:
        structural_problem = MMTOStructuralExamples.LBracket
        kwargs['topload'] = 1e4
        kwargs['midload'] = 0
        to_params.Comment  = "Stress Safety Factor"
        to_params.MaterialsExcelFile = './DataConstantTemperature/3Materials.xlsx'
        to_params.Objective = (TO_QOI.COMPLIANCE, None) 
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 10000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints = [(TO_QOI.MASS, None, 75),(TO_QOI.MAX_CRITICALITY, None, 0.5)] 
        vae_params.latentDim = 3

    elif to_problem == MMTOExamples.LBracketTopLoad_Stress_VolumeFraction_Mass:
        structural_problem = MMTOStructuralExamples.LBracket
        kwargs['topload'] = 1e4
        kwargs['midload'] = 0
        to_params.RelativeFilterRadius = 2.5
        to_params.Comment  = "Stress Safety Factor"
        to_params.MaterialsExcelFile = './DataConstantTemperature/3Materials.xlsx'
        to_params.Objective = (TO_QOI.PNORM_STRESS, None) 
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints = [ (TO_QOI.VOLUME_FRACTION, None, 0.3), (TO_QOI.MASS, None, 60)] 
        vae_params.latentDim = 2

    elif to_problem == MMTOExamples.LBracketTopLoad_Mass_StressFF:
        structural_problem = MMTOStructuralExamples.LBracket
        kwargs['topload'] = 5e4
        kwargs['midload'] = 0
        to_params.Comment  = "Stress Safety Factor"
        to_params.MaterialsExcelFile = './DataConstantTemperature/3Materials.xlsx'
        to_params.Objective = (TO_QOI.MASS, None) 
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints = [(TO_QOI.STRESS_FAILURE_FACTOR, None, 0.5)] 
        vae_params.latentDim = 2    

    elif to_problem == MMTOExamples.EdgeCantilever_Compliance_MassCost:
        structural_problem = MMTOStructuralExamples.EdgeCantilever
        to_params.Comment  = "Stress Safety Factor"
        to_params.MaterialsExcelFile = './DataConstantTemperature/3Materials.xlsx'
        to_params.Objective = (TO_QOI.COMPLIANCE, None) 
        to_params.YSymmetry = True
        to_params.nDOFDesired = 20000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints = [ (TO_QOI.MASS, None, 10), (TO_QOI.COST, None, 60)  ] 
        vae_params.latentDim = 3
    elif to_problem == MMTOExamples.BliskSection_Compliance_MassCost:
        structural_problem = MMTOStructuralExamples.BliskSection
        to_params.Comment  = "Large DOF"
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = False
        to_params.nDOFDesired = 100000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.MASS, None,  30), (TO_QOI.COST, None, 50)]
        to_params.MaterialsExcelFile = './DataConstantTemperature/20MaterialsTeledyne.xlsx'
        vae_params.latentDim = 3
        # for large number of materials and attributes, we need to train the VAE longer
        vae_params.learningRate = 2e-5
        vae_params.vae_hiddenDim = 750
        vae_params.numEpochs = 200000
    

    elif to_problem == MMTOExamples.BliskSection_Compliance_MassCriticality:
        structural_problem = MMTOStructuralExamples.BliskSection
        to_params.Comment  = "Large DOF"
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = False
        to_params.nDOFDesired = 100000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.MASS, None,25), 
                                 (TO_QOI.MEAN_CRITICALITY, None,0.5)]
        to_params.MaterialsExcelFile = './DataConstantTemperature/20MaterialsTeledyne.xlsx'

        # for large number of materials and attributes, we need to train the VAE longer
        vae_params.learningRate = 2e-5
        vae_params.vae_hiddenDim = 750
        vae_params.numEpochs = 200000
        vae_params.latentDim = 3   
    elif to_problem == MMTOExamples.BliskSection_Stress_Mass:
        structural_problem = MMTOStructuralExamples.BliskSection
        to_params.Comment  = "Large DOF"
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = False
        to_params.nDOFDesired = 100000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.PNORM_STRESS, None)
        to_params.Constraints = [(TO_QOI.MASS, None,30)]
        to_params.MaterialsExcelFile = './DataConstantTemperature/20MaterialsTeledyne.xlsx'

        # for large number of materials and attributes, we need to train the VAE longer
        vae_params.learningRate = 2e-5
        vae_params.vae_hiddenDim = 750
        vae_params.numEpochs = 200000
        vae_params.latentDim = 3
    else:
        raise ValueError(f"Unknown problem: {to_problem}")
    
    mesh, mat_prop, bc, elem_body_force = getMMTOStructuralProblem(structural_problem, nDOFDesired = to_params.nDOFDesired, **kwargs)


    # Add  elements to keep
    to_params.ElemsToKeep  = None # default value

    # Here we add additional parameters specific to the optimization problem
    if (to_params.KeepFixedElems):
        to_params.ElemsToKeep = find_elements_with_fixedDOF(mesh, bc,nDOFPerNode=3)


    if to_problem == MMTOExamples.BliskSection_Compliance_MassCost or \
        to_problem == MMTOExamples.BliskSection_Compliance_MassCriticality or \
        to_problem == MMTOExamples.BliskSection_Stress_Mass: 
        centerPt = [0,0,0]
        axis = [0,0,1]
        outerRadius1 = 0.558
        outerRadius2 = 1
        bladeElements = mesh.get_elems_within_annular_region(centerPt,axis,outerRadius1,outerRadius2)
        to_params.ElemsToKeep = np.union1d(to_params.ElemsToKeep, bladeElements)
        print(f"Number of elements in the blade region = {len(bladeElements)}")

    return mesh,mat_prop, bc, elem_body_force, to_params, vae_params

