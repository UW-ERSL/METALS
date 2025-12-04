import numpy as np
import os
import enum
script_dir = os.path.dirname(os.path.abspath(__file__))
from PyTOImports import *
import scipy.sparse as spy_sprs
from scipy.sparse import lil_matrix


class ThermoStructuralExamples(enum.Enum):
    BiClamp = enum.auto()
    MBBBeam = enum.auto()

def getThermoStructuralProblem(problem: ThermoStructuralExamples, **kwargs):
    if problem == ThermoStructuralExamples.BiClamp:
        return createBiClampProblem(**kwargs)
    elif problem == ThermoStructuralExamples.MBBBeam:
        return createMBBBeamProblem(**kwargs)
    else:
        raise ValueError("Invalid Thermo-structural problem specified.")
    
def createBiClampProblem(nDOFDesired=25000, structural_load = 1e5,TWall = 28):
    stl_file = os.path.abspath(os.path.join(script_dir, '..', 'PyTO', 'Models', 'Biclamp', 'BiClamp.STL'))

    mesh = hex_mesher.HexMesher()
    nElemsDesired = round(nDOFDesired/3)    # estimate
    mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    mesh.createEdofMatStructural()
    mesh.createEdofMatThermal()


    fixed_nodes_1 = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
    fixed_nodes_2 = mesh.getNodesOnBoundingBoxPlane(0,False) # x = xMax plane
    fixed_nodes = np.union1d(fixed_nodes_1, fixed_nodes_2)
    fixed_dofs = np.array([3 * fixed_nodes,
            3 * fixed_nodes + 1,
            3 * fixed_nodes + 2]).flatten().astype(int)
    dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
    mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

    # Get nodes on y = 0 plane
    y_plane_nodes = mesh.getNodesOnBoundingBoxPlane(1, True)  # y = 0 plane

    # Calculate x midpoint
    node_pts = mesh.node_xyz
    x_mid = (np.max(node_pts[:, 0]) + np.min(node_pts[:, 0])) / 2

    # Filter nodes within distance 0.01 from x_mid
    load_nodes = y_plane_nodes[np.abs(node_pts[y_plane_nodes, 0] - x_mid) < 0.01]
    mesh.node_indices[load_nodes, 3] = 2 # for plotting

    force = np.zeros(3*mesh.num_nodes)
    for node in load_nodes:
        force[3 * node + 1] = -structural_load / len(load_nodes)

    bcStructural = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

    thermal_fixed_dofs = np.array([fixed_nodes]).flatten().astype(int)
    thermal_dirichlet_values = TWall*np.ones_like(thermal_fixed_dofs, dtype = float)
    
    thermal_force = np.zeros(mesh.num_nodes) # no heat load

    bcThermal = bound_cond.BC(force = thermal_force,
						fixed_dofs = thermal_fixed_dofs,
						dirichlet_values = thermal_dirichlet_values) 
    mat_prop = mat_lib.create_material_with_defaults("CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0, yield_strength=1.0,thermal_expansion_coefficient=1.0,thermal_conductivity=1.0,specific_heat=1.0)
    elem_body_force = None

    return mesh, mat_prop, bcStructural,bcThermal, elem_body_force



def createMBBBeamProblem(nDOFDesired=25000, structural_load = 5000,Ta = 23,Tf  = 823):
    stl_file = os.path.abspath(os.path.join(script_dir, '..', 'PyTO', 'Models', 'MBBBeam', 'MBBBeam.STL'))

    mesh = hex_mesher.HexMesher()
    nElemsDesired = round(nDOFDesired/3)    # estimate
    mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    mesh.createEdofMatStructural()
    mesh.createEdofMatThermal()


    symmetryNodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
    hingedNodes = mesh.getNodesOnBoundingBoxPlane(1,True) # y = 0 plane
    node_pts = mesh.node_xyz
  
    xMax = np.max(node_pts[:, 0])

    hingedNodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(1,True), np.where(mesh.node_xyz[:,0] == xMax)[0])
   
    fixed_dofs = np.concatenate([3 * symmetryNodes,
            3 * hingedNodes + 1,
            3 * hingedNodes + 2]).astype(int)
    dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
    mesh.node_indices[hingedNodes, 3] = 1 # for plotting

  
    xMin = np.min(mesh.node_xyz[:,0])
    load_nodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(1,False), np.where(mesh.node_xyz[:,0] == xMin)[0])
    mesh.node_indices[load_nodes, 3] = 2 # for plotting

    force = np.zeros(3*mesh.num_nodes)
    for node in load_nodes:
        force[3 * node + 1] = -structural_load / len(load_nodes)

    bcStructural = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

    top_nodes = mesh.getNodesOnBoundingBoxPlane(1,False) # y =yMax plane
    bottom_nodes = mesh.getNodesOnBoundingBoxPlane(1,True) # y =yMin plane

    thermal_fixed_dofs = np.concatenate([top_nodes, bottom_nodes]).astype(int)
    thermal_dirichlet_values = np.concatenate([Ta * np.ones(len(top_nodes)), 
                                               Tf * np.ones(len(bottom_nodes))]).astype(float)
    
    thermal_force = np.zeros(mesh.num_nodes) # no heat load

    bcThermal = bound_cond.BC(force = thermal_force,
						fixed_dofs = thermal_fixed_dofs,
						dirichlet_values = thermal_dirichlet_values) 
    mat_prop = mat_lib.create_material_with_defaults("CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0, yield_strength=1.0,thermal_expansion_coefficient=1.0,thermal_conductivity=1.0,specific_heat=1.0)
    elem_body_force = None

    return mesh, mat_prop, bcStructural,bcThermal, elem_body_force