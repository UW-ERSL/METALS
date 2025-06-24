import numpy as np
import os
import enum
script_dir = os.path.dirname(os.path.abspath(__file__))
from LSRImports import *

class METALSStructuralExamples(enum.Enum):
	EdgeCantilever = enum.auto()  
	BliskWithBladeMass = enum.auto()
   
def getMETALSStructuralProblem(problem: METALSStructuralExamples, **kwargs):
  """Returns a structural problem based on the given problem name.

  Parameters:
  ----------
  problem : StructuralExamples
    The name of the problem to return.
  **kwargs : dict
    Additional keyword arguments to pass to the problem creation function.

  Returns:
  -------
  tuple
    A tuple containing the mesh, material properties, and boundary conditions for the problem.
  """
  if problem == METALSStructuralExamples.EdgeCantilever:
    return createEdgeCantileverProblem(**kwargs)
  elif problem == METALSStructuralExamples.BliskWithBladeMass:
    return createBliskSectionWithBlade(**kwargs)
  else:
    raise ValueError("Invalid structural example name.")

def createEdgeCantileverProblem(nDOFDesired: int = 10000, L: float = [0.4, 0.2, 0.1],
youngs_modulus=1,poissons_ratio=0.3,totalLoad = 10000):
  """Creates a edge loaded cantilever beam problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [0.1, 0.1, 0.1])
  youngs_modulus : float
    Young's modulus of material (default 2e11)
  poissons_ratio : float 
    Poisson's ratio of material (default 0.3)

  Returns:
  -------
  tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and load on right face
  """
  nVoxelsDesired = nDOFDesired/3    
  # Let the number of voxels be proportional to the length in each direction
  alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
  nelx = round(alpha*L[0])
  nely = round(alpha*L[1])
  nelz = round(alpha*L[2])
  mesh = hex_mesher.HexMesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  # line defined by x = xMax, and z = 0 
  load_nodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False), mesh.getNodesOnBoundingBoxPlane(2,True))
  load_dofs = 3 * load_nodes + 2  # z direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop=mat_lib.create_material_with_defaults(name=f"Test Material", youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
   
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
# ----------------------------------------

def createBliskSectionWithBlade(nDOFDesired: int = 50000, youngs_modulus = 1, 
                               poissons_ratio = 0.28, material_density = 1,rpm = 10000,radialForce =200000): #radial force zero
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, './Models/BliskModel/BliskSectionWithBlade2.STL')


  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()

  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  # fix inner radius
  centerPt = [0,0,0]
  axis = [0,0,1]
  innerRadius = 0.05
  fixed_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,innerRadius-mesh.elem_size[0]*0.707,
                                                     innerRadius+mesh.elem_size[0]*0.707)  
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  total_mesh_volume = np.prod(mesh.elem_size) * mesh.num_elems # * 0.0283168 # ft3 to m3
  print("total mesh volume in m3",total_mesh_volume)

  total_mass = material_density * total_mesh_volume
  print("total mass in kg",total_mass)


  elem_body_force = np.zeros(3*mesh.num_elems)
  omega = 2*np.pi*rpm/60
  for e in range(mesh.num_elems):
    center = mesh.elem_centers[e]
    # Add centrifugal force to each element in xy plane
    elem_body_force[3*e:3*e+2] = (material_density*np.prod(mesh.elem_size)) * omega**2 *  center[:2]

  print("total body force ",np.linalg.norm(elem_body_force))
  outerRadius = 0.22
  load_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,outerRadius-mesh.elem_size[0]*0.707,
                                                    outerRadius+mesh.elem_size[0]*0.707)    
  
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  boundaryForce = np.zeros(3*mesh.num_nodes) 
  # Apply radial force on each node on the circumference 
  
  for node in load_nodes:
    node_pos = mesh.node_xyz[node,:2] # get x,y coordinates
    r = np.sqrt(np.sum(node_pos**2)) # distance from center
    if r > 0:
      # Unit vector in radial direction
      radial_dir = node_pos/r
      # Add x and y dofs with force components
      boundaryForce[3*node] = radialForce/len(load_nodes) * radial_dir[0]  
      boundaryForce[3*node + 1] = radialForce/len(load_nodes) * radial_dir[1]
  
  bc = bound_cond.BC(force = boundaryForce,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  # mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
  #                     poissons_ratio=poissons_ratio)
  mat_prop=mat_lib.create_material_with_defaults(name=f"Test material Blisk", youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
   
  # elem_body_force = None
  # print("Total body force ",elem_body_force)
  # print("Num of elems ",mesh.num_elems)
  # print("shape of elem_body_force ",elem_body_force.shape)

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------