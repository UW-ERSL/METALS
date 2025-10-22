import numpy as np
import os
import enum
script_dir = os.path.dirname(os.path.abspath(__file__))
from PyTOImports import *
import scipy.sparse as spy_sprs
from scipy.sparse import lil_matrix
class MMTOStructuralExamples(enum.Enum):
	EdgeCantilever = enum.auto()  
	BliskWithBladeMass = enum.auto()
	BliskSection= enum.auto()
	Bridge = enum.auto()
	LBracket = enum.auto()

  
def getMMTOStructuralProblem(problem: MMTOStructuralExamples,nDOFDesired: int = 20000, **kwargs):
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
  if problem == MMTOStructuralExamples.EdgeCantilever:
    return createEdgeCantileverProblem(nDOFDesired=nDOFDesired,**kwargs)
  
  elif problem == MMTOStructuralExamples.BliskSection:
    return createBliskSectionProblem(nDOFDesired=nDOFDesired,**kwargs)
  
  elif problem == MMTOStructuralExamples.Bridge:
    return createBridgeProblem(nDOFDesired=nDOFDesired,**kwargs)
  
  elif problem == MMTOStructuralExamples.LBracket:
    return createLBracketProblem(nDOFDesired=nDOFDesired,**kwargs)
  else:
    raise ValueError("Invalid structural example name.")

def createEdgeCantileverProblem(nDOFDesired: int = 20000, L: float = [0.4, 0.2, 0.1],
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

def createLBracketProblem(nDOFDesired: int = 10000, topload = 1000,midload = 0):
  """Creates a structural problem setup for an L-bracket topology optimization.
  This function sets up a finite element mesh and boundary conditions for an L-bracket
  structural problem from an STL file. The mesh is created with approximately the desired
  number of degrees of freedom. The problem includes fixed boundary conditions on the top
  surface and a distributed load on a portion of the right surface.
  Args:
    nDOFDesired (int, optional): Desired number of degrees of freedom for the mesh. 
                  Defaults to 10000.
  Returns:
    tuple: A tuple containing:
      - mesh (Mesher): Mesh object with the L-bracket discretization
      - mat_prop (StructuralMaterial): Material properties object with structural parameters
      - bc (BC): Boundary conditions object with forces and constraints
  Notes:
    - The mesh is created from an STL file located at '../Models/LBracket/LBracket.STL'
    - Fixed boundary conditions are applied at y = yMax
    - Load is applied in the -y direction on nodes where y > 0.039 and x > 0.09
    - Total applied load is 1000 units distributed equally among loaded nodes
    - Material properties are set to E = 2.1e5 and ν = 0.3
  """
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.abspath(os.path.join(script_dir, '..', 'PyTO', 'Models', 'LBracket', 'LBracket.STL'))
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(1,False)  # y = yMax plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  force = np.zeros(3*mesh.num_nodes)
  node_pts = mesh.node_xyz
  if(abs(topload) > 0):
    topload_nodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False) , np.where((node_pts[:, 1] >= 0.36))[0]) # hard coded    
    topload_dofs = 3 * topload_nodes + 1  
    mesh.node_indices[topload_nodes, 3] = 2 # for plotting
    force[topload_dofs] = -topload/len(topload_nodes)

  if(abs(midload) > 0):
    midload_nodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False), np.where((node_pts[:, 1] >= 0.18) & (node_pts[:, 1] <= 0.22))[0]) # hard coded    
    midload_dofs = 3 * midload_nodes + 1  
    mesh.node_indices[midload_nodes, 3] = 2 # for plotting
    
    force[midload_dofs] = -midload/len(midload_nodes)

  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  # Define material properties
  mat_prop = mat_lib.create_material_with_defaults("CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0, yield_strength=1.0)
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------

def createBliskSectionProblem(nDOFDesired: int = 50000, rpm = 0, radialForce =0, downwardForce = 10000 , youngs_modulus = 1, poissons_ratio = 0.3): 
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, './Models/BliskModel/BliskSection.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()

  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  fixedTri = [1351,1352]
  
  fixed_nodes = mesh.get_nodes_on_triangles(fixedTri)
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten() # This is needed if the dofs are being retained for TO
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting
  C0 = lil_matrix((3*len(fixed_nodes), mesh.num_nodes * 3))
  for i, node in enumerate(fixed_nodes):
    C0[3*i, 3*node] = 1
    C0[3*i+1, 3*node+1] = 1
    C0[3*i+2, 3*node+2] = 1
  C0 = C0.tocsr()
  
  
  # Nodes on these triangles are to subject to sliding boundary condition
  # i.e. d.n = 0 where n is the normal to the surface of the triangle
  # Use PyTO to find these triangles
  #triSet1 = [1321, 1322, 1323, 1324, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1682, 1683, 1684]
  triSet1 = [1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680]
  
  sliding_nodes_1 = mesh.get_nodes_on_triangles(triSet1)
  normal_1 = mesh.stlGeom.get_triangle_normal(triSet1[0])
  
  udof1 = 3 * sliding_nodes_1
  vdof1 = 3 * sliding_nodes_1 + 1
  wdof1 = 3 * sliding_nodes_1 + 2
  # Create constraint matrix for sliding boundary conditions
  # First surface normal constraint
  C1 = lil_matrix((len(sliding_nodes_1), mesh.num_nodes * 3))
  for i, node in enumerate(sliding_nodes_1):
    C1[i, udof1[i]] = normal_1[0]
    C1[i, vdof1[i]] = normal_1[1]
    C1[i, wdof1[i]] = normal_1[2]
  C1 = C1.tocsr()

  #triSet2 = [1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804]
  triSet2 = [1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804]
  sliding_nodes_2 = mesh.get_nodes_on_triangles(triSet2)
  udof2 = 3 * sliding_nodes_2
  vdof2 = 3 * sliding_nodes_2 + 1 
  wdof2 = 3 * sliding_nodes_2 + 2
  # Second surface normal constraint
  C2 = lil_matrix((len(sliding_nodes_2), mesh.num_nodes * 3))
  normal_2 = mesh.stlGeom.get_triangle_normal(triSet2[0])
  for i, node in enumerate(sliding_nodes_2):
    C2[i, udof2[i]] = normal_2[0]
    C2[i, vdof2[i]] = normal_2[1]
    C2[i, wdof2[i]] = normal_2[2]
  C2 = C2.tocsr()

  # Combine constraints
  constraint_matrix = spy_sprs.vstack((spy_sprs.csr_matrix(C0), 
                       spy_sprs.csr_matrix(C1),
                       spy_sprs.csr_matrix(C2)))
  constraint_rhs = np.zeros(3*len(fixed_nodes) + len(sliding_nodes_1) + len(sliding_nodes_2))

  total_mesh_volume = np.prod(mesh.elem_size) * mesh.num_elems # * 0.0283168 # ft3 to m3

  mat_prop=mat_lib.create_material_with_defaults(name=f"Test Material", youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  material_density = mat_prop.mass_density
  total_mass = material_density * total_mesh_volume
  
  elem_body_force = None
  if (abs(rpm) > 0):
    elem_body_force = np.zeros(3*mesh.num_elems)
    omega = 2*np.pi*rpm/60
    for e in range(mesh.num_elems):
      center = mesh.elem_centers[e]
      # Add centrifugal force to each element in xy plane
      elem_body_force[3*e:3*e+2] = (material_density*np.prod(mesh.elem_size)) * omega**2 *  center[:2]


  axis = [0,0,1] # z-axis
  centerPt = [0,0,0] # center of the blisk section
  outerRadius = 0.565
  load_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,outerRadius-mesh.elem_size[0]*0.707,
                                                    outerRadius+mesh.elem_size[0]*0.707)  
  
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  boundaryForce = np.zeros(3*mesh.num_nodes) 
 
  print("Applying radial force of ",radialForce," N on ", len(load_nodes), " nodes on outer circumference")
  # Apply radial force on each node on the circumference 
  for node in load_nodes:
    node_pos = mesh.node_xyz[node,:2] # get x,y coordinates
    r = np.sqrt(np.sum(node_pos**2)) # distance from center
    # Unit vector in radial direction
    radial_dir = node_pos/r
    # Add x and y dofs with force components
    boundaryForce[3*node] = radialForce/len(load_nodes) * radial_dir[0]  
    boundaryForce[3*node + 1] = radialForce/len(load_nodes) * radial_dir[1]
    boundaryForce[3*node + 2] = downwardForce/len(load_nodes)
  
  print("Total applied radial force ",np.sum(boundaryForce[0::3]),np.sum(boundaryForce[1::3]))
  

  mesh.node_indices[sliding_nodes_1, 3] = 1 # for plotting
  mesh.node_indices[sliding_nodes_2, 3] = 1 # for plotting

  # All constraints are implemented using the constraint matrix

  bc = bound_cond.BC(force = boundaryForce,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values
                     ,constraint_matrix=constraint_matrix,constraint_rhs=constraint_rhs) 

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
def createBridgeProblem(nDOFDesired: None):
    # Define grid size and element size
    nelx, nely, nelz = 100, 50, 1
    Lx, Ly, Lz = 100.0, 50.0, 1.0  # Example physical dimensions (adjust as needed)
    mesh = hex_mesher.HexMesher()
    mesh.grid_mesh(num_elems=(nelx, nely, nelz),
                   elem_size=(Lx/nelx, Ly/nely, Lz/nelz))
    mesh.createEdofMatStructural()

    node_pts = mesh.node_xyz
    bridge_length = np.max(node_pts[:, 0]) - np.min(node_pts[:, 0])

    # Fix left bottom edge (all 3 DOFs fixed)
    left_bottom_nodes = np.where((np.abs(node_pts[:, 0] - np.min(node_pts[:, 0])) < mesh.elem_size[0]/2) &
                                 (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]
   
    left_bottom_dofs = np.array([3 * left_bottom_nodes,
                                 3 * left_bottom_nodes + 1,
                                 3 * left_bottom_nodes + 2]).flatten().astype(int)

    # Fix right bottom edge (only y and z fixed, x free)
    right_bottom_nodes = np.where((np.abs(node_pts[:, 0] - np.max(node_pts[:, 0])) < mesh.elem_size[0]/2) &
                                  (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]

    right_bottom_dofs = np.array([3 * right_bottom_nodes + 1,  # y DOF
                                  3 * right_bottom_nodes + 2]).flatten().astype(int)

    # Combine fixed DOFs
    fixed_dofs = np.union1d(left_bottom_dofs, right_bottom_dofs)
    dirichlet_values = 0 * np.ones_like(fixed_dofs, dtype=float)
    mesh.node_indices[left_bottom_nodes, 3] = 1  # for plotting
    mesh.node_indices[right_bottom_nodes, 3] = 1  # for plotting

    # Apply edge loads
    force = np.zeros(3 * mesh.num_nodes)

    # Load at 1/3rd the length of the bridge
    load_nodes_1 = np.where((np.abs(node_pts[:, 0] - (np.min(node_pts[:, 0]) + bridge_length / 3)) < mesh.elem_size[0] / 2) &
                            (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]
    
    force[3 * load_nodes_1 + 1] = -1 / len(load_nodes_1)  # y direction
    mesh.node_indices[load_nodes_1, 3] = 2  # for plotting

    # Load at 1/2 the length of the bridge
    load_nodes_2 = np.where((np.abs(node_pts[:, 0] - (np.min(node_pts[:, 0]) + bridge_length / 2)) < mesh.elem_size[0] / 2) &
                            (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]
    force[3 * load_nodes_2 + 1] = -2 / len(load_nodes_2)  # y direction
    
    mesh.node_indices[load_nodes_2, 3] = 2  # for plotting

    # Load at 2/3rd the length of the bridge
    load_nodes_3 = np.where((np.abs(node_pts[:, 0] - (np.min(node_pts[:, 0]) + 2 * bridge_length / 3)) < mesh.elem_size[0] / 2) &
                            (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]
    force[3 * load_nodes_3 + 1] = -1 / len(load_nodes_3)  # y direction
    mesh.node_indices[load_nodes_3, 3] = 2  # for plotting

    # Define material properties
    # Note that we are creating a template material with unit Young's modulus so that it can be scaled later.
    mat_prop = mat_lib.create_material_with_defaults("CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0)

    # Create boundary conditions
    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)
    elem_body_force = None  
    return mesh, mat_prop, bc, elem_body_force
