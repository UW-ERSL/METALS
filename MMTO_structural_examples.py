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
	BridgeHalf = enum.auto()
	BridgeSaitou = enum.auto()
	LBracket = enum.auto()
	CantileverBenchmark = enum.auto()
	MBBBeam = enum.auto()
	CenterCantilever = enum.auto()
	Table = enum.auto()
	GEGrabCAD= enum.auto()
  

  
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
  
  elif problem == MMTOStructuralExamples.BridgeHalf:
    return createBridgeProblemHalf(nDOFDesired=nDOFDesired,**kwargs)
  elif problem == MMTOStructuralExamples.BridgeSaitou:
    return createBridgeProblemSaitou(nDOFDesired=nDOFDesired,**kwargs)
  elif problem == MMTOStructuralExamples.CantileverBenchmark:
    return createCantileverBenchmarkProblem(nDOFDesired=nDOFDesired,**kwargs)
  elif problem == MMTOStructuralExamples.MBBBeam:
    return createMBBBeamProblem(nDOFDesired=nDOFDesired,**kwargs)
  elif problem == MMTOStructuralExamples.CenterCantilever:
    return createCenterCantileverProblem(nDOFDesired=nDOFDesired,**kwargs)
  elif problem == MMTOStructuralExamples.Table:
    return createTableProblem(nDOFDesired=nDOFDesired,**kwargs)
  elif problem == MMTOStructuralExamples.GEGrabCAD:
    return createGEGrabCADProblem(nDOFDesired=nDOFDesired,**kwargs)
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

def createBliskSectionProblem(nDOFDesired: int = 50000, rpm = 2000, radialForce = 0, downwardForce = 500 , 
                              youngs_modulus = 1, poissons_ratio = 0.3): 
 
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
  
  axis = [0,0,1] # z-axis
  centerPt = [0,0,0] # center of the blisk section
  outerRadius = 0.5
  tipRadius = 1.
  elem_body_force = None
  if (abs(rpm) > 0):
    elem_body_force = np.zeros(3*mesh.num_elems)
    omega = 2*np.pi*rpm/60
    for e in range(mesh.num_elems):
      center = mesh.elem_centers[e]
      if np.linalg.norm(center[:2]) > outerRadius:
        # Add centrifugal force to each element in xy plane
        elem_body_force[3*e:3*e+2] = (material_density*np.prod(mesh.elem_size)) * omega**2 *  center[:2]


  
  load_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,outerRadius-mesh.elem_size[0]*0.707,
                                                    outerRadius+mesh.elem_size[0]*0.707,)  
  
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  boundaryForce = np.zeros(3*mesh.num_nodes) 
 
  #print("Applying additional force on ", len(load_nodes), " nodes on outer circumference")
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
  

  mesh.node_indices[sliding_nodes_1, 3] = 1 # for plotting
  mesh.node_indices[sliding_nodes_2, 3] = 1 # for plotting

  # All constraints are implemented using the constraint matrix

  bc = bound_cond.BC(force = boundaryForce,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values
                     ,constraint_matrix=constraint_matrix,constraint_rhs=constraint_rhs) 

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
def createBridgeProblem(nDOFDesired: None):
    # Define grid size and element size
    nelx, nely, nelz = 200, 100, 1
    Lx, Ly, Lz = 200.0, 100.0, 1.0  # Example physical dimensions (adjust as needed)
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
    load_nodes_1 = np.where((np.abs(node_pts[:, 0] - (np.min(node_pts[:, 0]) + bridge_length / 4)) < mesh.elem_size[0] / 2) &
                            (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]
    
    force[3 * load_nodes_1 + 1] = -1 / len(load_nodes_1)  # y direction
    mesh.node_indices[load_nodes_1, 3] = 2  # for plotting

    # Load at 1/2 the length of the bridge
    load_nodes_2 = np.where((np.abs(node_pts[:, 0] - (np.min(node_pts[:, 0]) + bridge_length / 2)) < mesh.elem_size[0] / 2) &
                            (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]
    force[3 * load_nodes_2 + 1] = -2 / len(load_nodes_2)  # y direction
    
    mesh.node_indices[load_nodes_2, 3] = 2  # for plotting

    # Load at 2/3rd the length of the bridge
    load_nodes_3 = np.where((np.abs(node_pts[:, 0] - (np.min(node_pts[:, 0]) + 3 * bridge_length / 4)) < mesh.elem_size[0] / 2) &
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
def createBridgeProblemSaitou(nDOFDesired: None):
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
    load_nodes_1 = np.where((np.abs(node_pts[:, 0] - (np.min(node_pts[:, 0]) + bridge_length / 4)) < mesh.elem_size[0] / 2) &
                            (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]
    
    force[3 * load_nodes_1 + 1] = -1 / len(load_nodes_1)  # y direction
    mesh.node_indices[load_nodes_1, 3] = 2  # for plotting

    # Load at 1/2 the length of the bridge
    load_nodes_2 = np.where((np.abs(node_pts[:, 0] - (np.min(node_pts[:, 0]) + bridge_length / 2)) < mesh.elem_size[0] / 2) &
                            (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]
    force[3 * load_nodes_2 + 1] = -2 / len(load_nodes_2)  # y direction
    
    mesh.node_indices[load_nodes_2, 3] = 2  # for plotting

    # Load at 2/3rd the length of the bridge
    load_nodes_3 = np.where((np.abs(node_pts[:, 0] - (np.min(node_pts[:, 0]) + 3 * bridge_length / 4)) < mesh.elem_size[0] / 2) &
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
def createBridgeProblemHalf(nDOFDesired: None):
    # Define grid size and element size
    nelx, nely, nelz = 100, 100, 1
    Lx, Ly, Lz = 100.0, 100, 1.0  # Example physical dimensions (adjust as needed)
    mesh = hex_mesher.HexMesher()
    mesh.grid_mesh(num_elems=(nelx, nely, nelz),
                   elem_size=(Lx/nelx, Ly/nely, Lz/nelz))
    mesh.createEdofMatStructural()

    node_pts = mesh.node_xyz
    bridge_length = np.max(node_pts[:, 0]) - np.min(node_pts[:, 0])

    # Fix left bottom edge (only y and z fixed)
    left_bottom_nodes = np.where((np.abs(node_pts[:, 0] - np.min(node_pts[:, 0])) < mesh.elem_size[0]/2) &
                                 (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]
   
    left_bottom_dofs = np.array([
                                 3 * left_bottom_nodes + 1,
                                 3 * left_bottom_nodes + 2]).flatten().astype(int)

    # Fix right  edge (only x and z fixed, y free)
    right_edge_nodes = np.where((np.abs(node_pts[:, 0] - np.max(node_pts[:, 0])) < mesh.elem_size[0]/2))[0]

    right_edge_dofs = np.array([3 * right_edge_nodes,  # x DOF
                                3 * right_edge_nodes + 2]).flatten().astype(int)

    # Combine fixed DOFs
    fixed_dofs = np.union1d(left_bottom_dofs, right_edge_dofs)
    dirichlet_values = 0 * np.ones_like(fixed_dofs, dtype=float)
    mesh.node_indices[left_bottom_nodes, 3] = 1  # for plotting
    mesh.node_indices[right_edge_nodes, 3] = 1  # for plotting

    # Apply edge loads
    force = np.zeros(3 * mesh.num_nodes)

    # Load at 1/2 the length of the bridge
    load_nodes_1 = np.where((np.abs(node_pts[:, 0] - (np.min(node_pts[:, 0]) + bridge_length / 2)) < mesh.elem_size[0] / 2) &
                            (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]
    
    force[3 * load_nodes_1 + 1] = -1 / len(load_nodes_1)  # y direction
    mesh.node_indices[load_nodes_1, 3] = 2  # for plotting

    # Load at the length of the bridge
    load_nodes_2 = np.where((np.abs(node_pts[:, 0] - (np.min(node_pts[:, 0]) + bridge_length)) < mesh.elem_size[0] / 2) &
                            (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]
    force[3 * load_nodes_2 + 1] = -2 / len(load_nodes_2)  # y direction
    
    mesh.node_indices[load_nodes_2, 3] = 2  # for plotting


    # Define material properties
    # Note that we are creating a template material with unit Young's modulus so that it can be scaled later.
    mat_prop = mat_lib.create_material_with_defaults("CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0)

    # Create boundary conditions
    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)
    elem_body_force = None  
    return mesh, mat_prop, bc, elem_body_force

def createCantileverBenchmarkProblem(nDOFDesired: None):
    # Define grid size and element size
    nelx, nely, nelz = 120, 80, 1
    Lx, Ly, Lz = 120.0, 80, 1.0  # Example physical dimensions (adjust as needed)
    mesh = hex_mesher.HexMesher()
    mesh.grid_mesh(num_elems=(nelx, nely, nelz),
                   elem_size=(Lx/nelx, Ly/nely, Lz/nelz))
    mesh.createEdofMatStructural()

    node_pts = mesh.node_xyz
    cantilever_length = np.max(node_pts[:, 0]) - np.min(node_pts[:, 0])
    cantilever_height = np.max(node_pts[:, 1]) - np.min(node_pts[:, 1])
    # Fix left bottom edge (only y and z fixed)
    left_edge_nodes = np.where((np.abs(node_pts[:, 0] - np.min(node_pts[:, 0])) < mesh.elem_size[0]/2))[0]

    left_edge_dofs = np.array([  3*left_edge_nodes,
                                 3 * left_edge_nodes + 1,
                                 3 * left_edge_nodes + 2]).flatten().astype(int)


    # Combine fixed DOFs
    fixed_dofs = left_edge_dofs
    dirichlet_values = 0 * np.ones_like(fixed_dofs, dtype=float)
    mesh.node_indices[left_edge_nodes, 3] = 1  # for plotting

    # Apply edge loads
    force = np.zeros(3 * mesh.num_nodes)

    # Load at midpoint of right edge of the cantilever
    load_nodes_1 = np.where((np.abs(node_pts[:, 0] - (np.min(node_pts[:, 0]) + cantilever_length)) < mesh.elem_size[0] / 2) &
                            (np.abs(node_pts[:, 1] - (np.min(node_pts[:, 1]) + cantilever_height / 2)) < mesh.elem_size[1]/2))[0]
    
    force[3 * load_nodes_1 + 1] = -1 / len(load_nodes_1)  # y direction
    mesh.node_indices[load_nodes_1, 3] = 2  # for plotting




    # Define material properties
    # Note that we are creating a template material with unit Young's modulus so that it can be scaled later.
    mat_prop = mat_lib.create_material_with_defaults("CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0)

    # Create boundary conditions
    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)
    elem_body_force = None  
    return mesh, mat_prop, bc, elem_body_force

def createMBBBeamProblem(nDOFDesired=None):
    nelx, nely, nelz = 240, 80, 1
    Lx, Ly, Lz = 240, 80, 1.0
    mesh = hex_mesher.HexMesher()
    mesh.grid_mesh(num_elems=(nelx, nely, nelz),
                   elem_size=(Lx/nelx, Ly/nely, Lz/nelz))
    mesh.createEdofMatStructural()

    node_pts = mesh.node_xyz

    # Fix left bottom node (all DOFs fixed)
    left_nodes = np.where((np.abs(node_pts[:, 0] - np.min(node_pts[:, 0])) < mesh.elem_size[0]/2) &
                          (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]
    left_dofs = np.array([3 * left_nodes,
                          3 * left_nodes + 1,
                          3 * left_nodes + 2]).flatten().astype(int)

    # Fix right bottom node (roller: only y and z fixed, x free)
    right_nodes = np.where((np.abs(node_pts[:, 0] - np.max(node_pts[:, 0])) < mesh.elem_size[0]/2) &
                           (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]
    right_dofs = np.array([3 * right_nodes + 1,  # y DOF
                           3 * right_nodes + 2]).flatten().astype(int)

    # Combine fixed DOFs
    fixed_dofs = np.union1d(left_dofs, right_dofs)
    dirichlet_values = 0 * np.ones_like(fixed_dofs, dtype=float)
    mesh.node_indices[left_nodes, 3] = 1  # for plotting
    mesh.node_indices[right_nodes, 3] = 1  # for plotting

    # Load in the middle of the top edge
    mid_x = (np.min(node_pts[:, 0]) + np.max(node_pts[:, 0])) / 2
    top_nodes = np.where((np.abs(node_pts[:, 1] - np.max(node_pts[:, 1])) < mesh.elem_size[1]/2) &
                         (np.abs(node_pts[:, 0] - mid_x) < mesh.elem_size[0]/2))[0]
    force = np.zeros(3 * mesh.num_nodes)
    force[3 * top_nodes + 1] = -1 / len(top_nodes)  # y direction
    mesh.node_indices[top_nodes, 3] = 2  # for plotting

    # Define material properties
    mat_prop = mat_lib.create_material_with_defaults("CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0)

    # Create boundary conditions
    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)
    elem_body_force = None
    return mesh, mat_prop, bc, elem_body_force

def createCenterCantileverProblem(nDOFDesired=None, nVoxels: tuple = (40, 20, 10),
                                  youngs_modulus=1, poissons_ratio=0.3, totalLoad=1):
    """
    Cantilever beam with load along the rightmost horizontal bottom edge (x = xMax, z = zMin).
    Voxels are always 1x1x1.
    """
    nelx, nely, nelz = nVoxels
    mesh = hex_mesher.HexMesher()
    mesh.grid_mesh(num_elems=(nelx, nely, nelz),
                   elem_size=(1.0, 1.0, 1.0))  # Always 1x1x1 voxels
    mesh.createEdofMatStructural()

    # Fixed nodes on x = 0 plane (left face)
    fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0, True)
    fixed_dofs = np.array([3 * fixed_nodes,
                           3 * fixed_nodes + 1,
                           3 * fixed_nodes + 2]).flatten().astype(int)
    dirichlet_values = 0 * np.ones_like(fixed_dofs, dtype=float)
    mesh.node_indices[fixed_nodes, 3] = 1
    # # Load nodes: right face, middle horizontal edge (x = xMax, z = zMid)
    # node_pts = mesh.node_xyz
    # xMax = np.max(node_pts[:, 0])
    # zMid = (np.max(node_pts[:, 2]) + np.min(node_pts[:, 2])) / 2

    # load_nodes = np.where(
    #     (np.abs(node_pts[:, 0] - xMax) < mesh.elem_size[0]/2) &
    #     (np.abs(node_pts[:, 2] - zMid) < mesh.elem_size[2]/2)
    # )[0]
    # mesh.node_indices[load_nodes, 3] = 2  # for plotting

    # # Apply load in z direction at load nodes
    # force = np.zeros(3 * mesh.num_nodes)
    # force[3 * load_nodes + 2] = -totalLoad / len(load_nodes)  # z direction
    # Load node: center of right face (x = xMax)
    node_pts = mesh.node_xyz
    xMax = np.max(node_pts[:, 0])
    right_face_nodes = mesh.getNodesOnBoundingBoxPlane(0, False)

    y_center = (np.max(node_pts[right_face_nodes, 1]) + np.min(node_pts[right_face_nodes, 1])) / 2
    z_center = (np.max(node_pts[right_face_nodes, 2]) + np.min(node_pts[right_face_nodes, 2])) / 2

    distances = np.linalg.norm(node_pts[right_face_nodes][:, 1:3] - np.array([y_center, z_center]), axis=1)
    center_node = right_face_nodes[np.argmin(distances)]
    mesh.node_indices[center_node, 3] = 2  # for plotting

    # Apply load in z direction at the center node
    force = np.zeros(3 * mesh.num_nodes)
    force[3 * center_node + 2] = -totalLoad
    bc = bound_cond.BC(force=force,
                       fixed_dofs=fixed_dofs,
                       dirichlet_values=dirichlet_values)

    mat_prop = mat_lib.create_material_with_defaults(name="Test Material",
                                                    youngs_modulus=youngs_modulus,
                                                    poissons_ratio=poissons_ratio, mass_density=1.0)
    elem_body_force = None
    return mesh, mat_prop, bc, elem_body_force

def createTableProblem(nDOFDesired=None, nVoxels: tuple = (30, 30, 30),
                       youngs_modulus=1, poissons_ratio=0.3, totalLoad=1, percnodes=0.025):
    """
    Table problem: load at center of top face (z = zMax), fix percnodes*100% of bottom face nodes at four corners as square grids.
    Voxels are always 1x1x1.

    Parameters:
    ----------
    percnodes : float
        Fraction of bottom face nodes to fix at the corners (e.g., 0.1 for 10% total, 0.025 for 2.5% per corner)
    """
    nelx, nely, nelz = nVoxels
    mesh = hex_mesher.HexMesher()
    mesh.grid_mesh(num_elems=(nelx, nely, nelz),
                   elem_size=(1.0, 1.0, 1.0))
    mesh.createEdofMatStructural()

    node_pts = mesh.node_xyz

    # Top face (z = zMax): load at center node
    zMax = np.max(node_pts[:, 2])
    top_nodes = mesh.getNodesOnBoundingBoxPlane(2, False)
    y_center = (np.max(node_pts[top_nodes, 1]) + np.min(node_pts[top_nodes, 1])) / 2
    x_center = (np.max(node_pts[top_nodes, 0]) + np.min(node_pts[top_nodes, 0])) / 2
    distances = np.linalg.norm(node_pts[top_nodes][:, :2] - np.array([x_center, y_center]), axis=1)
    center_node = top_nodes[np.argmin(distances)]
    mesh.node_indices[center_node, 3] = 2  # for plotting

    force = np.zeros(3 * mesh.num_nodes)
    force[3 * center_node + 2] = -totalLoad  # z direction

    # Bottom face (z = zMin): fix percnodes*100% of nodes at four corners as square grids
    bottom_nodes = mesh.getNodesOnBoundingBoxPlane(2, True)
    n_bottom = len(bottom_nodes)
    desired_total = int(np.ceil(percnodes * n_bottom))
    n_per_corner = int(np.ceil(np.sqrt(desired_total / 4)))

    # Get unique sorted x and y coordinates on bottom face
    x_vals = np.sort(np.unique(node_pts[bottom_nodes, 0]))
    y_vals = np.sort(np.unique(node_pts[bottom_nodes, 1]))

    # Select n_per_corner grid points for each corner
    bl_nodes = bottom_nodes[
        np.isin(node_pts[bottom_nodes, 0], x_vals[:n_per_corner]) &
        np.isin(node_pts[bottom_nodes, 1], y_vals[:n_per_corner])
    ]
    br_nodes = bottom_nodes[
        np.isin(node_pts[bottom_nodes, 0], x_vals[-n_per_corner:]) &
        np.isin(node_pts[bottom_nodes, 1], y_vals[:n_per_corner])
    ]
    tl_nodes = bottom_nodes[
        np.isin(node_pts[bottom_nodes, 0], x_vals[:n_per_corner]) &
        np.isin(node_pts[bottom_nodes, 1], y_vals[-n_per_corner:])
    ]
    tr_nodes = bottom_nodes[
        np.isin(node_pts[bottom_nodes, 0], x_vals[-n_per_corner:]) &
        np.isin(node_pts[bottom_nodes, 1], y_vals[-n_per_corner:])
    ]

    fixed_nodes = np.unique(np.concatenate([bl_nodes, br_nodes, tl_nodes, tr_nodes]))
    mesh.node_indices[fixed_nodes, 3] = 1  # for plotting

    fixed_dofs = np.array([3 * fixed_nodes,
                           3 * fixed_nodes + 1,
                           3 * fixed_nodes + 2]).flatten().astype(int)
    dirichlet_values = 0 * np.ones_like(fixed_dofs, dtype=float)

    bc = bound_cond.BC(force=force,
                       fixed_dofs=fixed_dofs,
                       dirichlet_values=dirichlet_values)

    mat_prop = mat_lib.create_material_with_defaults(name="TableMaterial",
                                                    youngs_modulus=youngs_modulus,
                                                    poissons_ratio=poissons_ratio, mass_density=1.0)
    elem_body_force = None
    return mesh, mat_prop, bc, elem_body_force

def createGEGrabCADProblem(nDOFDesired: int = 50000, axialLoad = 10000): 

    # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, './Models/GEGrabCAD/GEGrabCAD.STL')

    nElemsDesired = nDOFDesired/3    # estimate
    mesh = hex_mesher.HexMesher()

    mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    mesh.createEdofMatStructural()

    fixedTri = list(range(3245, 3533 + 1))
    fixed_nodes = mesh.get_nodes_on_triangles(fixedTri)
    mesh.node_indices[fixed_nodes, 3] = 1 # for plotting
    fixed_dofs = np.array([3 * fixed_nodes,
                3 * fixed_nodes + 1,
                3 * fixed_nodes + 2]).flatten().astype(int)
    dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

    forceTri = [5071,5072,5073,5074,5075,5076,5077,5078,5079,5080,5081,5082,5083,5084,5085,5086,5087,5088,5089,5090,5091,5092,5093,5094,5095,5096,5097,5098,5099,5100,5101,5102,5103,5104,5105,5106,5107,5108,5109,5110,5111,5112,5113,5114,5115,5116,5117,5118,5119,5120,5121,5122,5123,5124,5125,5126,5127,5128,5129,5130,5131,5132,5133,5134,5135,5136,5137,5138,5139,5140,5141,5142,5143,5144,5145,5146,5147,5148,5149,5150,5151,5152,5153,5154,5155,5156,5157,5158,5159,5160,5161,5162,5163,5164,5165,5166,5167,5168,5169,5170,5171,5172,5173,5174,5175,5176,5177,5178,5179,5180,5181,5182,5183,5184,5185,5186,5187,5188,5189,5190,5191,5192,5193,5194,5195,5196,5197,5198,5199,5200,5201,5202,5203,5204,5205,5206,5207,5208,5209,5210,5211,5212,5213,5214,5215,5216,5217,5218,5219,5220,5221,5222,5223,5224,5225,5226,5227,5228,5229,5230,5231,5232,5233,5234,5235,5236,5237,5238,5239,5240,5241,5242,5243,5244,5245,5246,5247,5248,5249,5250,5251,5252,5253,5254,5255,5256,5257,5258,5259,5260,5261,5262,5263,5264,5265,5266,5267,5268,5269,5270,5271,5272,5273,5274,5275,5276,5277,5278,5279,5280,5281,5282,5283,5284,5285,5286,6351,6352,6353,6354,6355,6356,6357,6358,6359,6360,6361,6362,6363,6364,6365,6366,6367,6368,6369,6370,6371,6372,6373,6374,6375,6376,6377,6378,6379,6380,6381,6382,6383,6384,6385,6386,6387,6388,6389,6390,6391,6392,6393,6394,6395,6396,6397,6398,6399,6400,6401,6402,6403,6404,6405,6406,6407,6408,6409,6410,6411,6412,6413,6414,6415,6416,6417,6418,6419,6420,6421,6422,6423,6424,6425,6426,6427,6428,6429,6430,6431,6432,6433,6434,6435,6436,6437,6438,6439,6440,6441,6442,6443,6444,6445,6446,6447,6448,6449,6450,6451,6452,6453,6454,6455,6456,6457,6458,6459,6460,6461,6462,6463,6464,6465,6466,6467,6468,6469,6470,6471,6472,6473,6474,6475,6476,6477,6478,6479,6480,6481,6482,6483,6484,6485,6486,6487,6488,6489,6490,6491,6492,6493,6494,6495,6496,6497,6498,6499,6500,6501,6502,6503,6504,6505,6506,6507,6508,6509,6510,6511,6512,6513,6514,6515,6516,6517,6518,6519,6520,6521,6522,6523,6524,6525,6526,6527,6528,6529,6530,6531,6532,6533,6534,6535,6536,6537,6538,6539,6540,6541,6542,6543,6544,6545,6546,6547,6548,6549,6550,6551,6552,6553,6554,6555,6556,6557,6558,6559,6560,6561,6562,6563,6564,6565,6566]
    load_nodes = mesh.get_nodes_on_triangles(forceTri)


    load_dofs = 3 * load_nodes   # x direction

    load_per_dof = axialLoad/len(load_nodes)
    force = np.zeros(3*mesh.num_nodes)
    force[load_dofs] = load_per_dof

    mat_prop = mat_lib.create_material_with_defaults(name="GEMaterial",
                                                      youngs_modulus=1,
                                                      poissons_ratio=0.3, mass_density=1.0)

    nElems = mesh.num_elems
    elemVolume = mesh.elem_size[0]*mesh.elem_size[1]*mesh.elem_size[2]
    totalMass = nElems * elemVolume * mat_prop.mass_density
    print("Total mass of GEGrabCAD: {:.2f} kg".format(totalMass))
    elem_body_force = None

    # All constraints are implemented using the constraint matrix
    # Therefore fixed_dofs and dirichlet_values are empty
    bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

    return mesh, mat_prop, bc, elem_body_force

# ----------------------------------------
