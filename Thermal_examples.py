import numpy as np
import os
import enum
script_dir = os.path.dirname(os.path.abspath(__file__))
from PyTOImports import *
import pyvista as pv

class MMTOThermalExamples(enum.Enum):
    EdgeCantilever = enum.auto()
    EdgeCantilever_TempBC = enum.auto()  # Edge cantilever with temperature boundary conditions
    BliskSection = enum.auto()  # Blisk section thermal problem
    LBracketThermal = enum.auto()
    # Add more as needed

def getMMTOThermalProblem(problem: MMTOThermalExamples,nDOFDesired: int = 20000, mesh=None, **kwargs):
    """
    Returns a thermal problem based on the given problem name.
    If mesh is provided, it will be reused for the thermal problem.
    """
    if problem == MMTOThermalExamples.EdgeCantilever:
        return createEdgeCantileverThermalProblem(mesh=mesh, nDOFDesired=nDOFDesired, **kwargs)
    elif problem == MMTOThermalExamples.BliskSection:
        return createBliskBladeThermalProblem_TempBC(mesh=mesh, nDOFDesired=nDOFDesired, **kwargs)
    elif problem == MMTOThermalExamples.EdgeCantilever_TempBC:
        return createEdgeCantileverThermalProblem_TempBC(mesh=mesh, nDOFDesired=nDOFDesired, **kwargs)
    elif problem == MMTOThermalExamples.LBracketThermal:
        return createLBracketThermalProblem(mesh=mesh, nDOFDesired=nDOFDesired)
    else:
        raise ValueError("Invalid thermal example name.")

def createEdgeCantileverThermalProblem(mesh=None, nDOFDesired: int = 20000, L: list = [0.4, 0.2, 0.1],
                                       thermal_conductivity=1.0, heat_source=1000.0, boundary_temp=300.0):
    """
    Creates a thermal edge cantilever problem using the same mesh as the structural problem if provided.
    """
    # If mesh is not provided, create a new one
    if mesh is None:
        nVoxelsDesired = nDOFDesired / 3
        alpha = (nVoxelsDesired / (L[0] * L[1] * L[2])) ** (1 / 3)
        nelx = round(alpha * L[0])
        nely = round(alpha * L[1])
        nelz = round(alpha * L[2])
        mesh = hex_mesher.HexMesher()
        mesh.grid_mesh(num_elems=(nelx, nely, nelz),
                       elem_size=(L[0] / nelx, L[1] / nely, L[2] / nelz))
        mesh.createEdofMatThermal()
    else:
        # Use the provided mesh and create thermal edofMat if needed
        if not hasattr(mesh, 'edofMat') or mesh.edofMat is None:
            mesh.createEdofMatThermal()

    # Thermal material properties
    mat_prop = mat_lib.create_material_with_defaults(
        name="Thermal Material",
        thermal_conductivity=thermal_conductivity
    )

    # Boundary conditions: fix temperature at left face (x=0), heat source on right face (x=L)
    fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0, True)  # x = 0 plane
    fixed_dofs = fixed_nodes.astype(int)
    dirichlet_values = boundary_temp * np.ones_like(fixed_dofs, dtype=float)

    # Heat source: apply to nodes on x = xMax plane
    source_nodes = mesh.getNodesOnBoundingBoxPlane(0, False)
    heat_source_vector = np.zeros(mesh.num_nodes)
    heat_source_vector[source_nodes] = heat_source / len(source_nodes)

    bc = bound_cond.BC(
        force=heat_source_vector,  # For thermal, this is the heat source
        fixed_dofs=fixed_dofs,
        dirichlet_values=dirichlet_values
    )

    return mesh, mat_prop, bc
def createEdgeCantileverThermalProblem_TempBC(mesh=None, nDOFDesired: int = 5000, L: list = [0.4, 0.2, 0.1],
                                              thermal_conductivity=1.0, temp_left=275.0, temp_right=50.0):
    """
    Creates a thermal edge cantilever problem with prescribed temperatures on both ends.
    """
    # If mesh is not provided, create a new one
    if mesh is None:
        nVoxelsDesired = nDOFDesired / 3
        alpha = (nVoxelsDesired / (L[0] * L[1] * L[2])) ** (1 / 3)
        nelx = round(alpha * L[0])
        nely = round(alpha * L[1])
        nelz = round(alpha * L[2])
        mesh = hex_mesher.HexMesher()
        mesh.grid_mesh(num_elems=(nelx, nely, nelz),
                       elem_size=(L[0] / nelx, L[1] / nely, L[2] / nelz))
        mesh.createEdofMatThermal()
    else:
        if not hasattr(mesh, 'edofMat') or mesh.edofMat is None:
            mesh.createEdofMatThermal()

    # Thermal material properties
    mat_prop = mat_lib.create_material_with_defaults(
        name="Thermal Material",
        thermal_conductivity=thermal_conductivity
    )

    # Boundary conditions: fix temperature at left face (x=0) and right face (x=L)
    fixed_nodes_left = mesh.getNodesOnBoundingBoxPlane(0, True)  # x = 0 plane
    fixed_nodes_right = mesh.getNodesOnBoundingBoxPlane(0, False)  # x = L plane
    fixed_dofs = np.concatenate([fixed_nodes_left, fixed_nodes_right]).astype(int)
    dirichlet_values = np.concatenate([
        temp_left * np.ones_like(fixed_nodes_left, dtype=float),
        temp_right * np.ones_like(fixed_nodes_right, dtype=float)
    ])

    # No heat source
    heat_source_vector = np.zeros(mesh.num_nodes)

    bc = bound_cond.BC(
        force=heat_source_vector,  # For thermal, this is the heat source
        fixed_dofs=fixed_dofs,
        dirichlet_values=dirichlet_values
    )

    return mesh, mat_prop, bc
def createBliskBladeThermalProblem_TempBC(mesh=None, nDOFDesired: int = 10000, 
                                          thermal_conductivity=1.0, temp_tip=1000.0, temp_bottom=50.0):
    """
    Creates a thermal blisk blade problem with prescribed temperatures at the tip and bottom.
    Uses mesh from the structural blisk blade example if provided.
    """
    # If mesh is not provided, create it using the structural example
    if mesh is None:
        from Structural_examples import createBliskSectionProblem
        mesh, _, _, _ = createBliskSectionProblem(nDOFDesired=nDOFDesired)
        mesh.createEdofMatThermal()
    else:
        if not hasattr(mesh, 'edofMat') or mesh.edofMat is None:
            mesh.createEdofMatThermal()

    # Thermal material properties
    mat_prop = mat_lib.create_material_with_defaults(
        name="Thermal Material",
        thermal_conductivity=thermal_conductivity
    )

    # Identify tip and bottom nodes
    vertices = mesh.node_xyz
    x_min = np.min(vertices[:, 0])
    x_max = np.max(vertices[:, 0])
    tol = 1e-8 * (x_max - x_min)

    # Bottom nodes: x == x_min
    bottom_nodes = np.where(np.isclose(vertices[:, 0], x_min, atol=tol))[0]
    # Tip nodes: x == x_max
    tip_nodes = np.where(np.isclose(vertices[:, 0], x_max, atol=tol))[0]

    fixed_dofs = np.concatenate([bottom_nodes, tip_nodes]).astype(int)
    dirichlet_values = np.concatenate([
        temp_bottom * np.ones_like(bottom_nodes, dtype=float),
        temp_tip * np.ones_like(tip_nodes, dtype=float)
    ])

    # No heat source
    heat_source_vector = np.zeros(mesh.num_nodes)

    bc = bound_cond.BC(
        force=heat_source_vector,
        fixed_dofs=fixed_dofs,
        dirichlet_values=dirichlet_values
    )

    return mesh, mat_prop, bc

def createLBracketThermalProblem(mesh=None, nDOFDesired: int = 10000, temp_right=150, temp_top=50, thermal_conductivity=1.0):
    """
    Creates a thermal problem setup for an L-bracket topology optimization.
    If mesh is not provided, creates it using the structural LBracket problem.
    Applies Dirichlet BCs: temp_right at rightmost end (x = xMax), temp_top at topmost end (y = yMax).
    """
    # If mesh is not provided, create it using the structural example
    if mesh is None:
        from Structural_examples import createLBracketProblem
        mesh, _, _, _ = createLBracketProblem(nDOFDesired=nDOFDesired)
        mesh.createEdofMatThermal()
    else:
        if not hasattr(mesh, 'edofMat') or mesh.edofMat is None:
            mesh.createEdofMatThermal()

    node_pts = mesh.node_xyz

    # Dirichlet BC at y = yMax (topmost end)
    top_nodes = np.where(node_pts[:, 1] == np.max(node_pts[:, 1]))[0]  # y = yMax plane
    #top_nodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False) , np.where((node_pts[:, 1] >= 0.36))[0]) # hard coded    
    
    # Dirichlet BC at x = xMax (rightmost end)
    #right_nodes = np.where(node_pts[:, 0] == np.max(node_pts[:, 0]))[0]  # x = xMax plane
    right_nodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False) , np.where((node_pts[:, 1] >= 0.36))[0]) # hard coded    
    

    fixed_dofs = np.concatenate([top_nodes, right_nodes]).astype(int)
    dirichlet_values = np.concatenate([
        temp_top * np.ones_like(top_nodes, dtype=float),
        temp_right * np.ones_like(right_nodes, dtype=float)
    ])

    mesh.node_indices[top_nodes, 3] = 1  # for plotting
    mesh.node_indices[right_nodes, 3] = 2  # for plotting

    # No heat source (Neumann BCs removed)
    force = np.zeros(mesh.num_nodes)

    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)
    # Thermal material properties
    mat_prop = mat_lib.create_material_with_defaults(
        name="Thermal Material",
        thermal_conductivity=thermal_conductivity
    )
    return mesh, mat_prop, bc

def plot_thermal_bc(mesh, bc, title="Thermal BCs on Elements"):
    """
    Plots mesh elements, highlighting those with Dirichlet BCs.
    """
    # Get element centers and edofMat
    elem_centers = mesh.elem_centers  # shape: (num_elems, 3)
    edofMat = mesh.edofMat  # shape: (num_elems, nodes_per_elem)
    fixed_nodes = set(bc.fixed_dofs)

    # Find elements with at least one fixed node
    highlight_elems = [i for i in range(mesh.num_elems) if any(n in fixed_nodes for n in edofMat[i])]
    highlight_elems = np.array(highlight_elems)

    # Create PyVista point cloud for all elements
    cloud = pv.PolyData(elem_centers)
    colors = np.full(mesh.num_elems, 0)
    colors[highlight_elems] = 1  # 1 for elements with BCs

    # Plot
    plotter = pv.Plotter()
    plotter.add_title(title)
    plotter.add_points(elem_centers, scalars=colors, cmap="coolwarm", point_size=12, render_points_as_spheres=True)
    plotter.add_axes()
    plotter.show()
import pyvista as pv
import numpy as np

import pyvista as pv
import numpy as np

def plot_thermal_bc_voxel(mesh, bc, title="Thermal BCs on Voxel Mesh", auto_close=True, save_path=None):
    """
    Plots the voxel mesh (green, semi-transparent) and overlays nodes with Dirichlet (fixed temperature, black)
    and Neumann (heat source, red) BCs, just like the structural BC plot.
    """
    vertices = mesh.node_xyz  # (num_nodes, 3)
    elemArray = mesh.elemArray  # (num_elems, 8) node indices for each voxel

    plotter = pv.Plotter()
    plotter.add_title(title, font_size=8)

    # Build the unstructured grid for all voxels
    n_elems = elemArray.shape[0]
    n_nodes_per_elem = elemArray.shape[1]
    cells = np.hstack([np.full((n_elems, 1), n_nodes_per_elem), elemArray]).astype(np.int64)
    cells = cells.flatten()
    celltypes = np.full(n_elems, pv.CellType.HEXAHEDRON, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, vertices)

    # Plot the voxel mesh (green, semi-transparent)
    plotter.add_mesh(
        grid,
        color='lightgreen',
        show_edges=True,
        edge_color='black',
        line_width=1,
        opacity=0.7
    )

import pyvista as pv
import numpy as np

import pyvista as pv
import numpy as np

import pyvista as pv
import numpy as np

import pyvista as pv
import numpy as np

import pyvista as pv
import numpy as np

def plot_thermal_mesh_with_bc(mesh, bc, title="Thermal BCs on Voxel Mesh", auto_close=True, save_path=None):
    """
    Plots the voxel mesh (green, fully opaque) and overlays nodes with Dirichlet (fixed temperature, black for left face,
    red for right face). No temperature labels are shown.
    """
    vertices = mesh.node_xyz  # (num_nodes, 3)
    elemArray = mesh.elemArray  # (num_elems, 8) node indices for each voxel

    plotter = pv.Plotter()
    plotter.add_title(title, font_size=8)

    # Build the unstructured grid for all voxels
    n_elems = elemArray.shape[0]
    n_nodes_per_elem = elemArray.shape[1]
    cells = np.hstack([np.full((n_elems, 1), n_nodes_per_elem), elemArray]).astype(np.int64)
    cells = cells.flatten()
    celltypes = np.full(n_elems, pv.CellType.HEXAHEDRON, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, vertices)

    # Plot the voxel mesh (green, fully opaque)
    plotter.add_mesh(
        grid,
        color='lightgreen',
        show_edges=True,
        edge_color='black',
        line_width=1,
        opacity=1.0  # Fully opaque
    )

    # Identify left and right face nodes and their temperatures
    if hasattr(bc, 'fixed_dofs') and bc.fixed_dofs is not None and len(bc.fixed_dofs) > 0 and \
       hasattr(bc, 'dirichlet_values') and bc.dirichlet_values is not None:
        fixed_nodes = np.array(bc.fixed_dofs)
        dirichlet_values = np.array(bc.dirichlet_values)

        # Get left and right face nodes using mesh geometry
        x_coords = vertices[fixed_nodes, 0]
        x_min = np.min(vertices[:, 0])
        x_max = np.max(vertices[:, 0])
        tol = 1e-8 * (x_max - x_min)

        left_mask = np.isclose(x_coords, x_min, atol=tol)
        right_mask = np.isclose(x_coords, x_max, atol=tol)

        left_nodes = fixed_nodes[left_mask]
        right_nodes = fixed_nodes[right_mask]

        # Plot left face nodes as black
        if len(left_nodes) > 0:
            plotter.add_points(vertices[left_nodes], color='black', point_size=12, render_points_as_spheres=True)

        # Plot right face nodes as red
        if len(right_nodes) > 0:
            plotter.add_points(vertices[right_nodes], color='red', point_size=12, render_points_as_spheres=True)

    plotter.add_axes(
        xlabel='X',
        ylabel='Y',
        zlabel='Z',
        line_width=2,
        labels_off=False,
        color='black'
    )

    if save_path:
        plotter.screenshot(save_path)
        plotter.close()
    else:
        plotter.show(interactive_update=not auto_close, auto_close=auto_close)

# Example usage:
# mesh, mat_prop, bc = getMMTOThermalProblem(MMTOThermalExamples.EdgeCantilever, mesh=structural_mesh)