import numpy as np
import torch
import time
from MMTO_examples import MMTOExamples, getMMTOProblem
from materialEncoder import MaterialEncoder
from MMTO_obj_cons_sensitivities import (
    compute_mmto_objective_and_gradient,
    compute_mmto_constraint_and_gradient,
)
from PyTOImports import *
import matplotlib.pyplot as plt

from enum import Enum

# Load saved results
zOptimalPts = torch.load("results2/zOptimalPts.pt")
xOptimal = np.load("results2/xOptimal.npy")

# Load the TO problem and encoder (make sure to use the same problem and Excel file as before)
to_problem = MMTOExamples.GEGrabCAD_Compliance_Mass
mesh_structural, mat_prop_struct, bc_struct, elem_body_force, to_params, vae_params = getMMTOProblem(to_problem)
matEncoder = MaterialEncoder(vae_params)
matEncoder.readExcel(to_params.MaterialsExcelFile)
matEncoder.loadAutoencoderFromFile(to_params.MaterialsExcelFile.replace('.xlsx', '.nt'))

# Decode material properties
decoded = matEncoder.vaeNet.decoder(zOptimalPts)
material_properties = matEncoder.getMaterialProperties(decoded)
Youngs_Modulus = material_properties['Youngs_Modulus'].detach().cpu().numpy()

# Set up FEA solver
solver = linear_solvers.Solvers.PARDISO
dsolver = deflation.DeflationSolver()
fe_solver_structural = hex_structural_fea.HexStructuralFEA(
    mesh=mesh_structural,
    mat_prop=mat_prop_struct,
    bc=bc_struct,
    solver=solver,
    dsolver=dsolver,
    rtol=1e-8,
    elem_body_force=elem_body_force)

fe_solver_structural.mesh.setPseudoDensity(xOptimal)
# fe_solver_structural.plot_elem_field(Youngs_Modulus, title='YoungModulus', colormap='viridis')
plt.show()
excel_file = to_params.MaterialsExcelFile

if excel_file == './DataConstantTemperature/5MaterialsCantilever.xlsx':
    material_colors = {
        0: '#fe4d02', 
        1: '#e6fd1a',
        2: '#1dfde1',
        3: '#004fff',
        4: '#020a86',
    }
elif excel_file == './DataConstantTemperature/3MaterialsBridgev2.xlsx' or excel_file == './DataConstantTemperature/3MaterialsBridge.xlsx':
    material_colors = {
        0: '#04fd05', 
        1: '#0505f0',
        2: '#ef0711',
    }
elif excel_file == './DataConstantTemperature/20MaterialsTeledyne.xlsx' or excel_file == './DataConstantTemperature/20MaterialsTeledyneSimple.xlsx':
    material_colors = {
        1:  '#e6194b',  # IN 100 – vibrant cyan  
        5:  '#4357d8',  # Nickel Alloy 263 – vibrant yellow 
        10: "#fe9800",  # Haynes 214 – vibrant magenta  
        11: "#fafa02",  # 17-4PH SS – vibrant red '#42d4f4' '#04fd05' 
        13: '#04fd05',  # Nitronic 60 – vibrant blue '#04fd05'
        17: "#32f0d3",  # 7068 Al – vibrant orange   
        18: "#2206F7",  # 7075 Al – neon green '#42d4f4' "#fe6a00" 
        19: '#640c7f',  # 6061 Al – deep violet '#42d4f4'
    }
    material_colors.update({
        0:  '#a00000',  # MAR M247 – richer crimson
        2:  '#5a9bd5',  # Inconel 718 – brighter steel blue
        3:  '#8399a9',  # Inconel 625 – enhanced slate gray
        4:  '#688f3f',  # Ultimet (r) – livelier olive green
        6:  '#b86b3f',  # Hastelloy 276 – warm sienna
        7:  "#c0ff72",  # Hastelloy C-4 – vibrant olive
        8:  '#3f6f6f',  # Incoloy 27-7MO – teal-slate hybrid
        9:  '#a0d6b4',  # Incoloy 825 – fresh seafoam
        12: "#000000",  # Grade 304 SS – brighter steel blue
        14: '#e0c28c',  # Ti-10%V-2%Fe-3%Al – golden tan
        15: '#e09b5f',  # Ti-6%Al-4%V – vibrant peru
        16: '#f0c49c',  # Ti Grade 4 – warm burlywood
    })

elif excel_file == './DataConstantTemperature/8Materials.xlsx':
    material_colors = {
        0: '#e6194b',  # vibrant red
        1: '#3cb44b',  # vibrant green
        2: '#ffe119',  # vibrant yellow
        3: '#4363d8',  # vibrant blue
        4: '#f58231',  # vibrant orange
        5: '#911eb4',  # vibrant purple
        6: '#42d4f4',  # vibrant cyan
        7: '#f032e6',  # vibrant magenta
    }
elif excel_file == './DataConstantTemperature/3Materials.xlsx':
    material_colors = {
        0: '#0201fc', 
        1: '#f60004',
        2: '#080101',
    }
else:
    # Default colors for up to 20 materials
    default_colors = plt.cm.get_cmap('tab20', matEncoder.nMaterials)
    material_colors = {i: default_colors(i) for i in range(matEncoder.nMaterials)}
material_indices = matEncoder.getClosestRealMaterialIndex(zOptimalPts) 
colors = [material_colors[int(idx.item()) if hasattr(idx, "item") else int(idx)] for idx in material_indices]
import matplotlib.colors as mcolors
rgb_colors = np.array([mcolors.to_rgb(c) for c in colors])

fe_solver_structural.plot_elem_field(material_indices, title='Real Materials', colors=rgb_colors)
with torch.no_grad():
    matEncoder.training_latents = matEncoder.vaeNet.encoder(matEncoder.scaledMaterialData).cpu()
zRealPoints = matEncoder.training_latents
# Plot latent space with designs and training data
matEncoder.plotLSR(zRealPoints.detach().cpu().numpy(), zOptimalPts, xDesign=xOptimal)