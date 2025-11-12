import os
from openpyxl import Workbook

# Final normalized material data
materials = {
    "A": [0.38, 0.61, 0.36],
    "B": [0.32, 0.95, 1.00],
    "C": [0.67, 0.80, 0.90],  # Updated cost
    "D": [0.80, 0.89, 0.67],
    "E": [0.82, 0.70, 0.48],
    "F": [0.76, 1.00, 0.72],
    "G": [1.00, 0.88, 0.58],
    "H": [1.00, 1.00, 1.00]   # Anchor material
}

# Create folder if it doesn't exist
folder_path = "DataConstantTemperature"
os.makedirs(folder_path, exist_ok=True)

# Create workbook and sheet
wb = Workbook()
ws = wb.active
ws.title = "Materials"

# Write header rows
ws.append(["Attributes", "Density", "Youngs_Modulus", "Cost"])
ws.append(["Material/Units", "kg/m^3", "Pa", "USD/kg"])

# Write data rows
for label, values in materials.items():
    ws.append([label] + values)

# Save workbook
output_path = os.path.join(folder_path, "8Materials.xlsx")
wb.save(output_path)