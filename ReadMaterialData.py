import torch
import numpy as np
import pandas as pd

class ReadMaterialData:
    def __init__(self, excel_file):
        self.excel_file = excel_file
        self.scaledMaterialData, self.materialAttributes, self.materialNames, self.trainInfo = self.preprocessData()

    def preprocessData(self):
        df = pd.read_excel(self.excel_file, header=None)
        # First column: material names, but first cell is "Attribute", second is "Material/Units"
        # Second column onwards: attribute names (first cell), units (second cell), values (third row onwards)

        # Extract attribute names and units from second column onwards
        attribute_names = df.iloc[0, 1:].tolist()
        units = df.iloc[1, 1:].tolist()
        # Extract material names from first column, starting from third row
        material_names = df.iloc[2:, 0].tolist()
        # Extract attribute values from second column onwards, starting from third row
        values = df.iloc[2:, 1:].to_numpy(dtype=float)

        # Custom log transform: log10(x + min + 10)
        
        min_vals = np.min(values, axis=0)
        log_values = np.log10(values - min_vals + 10)
        # Min-max normalization
        dataScaleMin = log_values.min(axis=0)
        dataScaleMax = log_values.max(axis=0)

        normalizedData = (log_values - dataScaleMin) / (dataScaleMax - dataScaleMin + 1e-12)
        scaledMaterialData = torch.tensor(normalizedData).float()

        # Build materialAttributes dictionary
        materialAttributes = {}
        for i, name in enumerate(attribute_names):
            materialAttributes[name] = {
                'idx': i,
                'unit': units[i],
                'scaleMin': dataScaleMin[i],
                'scaleMax': dataScaleMax[i],
                'minAdded': min_vals[i]
            }

        # Identifier: first column is material name, second/third columns can be className/classID if present
        materialNames = {}
        materialNames['name'] = material_names
      
        trainInfo = log_values

        return scaledMaterialData, materialAttributes, materialNames, trainInfo