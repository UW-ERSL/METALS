import torch
import numpy as np
import pandas as pd


class ReadMaterialData:
    def __init__(self, excel_file):
        self.excel_file = excel_file
        file_lower = excel_file.lower()
        if "temp" in file_lower:
            self.mode = "tempdependent"
            self.trainingData, self.dataInfo, self.dataIdentifier, self.trainInfo = self.preprocessData_tempdependent()
        elif "lbracket" in file_lower:
            self.mode = "structuralyield"
            self.trainingData, self.dataInfo, self.dataIdentifier, self.trainInfo = self.preprocessData_structuralyield()
        elif "cost" in file_lower:
            self.mode = "structuralcost"
            self.trainingData, self.dataInfo, self.dataIdentifier, self.trainInfo = self.preprocessData_structuralcost()
        else:
            self.mode = "structural"
            self.trainingData, self.dataInfo, self.dataIdentifier, self.trainInfo= self.preprocessData_structural()
    def preprocessData_structuralyield(self):
        df = pd.read_excel(self.excel_file)
        rawData = df.iloc[:, [1, 2, 3]].to_numpy()
        feature_names = ['MassDensity', 'ElasticModulus', 'YieldStrength']
       
        trainInfo = np.log10(rawData)
        dataScaleMax = torch.tensor(np.max(trainInfo, axis=0))
        dataScaleMin = torch.tensor(np.min(trainInfo, axis=0))
        normalizedData = (torch.tensor(trainInfo) - dataScaleMin) / (dataScaleMax - dataScaleMin)
        trainingData = normalizedData.clone().float()
        dataInfo = {}
        for i, name in enumerate(feature_names):
            dataInfo[name] = {'idx': i, 'scaleMin': dataScaleMin[i], 'scaleMax': dataScaleMax[i]}
        dataIdentifier = {'name': df[df.columns[0]]}
        return trainingData, dataInfo, dataIdentifier, trainInfo
    
    def preprocessData_structuralcost(self):
        df = pd.read_excel(self.excel_file)
 
        rawData = df.iloc[:, [1, 2, 3]].to_numpy()
        feature_names = ['MassDensity', 'ElasticModulus', 'Cost']
        trainInfo = np.log10(rawData)
        dataScaleMax = torch.tensor(np.max(trainInfo, axis=0))
        dataScaleMin = torch.tensor(np.min(trainInfo, axis=0))
        normalizedData = (torch.tensor(trainInfo) - dataScaleMin) / (dataScaleMax - dataScaleMin)
        trainingData = normalizedData.clone().float()
        dataInfo = {}
        for i, name in enumerate(feature_names):
            dataInfo[name] = {'idx': i, 'scaleMin': dataScaleMin[i], 'scaleMax': dataScaleMax[i]}
        dataIdentifier = {
            'name': df[df.columns[0]],
        }
        return trainingData, dataInfo, dataIdentifier, trainInfo

    def preprocessData_structural(self):
        df = pd.read_excel(self.excel_file)
        rawData = df.iloc[:, [5, 10]].to_numpy()
        feature_names = ['MassDensity', 'ElasticModulus']
        trainInfo = np.log10(rawData)
        dataScaleMax = torch.tensor(np.max(trainInfo, axis=0))
        dataScaleMin = torch.tensor(np.min(trainInfo, axis=0))
        normalizedData = (torch.tensor(trainInfo) - dataScaleMin) / (dataScaleMax - dataScaleMin)
        trainingData = normalizedData.clone().float()
        dataInfo = {}
        for i, name in enumerate(feature_names):
            dataInfo[name] = {'idx': i, 'scaleMin': dataScaleMin[i], 'scaleMax': dataScaleMax[i]}
        dataIdentifier = {
            'name': df[df.columns[0]],
            'className': df[df.columns[1]],
            'classID': df[df.columns[2]]
        }
        return trainingData, dataInfo, dataIdentifier, trainInfo

    def preprocessData_tempdependent(self):
        df = pd.read_excel(self.excel_file)
        # MassDensity (6th col, index 5), Ea (13th, 12), Eb (14th, 13), Ec (15th, 14), Ed (16th, 15)
        rawData = df.iloc[:, [5, 12, 13, 14, 15, 16]].to_numpy()
        feature_names = ['MassDensity', 'Ea', 'Eb', 'Ec', 'Ed','ThermalConductivity']

        # Only log-transform MassDensity (col 0), min-max normalize the rest
        mass_density = rawData[:, 0]
        mass_density = np.where(mass_density <= 0, 1e-8, mass_density)
        log_mass_density = np.log10(mass_density)
        md_min, md_max = log_mass_density.min(), log_mass_density.max()
        norm_mass_density = (log_mass_density - md_min) / (md_max - md_min)

        poly_coeffs = rawData[:, 1:5]
        poly_min = poly_coeffs.min(axis=0)
        poly_max = poly_coeffs.max(axis=0)
        norm_poly_coeffs = np.zeros_like(poly_coeffs)
        for i in range(4):
            if poly_max[i] == poly_min[i]:
                norm_poly_coeffs[:, i] = poly_coeffs[:, i]
                print(f"{feature_names[i+1]} not normalized (constant value).")
            else:
                norm_poly_coeffs[:, i] = (poly_coeffs[:, i] - poly_min[i]) / (poly_max[i] - poly_min[i])
        # Thermal conductivity normalization
        thermal_cond = rawData[:, 5]
        tc_min, tc_max = thermal_cond.min(), thermal_cond.max()
        if tc_max == tc_min:
            norm_thermal_cond = thermal_cond
            print("Thermal conductivity not normalized (constant value).")
        else:
            norm_thermal_cond = (thermal_cond - tc_min) / (tc_max - tc_min)

        normalizedData = np.column_stack([norm_mass_density, norm_poly_coeffs, norm_thermal_cond])
        trainingData = torch.tensor(normalizedData).float()

        dataInfo = {
            'MassDensity': {'idx': 0, 'scaleMin': md_min, 'scaleMax': md_max, 'is_log': True},
            'Ea': {'idx': 1, 'scaleMin': poly_min[0], 'scaleMax': poly_max[0], 'is_log': False},
            'Eb': {'idx': 2, 'scaleMin': poly_min[1], 'scaleMax': poly_max[1], 'is_log': False},
            'Ec': {'idx': 3, 'scaleMin': poly_min[2], 'scaleMax': poly_max[2], 'is_log': False},
            'Ed': {'idx': 4, 'scaleMin': poly_min[3], 'scaleMax': poly_max[3], 'is_log': False},
            'ThermalConductivity': {'idx': 5, 'scaleMin': tc_min, 'scaleMax': tc_max, 'is_log': False}
        }
        dataIdentifier = {
            'name': df[df.columns[0]],
            'className': df[df.columns[1]],
            'classID': df[df.columns[2]]
        }
        trainInfo = normalizedData
    
        return trainingData, dataInfo, dataIdentifier, trainInfo