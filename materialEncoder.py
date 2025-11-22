from networks import VariationalAutoencoder
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from InterpolationFunctions import bezierInterpolation, bezierInterpolation_torch, TMin, TMax
from InterpolationFunctions import logBezierInterpolation, logBezierInterpolation_torch
from sklearn.decomposition import PCA
import matplotlib.cm as cm
class MaterialEncoder:
    def __init__(self, vae_params):
        self.nAttributes = 0
        self.vae_params = vae_params
        # self.RELATIVE_MATERIAL_MIN_VAL = 1e-6

    def readExcel(self, excel_file):
        self.excel_file = excel_file
        self.preprocessData()

        self.nMaterials = self.scaledMaterialData.shape[0]
        self.nAttributes = self.scaledMaterialData.shape[1]

        self.vaeSettings = {
            'encoder': {'inputDim': self.nAttributes, 'hiddenDim': self.vae_params.vae_hiddenDim, 'latentDim': self.vae_params.latentDim},
            'decoder': {'latentDim': self.vae_params.latentDim, 'hiddenDim': self.vae_params.vae_hiddenDim, 'outputDim': self.nAttributes}
        }
        self.vaeNet = VariationalAutoencoder(self.vaeSettings)

    def preprocessData(self):
        df = pd.read_excel(self.excel_file, header=None)
        attribute_names = df.iloc[0, 1:].tolist()
        units = df.iloc[1, 1:].tolist()
        material_names = df.iloc[2:, 0].tolist()
        values = df.iloc[2:, 1:].to_numpy(dtype=float)

        self.rawData = values

        normalizedData = np.zeros_like(values)
        dataScaleMin = np.zeros(values.shape[1])
        dataScaleMax = np.zeros(values.shape[1])

        for i, name in enumerate(attribute_names):
            col = values[:, i]
            if name in ['E0', 'E1', 'E2', 'E3', 'Y0', 'Y1', 'Y2', 'Y3']:
                log_col = np.log10(col)
                dataScaleMin[i] = log_col.min()
                dataScaleMax[i] = log_col.max()
                normalizedData[:, i] = (log_col - dataScaleMin[i]) / (dataScaleMax[i] - dataScaleMin[i] + 1e-12)
            else:
                norm_col = col
                dataScaleMin[i] = norm_col.min()
                dataScaleMax[i] = norm_col.max()
                normalizedData[:, i] = (norm_col - dataScaleMin[i]) / (dataScaleMax[i] - dataScaleMin[i] + 1e-12)

        scaledMaterialData = torch.tensor(normalizedData).float()

        materialAttributes = {}
        for i, name in enumerate(attribute_names):
            materialAttributes[name] = {
                'idx': i,
                'unit': units[i],
                'scaleMin': dataScaleMin[i],
                'scaleMax': dataScaleMax[i],
            }

        self.materialNames = material_names
        self.scaledMaterialData = scaledMaterialData
        self.materialAttributes = materialAttributes

        attribute_means = normalizedData.mean(axis=0)
        attribute_stds = normalizedData.std(axis=0)
        self.attribute_means = attribute_means
        self.attribute_stds = attribute_stds

    def largestEncodingErrorPercent(self):
        self.vaeNet.encoder.isTraining = False
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            diffMatrix = self.scaledMaterialData - decoded

            largestErrorPercent = 100.0 * torch.max(torch.abs(diffMatrix)).item()
        self.vaeNet.encoder.isTraining = True
        return largestErrorPercent

    def printYieldStrengthDropAtTempLimit(self):
        """Print the ratio of Y0 to Y at Temp_Limit for each material."""
        if 'Y0' not in self.materialAttributes or 'Temp_Limit' not in self.materialAttributes:
            print("ERROR: Missing required attributes (Y0 or Temp_Limit)")
            return
        
        print("\n" + "=" * 60)
        print("Yield Strength Drop at Temperature Limit")
        print("=" * 60)
        print("{:<20} {:>12} {:>12} {:>12} {:>12}".format("Material", "T_limit (C)", "Y0 (Pa)", "Y@T_limit", "Y0/Y ratio"))
        print("-" * 60)
        
        for i, name in enumerate(self.materialNames):
            Y0 = self.rawData[i, self.materialAttributes['Y0']['idx']]
            Y1 = self.rawData[i, self.materialAttributes['Y1']['idx']]
            Y2 = self.rawData[i, self.materialAttributes['Y2']['idx']]
            Y3 = self.rawData[i, self.materialAttributes['Y3']['idx']]
            Temp_Limit = self.rawData[i, self.materialAttributes['Temp_Limit']['idx']]
            
            Y_at_limit = logBezierInterpolation(Temp_Limit, Y0, Y1, Y2, Y3)
            ratio = Y0 / Y_at_limit
            
            print("{:<20} {:>12.1f} {:>12.3e} {:>12.3e} {:>12.2f}".format(name, Temp_Limit, Y0, Y_at_limit, ratio))
        print("=" * 60 + "\n")
    def printModulusDropAtTempLimit(self):
        """Print the ratio of E0 to E at Temp_Limit for each material."""
        if 'E0' not in self.materialAttributes or 'Temp_Limit' not in self.materialAttributes:
            print("ERROR: Missing required attributes (E0 or Temp_Limit)")
            return
        
        print("\n" + "=" * 60)
        print("Young's Modulus Drop at Temperature Limit")
        print("=" * 60)
        print("{:<20} {:>12} {:>12} {:>12} {:>12}".format("Material","T_limit (C)", "E0 (Pa)", "E@T_limit", "E0/E ratio"))
        print("-" * 60)
        
        for i, name in enumerate(self.materialNames):
            E0 = self.rawData[i, self.materialAttributes['E0']['idx']]
            E1 = self.rawData[i, self.materialAttributes['E1']['idx']]
            E2 = self.rawData[i, self.materialAttributes['E2']['idx']]
            E3 = self.rawData[i, self.materialAttributes['E3']['idx']]
            Temp_Limit = self.rawData[i, self.materialAttributes['Temp_Limit']['idx']]
            
            E_at_limit = logBezierInterpolation(Temp_Limit, E0, E1, E2, E3)
            ratio = E0 / E_at_limit
            
            print("{:<20} {:>12.1f} {:>12.3e} {:>12.3e} {:>12.2f}".format(name, Temp_Limit, E0, E_at_limit, ratio))
        print("=" * 60 + "\n")
        
    def printEncodingErrors(self):
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            diffMatrix = self.scaledMaterialData - decoded

        attribute_names = list(self.materialAttributes.keys())
        print("-" * 40)
        print("{:<25} {:>15}".format("Attribute", "Max % Error"))
        print("-" * 40)
        col = 0
        for name in attribute_names:
            percent_err = 100 * diffMatrix[:, col].abs().numpy()
            max_percent_err = np.max(percent_err)
            print("{:<25} {:>15.6f}".format(name, max_percent_err))
            col += 1
        print("-" * 40)

    def loadAutoencoderFromFile(self, fileName):
        self.vaeNet.load_state_dict(torch.load(fileName))
        self.vaeNet.eval()

    def trainAutoencoder(self, numEpochs, klFactor, savedNet, learningRate, maxAttributeErrorPercent=0.0005):
        opt = torch.optim.AdamW(self.vaeNet.parameters(), learningRate)
        convgHistory = {'reconLoss': [], 'klLoss': [], 'loss': []}
        self.vaeNet.encoder.isTraining = True
        for epoch in range(numEpochs):
            opt.zero_grad()
            predData = self.vaeNet(self.scaledMaterialData)
            klLoss = klFactor * self.vaeNet.encoder.kl
            reconLoss = ((self.scaledMaterialData - predData) ** 2).sum()
            loss = reconLoss + klLoss
            loss.backward()
            convgHistory['reconLoss'].append(reconLoss)
            convgHistory['klLoss'].append(klLoss / klFactor)
            convgHistory['loss'].append(loss)
            largestErrorPercent = self.largestEncodingErrorPercent()
            if largestErrorPercent < maxAttributeErrorPercent:
                print('Epoch {:d}: reconLoss {:.3e}, klLoss {:.3e}, loss {:.3e}, maxPercentErr {:.3e}'. \
                      format(epoch, reconLoss.item(), klLoss.item(), loss.item(), largestErrorPercent))
                print("Converged!")
                break
            if (epoch % 5000 == 0):
                print('Epoch {:d}: reconLoss {:.3e}, klLoss {:.3e}, loss {:.3e}, maxPercentErr {:.3e}'. \
                      format(epoch, reconLoss.item(), klLoss.item(), loss.item(), largestErrorPercent))
            opt.step()
        self.vaeNet.encoder.isTraining = False
        torch.save(self.vaeNet.state_dict(), savedNet)
        return convgHistory

    def getMaterialName(self, index):
        if 0 <= index < len(self.materialNames):
            return self.materialNames[index]
        else:
            raise IndexError("Index out of range for material names.")

    def getValuesAtLatentPoints(self, attributeName, zPts):
        decoded = self.vaeNet.decoder(zPts)
        material_properties = self.getMaterialProperties(decoded)
        return material_properties[attributeName].detach().numpy()

    def getMaterialPropertyAtTemperature(self, name, zPts, T):
        decoded = self.vaeNet.decoder(zPts)
        material_properties = self.getMaterialProperties(decoded)

        # Temp_Limit = material_properties['Temp_Limit'].detach().numpy()

        if name in ['E', 'Y']:
            M0 = material_properties[name + '0'].detach().numpy()
            M1 = material_properties[name + '1'].detach().numpy()
            M2 = material_properties[name + '2'].detach().numpy()
            M3 = material_properties[name + '3'].detach().numpy()
            # M = logBezierInterpolation(T, M0, M1, M2, M3, 0, Temp_Limit)
            M = logBezierInterpolation(T, M0, M1, M2, M3)
        elif name == 'K':
            M0 = material_properties[name + '0'].detach().numpy()
            M1 = material_properties[name + '1'].detach().numpy()
            M2 = material_properties[name + '2'].detach().numpy()
            M3 = material_properties[name + '3'].detach().numpy()
            M = bezierInterpolation(T, M0, M1, M2, M3)
        #     # Enforce minimum property value
        # min_db_value = np.min(self.rawData[:, self.materialAttributes[name + '0']['idx']])
        # min_allowed = self.RELATIVE_MATERIAL_MIN_VAL * min_db_value
        # M = np.maximum(M, min_allowed)   
        return M

    def getMaterialPropertyAtTemperatureTorch(self, name, zPts, T):
        decoded = self.vaeNet.decoder(zPts)
        material_properties = self.getMaterialProperties(decoded)

        # Temp_Limit = material_properties['Temp_Limit'].detach().numpy()

        if name in ['E', 'Y']:
            M0 = material_properties[name + '0']
            M1 = material_properties[name + '1']
            M2 = material_properties[name + '2']
            M3 = material_properties[name + '3']
            # M = logBezierInterpolation_torch(T, M0, M1, M2, M3, 0, Temp_Limit)
            M = logBezierInterpolation_torch(T, M0, M1, M2, M3)
        elif name == 'K':
            M0 = material_properties[name + '0']
            M1 = material_properties[name + '1']
            M2 = material_properties[name + '2']
            M3 = material_properties[name + '3']
            M = bezierInterpolation_torch(T, M0, M1, M2, M3)
        # min_db_value = torch.min(torch.tensor(self.rawData[:, self.materialAttributes[name + '0']['idx']], dtype=M.dtype, device=M.device))
        # min_allowed = self.RELATIVE_MATERIAL_MIN_VAL * min_db_value
        # M = torch.maximum(M, min_allowed)
        return M

    def getMaterialProperties(self, decoded):
        properties = {}
        for name, attribute in self.materialAttributes.items():
            idx = attribute['idx']
            scaleMax = attribute['scaleMax']
            scaleMin = attribute['scaleMin']
            if name in ['E0', 'E1', 'E2', 'E3', 'Y0', 'Y1', 'Y2', 'Y3']:
                properties[name] = 10 ** (decoded[:, idx] * (scaleMax - scaleMin) + scaleMin)
            else:
                properties[name] = (decoded[:, idx] * (scaleMax - scaleMin) + scaleMin)
        return properties

    def getClosestRealMaterialIndex(self, zDesign):
        with torch.no_grad():
            zReal = self.vaeNet.encoder(self.scaledMaterialData)

        distances = torch.cdist(zDesign, zReal)
        closest_indices = torch.argmin(distances, dim=1)
        return closest_indices

    def getClosestRealMaterialZValues(self, zDesign):
        with torch.no_grad():
            zReal = self.vaeNet.encoder(self.scaledMaterialData)

        distances = torch.cdist(zDesign, zReal)
        closest_indices = torch.argmin(distances, dim=1)
        return zReal[closest_indices].detach().numpy()

    def materialDistance(self, zDesign, xDesign, gamma):
        zDesign_tensor = torch.tensor(zDesign, dtype=torch.float32, requires_grad=True)
        decoded_design = self.vaeNet.decoder(zDesign_tensor)

        props_real = self.rawData
        props_design = self.getMaterialProperties(decoded_design)
        props_design = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in props_design.values()], dim=1)
        props_real = torch.tensor(props_real) if not isinstance(props_real, torch.Tensor) else props_real

        design_exp = props_design.unsqueeze(1)
        real_exp = props_real.unsqueeze(0)

        norm_diff = ((design_exp - real_exp) ** 2) / (real_exp ** 2 + 1e-12)

        net_distance = norm_diff.sum(dim=2)
        min_distances, _ = torch.min(net_distance, dim=1)

        penalty = gamma * torch.mean(min_distances * xDesign)
        penalty.backward()
        grad = zDesign_tensor.grad.detach().numpy()
        grad = grad.T.reshape(-1)
        return penalty.detach().numpy(), grad

    def materialAttributeDistance(self, attributeName, zDesign, xDesign, gamma):
        attributeId = list(self.materialAttributes.keys()).index(attributeName)
        zDesign_tensor = torch.tensor(zDesign, dtype=torch.float32, requires_grad=True)
        decoded_design = self.vaeNet.decoder(zDesign_tensor)

        props_real = self.rawData
        props_design = self.getMaterialProperties(decoded_design)
        props_design = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in props_design.values()], dim=1)
        props_real = torch.tensor(props_real) if not isinstance(props_real, torch.Tensor) else props_real

        design_exp = props_design[:, attributeId].unsqueeze(1)
        real_exp = props_real[:, attributeId].unsqueeze(0)

        norm_diff = ((design_exp - real_exp) ** 2) / (real_exp ** 2 + 1e-12)

        net_distance = norm_diff
        min_distances, _ = torch.min(net_distance, dim=1)

        penalty = gamma * torch.mean(min_distances * xDesign)
        penalty.backward()
        grad = zDesign_tensor.grad.detach().numpy()
        grad = grad.T.reshape(-1)
        return penalty.detach().numpy(), grad

    def plotLSR(self, zRealPts, zDesignPts=None, xDesign=None):
        zRealPts = np.asarray(zRealPts)
        if zDesignPts is not None:
            zDesignPts = np.asarray(zDesignPts)
        latentDim = zRealPts.shape[1]

        # PCA projection only for latentDim > 3
        if latentDim > 3:
            pca = PCA(n_components=2)
            zRealPts_2d = pca.fit_transform(zRealPts)
            zDesignPts_2d = pca.transform(zDesignPts) if zDesignPts is not None else None

            plt.figure()
            plt.title(f"Latent Space (PCA projection from {latentDim}D)")
            if zDesignPts_2d is not None and xDesign is not None:
                mask = np.asarray(xDesign) > 0.5
                if np.any(mask):
                    plt.scatter(zDesignPts_2d[mask, 0], zDesignPts_2d[mask, 1], c='red', marker='o', s=20, label='Optimized Materials', alpha=0.2)
            plt.scatter(zRealPts_2d[:, 0], zRealPts_2d[:, 1], c='black', marker='*', s=200, label='Real Materials', alpha=0.4)
            for i, label in enumerate(self.materialNames):
                plt.text(zRealPts_2d[i, 0] + 0.1, zRealPts_2d[i, 1], str(label), fontsize=12, color='black', ha='center', va='bottom')
            plt.xlabel('$z_1$')
            plt.ylabel('$z_2$')
            plt.xlim(-5, 5)
            plt.ylim(-5, 5)
            plt.grid(True)
            plt.legend()
            plt.show()

        # 3D plot for latentDim == 3
        elif latentDim == 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if zDesignPts is not None and xDesign is not None:
                mask = np.asarray(xDesign) > 0.5
                if np.any(mask):
                    ax.scatter(zDesignPts[mask, 0], zDesignPts[mask, 1], zDesignPts[mask, 2],
                               c='red', marker='o', s=20, label='Optimized Materials', alpha=0.6)
            ax.scatter(zRealPts[:, 0], zRealPts[:, 1], zRealPts[:, 2],
                       c='black', marker='*', s=200, label='Real Materials', alpha=0.6)
            for i, label in enumerate(self.materialNames):
                ax.text(zRealPts[i, 0] + 0.1, zRealPts[i, 1], zRealPts[i, 2], str(label),
                        fontsize=10, color='black')
            ax.set_xlabel('$z_1$')
            ax.set_ylabel('$z_2$')
            ax.set_zlabel('$z_3$')
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_zlim(-5, 5)
            plt.legend()
            plt.show()

        # 2D plot for latentDim <= 2
        else:
            zRealPts_2d = zRealPts
            zDesignPts_2d = zDesignPts
            plt.figure()
            if zDesignPts_2d is not None and xDesign is not None:
                mask = np.asarray(xDesign) > 0.5
                if np.any(mask):
                    plt.scatter(zDesignPts_2d[mask, 0], zDesignPts_2d[mask, 1], c='red', marker='o', s=20, label='Optimized Materials', alpha=0.2)
            plt.scatter(zRealPts_2d[:, 0], zRealPts_2d[:, 1], c='black', marker='*', s=200, label='Real Materials', alpha=0.4)
            for i, label in enumerate(self.materialNames):
                plt.text(zRealPts_2d[i, 0] + 0.1, zRealPts_2d[i, 1], str(label), fontsize=12, color='black', ha='center', va='bottom')
            plt.xlabel('$z_1$')
            plt.ylabel('$z_2$')
            # plt.xlim(-5, 5)
            # plt.ylim(-5, 5)
            plt.grid(True)
            plt.legend()
            plt.show()

    def plotLSRContours(self, attributeName, title=""):
        attributeId = list(self.materialAttributes.keys()).index(attributeName)
        zReal = self.training_latents
        latentDim = zReal.shape[1]
        n_points = 50


        if latentDim > 2:
            # Use PCA to project latent space to 2D grid for contour plotting
            pca = PCA(n_components=2)
            zReal_2d = pca.fit_transform(zReal)
            # Create grid in PCA space
            z1 = np.linspace(-5, 5, n_points)
            z2 = np.linspace(-5, 5, n_points)
            Z1, Z2 = np.meshgrid(z1, z2)
            Z_grid_2d = np.stack([Z1.ravel(), Z2.ravel()], axis=1)
            # Inverse transform grid to latent space
            Z_grid_full = pca.inverse_transform(Z_grid_2d)
        else:
            zReal_2d = zReal
            z1 = np.linspace(-5, 5, n_points)
            z2 = np.linspace(-5, 5, n_points)
            Z1, Z2 = np.meshgrid(z1, z2)
            Z_grid_full = np.stack([Z1.ravel(), Z2.ravel()], axis=1)

        QOI = []
        with torch.no_grad():
            for z in Z_grid_full:
                z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
                decoded = self.vaeNet.decoder(z_tensor)
                decodedValues = self.getMaterialProperties(decoded)
                QOI.append(decodedValues[list(decodedValues.keys())[attributeId]].item())
        QOI = np.array(QOI).reshape(Z1.shape)
        plt.figure(figsize=(7.5, 6))
        contour = plt.contourf(Z1, Z2, QOI, levels=30, cmap='viridis')
        units = self.materialAttributes[list(self.materialAttributes.keys())[attributeId]]['unit']
        plt.colorbar(contour, label=list(self.materialAttributes.keys())[attributeId] + " (" + units + ")")
        plt.scatter(zReal_2d[:, 0], zReal_2d[:, 1], c='black', marker='*', s=200, alpha=1.0)
        for i, label in enumerate(self.materialNames):
            plt.text(zReal_2d[i, 0] + 0.1, zReal_2d[i, 1], str(label), fontsize=12, color='black', ha='center', va='bottom')
        plt.xlabel('$z_1$')
        plt.ylabel('$z_2$')
        plt.title(title)
        plt.show()

    def plotTemperatureVsMaterialProperty(self, attrName, semilogy=False):
        zRealPts = self.vaeNet.encoder.z
        plt.figure()
        T = np.linspace(TMin, TMax, 300)
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
        for i in range(zRealPts.shape[0]):
            zPt = zRealPts[i, :].view(1, -1)
            M = self.getMaterialPropertyAtTemperature(attrName, zPt, T)
            marker = markers[i % len(markers)]
            if semilogy:
                plt.semilogy(T, M, label=self.materialNames[i], marker=marker, markevery=30)
            else:
                plt.plot(T, M, label=self.materialNames[i], marker=marker, markevery=30)

        plt.xlabel("Temperature (C)")
        plt.ylabel(f"{attrName}")
        plt.title(f"Temperature vs {attrName}")
        plt.legend(self.materialNames)
        plt.grid()
        plt.show()
        
    def plotTemperatureVsMaterialPropertyRaw(self, attrName, semilogy=False):
        plt.figure()
        T = np.linspace(TMin, TMax, 300)
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
        
        # Define colors for the first three materials and generate others
        colors = ['#0201fc', '#3cb44b', '#654321']
        if self.rawData.shape[0] > 3:
            
            additional_colors = cm.tab10(np.linspace(0.3, 1, self.rawData.shape[0] - 3))
            colors.extend([cm.colors.rgb2hex(c) for c in additional_colors])
        for i in range(self.rawData.shape[0]):
            # Get control points for this material
            if attrName in ['E', 'Y']:
                M0 = self.rawData[i, self.materialAttributes[attrName + '0']['idx']]
                M1 = self.rawData[i, self.materialAttributes[attrName + '1']['idx']]
                M2 = self.rawData[i, self.materialAttributes[attrName + '2']['idx']]
                M3 = self.rawData[i, self.materialAttributes[attrName + '3']['idx']]
                # Temp_Limit = self.rawData[i, self.materialAttributes['Temp_Limit']['idx']]
                M = logBezierInterpolation(T, M0, M1, M2, M3)
            elif attrName == 'K':
                M0 = self.rawData[i, self.materialAttributes[attrName + '0']['idx']]
                M1 = self.rawData[i, self.materialAttributes[attrName + '1']['idx']]
                M2 = self.rawData[i, self.materialAttributes[attrName + '2']['idx']]
                M3 = self.rawData[i, self.materialAttributes[attrName + '3']['idx']]
                M = bezierInterpolation(T, M0, M1, M2, M3)
            else:
                continue
            # min_db_value = np.min(self.rawData[:, self.materialAttributes[attrName + '0']['idx']])
            # min_allowed = self.RELATIVE_MATERIAL_MIN_VAL * min_db_value
            # M = np.maximum(M, min_allowed)  
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            if semilogy:
                plt.semilogy(T, M, label=self.materialNames[i], marker=marker, markevery=30, color=color)
            else:
                plt.plot(T, M, label=self.materialNames[i], marker=marker, markevery=30, color=color)
        plt.xlabel("Temperature (C)")
        plt.ylabel(f"{attrName}")
        plt.title(f"Temperature vs {attrName} (Raw Data)")
        plt.legend(self.materialNames)
        plt.grid()
        plt.show()
        
    def getHeaviestMaterial(self):
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            decoded_properties = self.getMaterialProperties(decoded)

        density_values = decoded_properties['Density'].detach().cpu().numpy().flatten()
        heaviest_idx = np.argmax(density_values)
        heaviest_z = z_real[heaviest_idx].detach().cpu().numpy()
        return heaviest_z

    def getLightestMaterial(self):
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            decoded_properties = self.getMaterialProperties(decoded)

        density_values = decoded_properties['Density'].detach().cpu().numpy().flatten()
        lightest_idx = np.argmin(density_values)
        lightest_z = z_real[lightest_idx].detach().cpu().numpy()
        return lightest_z
    

    def getStrongestMaterial(self):
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            decoded_properties = self.getMaterialProperties(decoded)

        yield_values = decoded_properties['Y0'].detach().cpu().numpy().flatten()
        strongest_idx = np.argmax(yield_values)
        strongest_z = z_real[strongest_idx].detach().cpu().numpy()
        return strongest_z
    
    def getBestStiffnessToDensityMaterial(self):
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            decoded_properties = self.getMaterialProperties(decoded)

        stiffness_values = decoded_properties['E0'].detach().cpu().numpy().flatten()
        density_values = decoded_properties['Density'].detach().cpu().numpy().flatten()
        best_idx = np.argmax(stiffness_values/density_values)
        best_z = z_real[best_idx].detach().cpu().numpy()
        return best_z
    
    def getBestStrengthToDensityMaterial(self):
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            decoded_properties = self.getMaterialProperties(decoded)

        yield_values = decoded_properties['Y0'].detach().cpu().numpy().flatten()
        density_values = decoded_properties['Density'].detach().cpu().numpy().flatten()
        best_idx = np.argmax(yield_values/density_values)
        best_z = z_real[best_idx].detach().cpu().numpy()
        return best_z