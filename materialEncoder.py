from networks import VariationalAutoencoder
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from InterpolationFunctions import  bezierInterpolation, bezierInterpolation_torch
from InterpolationFunctions import  logBezierInterpolation, logBezierInterpolation_torch

class MaterialEncoder:
  def __init__(self,vae_params):
    self.nAttributes = 0
    self.vae_params = vae_params
    
  def readExcel(self, excel_file):
    self.excel_file = excel_file
    self.preprocessData()  
    
    self.nMaterials = self.scaledMaterialData.shape[0]
    self.nAttributes = self.scaledMaterialData.shape[1]

    self.vaeSettings  = {
        'encoder': {'inputDim': self.nAttributes, 'hiddenDim': self.vae_params.vae_hiddenDim, 'latentDim': self.vae_params.latentDim},
        'decoder': {'latentDim': self.vae_params.latentDim, 'hiddenDim': self.vae_params.vae_hiddenDim, 'outputDim': self.nAttributes}
    }
    self.vaeNet = VariationalAutoencoder(self.vaeSettings)

  def preprocessData(self):
      df = pd.read_excel(self.excel_file, header=None)
      # First column: material names, but first cell is "Attribute", second is "Material/Units"
      # Second column onwards: attribute names (first cell), units (second cell), values (third row onwards)

      # Extract attribute names and units from second column onwards
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

      # Build materialAttributes dictionary
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

      # Compute mean and standard deviation for each attribute
      attribute_means = normalizedData.mean(axis=0)
      attribute_stds = normalizedData.std(axis=0)
      self.attribute_means = attribute_means
      self.attribute_stds = attribute_stds

      # Calculate sigma/mean for each attribute
      sigma_mean_ratios = {}
      for i, name in enumerate(attribute_names):
          sigma = attribute_stds[i]
          mean = attribute_means[i]
          sigma_mean_ratios[name] = sigma / mean if mean != 0 else float('inf')

      # Print the name and sigma/mean for each attribute
      print("Attribute Name       Sigma/Mean for Normalized Data")
      print("-" * 40)
      for name, ratio in sigma_mean_ratios.items():
          print(f"{name:<20} {ratio:.6f}")

      
  def printEncodingErrors(self):
      """
      Prints a table of maximum percent error for each decoded real material attribute
      compared to the actual Excel sheet values.
      """
      # Get real latent points for all materials
      with torch.no_grad():
          z_real = self.vaeNet.encoder(self.scaledMaterialData)
          decoded = self.vaeNet.decoder(z_real)
          decoded_properties = self.getMaterialProperties(decoded)

      print("-" * 40)
      print(f"Number of materials: {self.nMaterials}")
      print(f"Number of attributes: {self.nAttributes}")
      attribute_names = list(self.materialAttributes.keys())  
      print("-" * 40)
      print("{:<25} {:>15}".format("Attribute", "Max % Error"))
      print("-" * 40)
      col = 0
      for name in attribute_names:
          decoded_vals = decoded_properties[name]
          if hasattr(decoded_vals, "detach"):
              decoded_vals = decoded_vals.detach().cpu().numpy().flatten()
          else:
              decoded_vals = np.array(decoded_vals).flatten()
          true_vals = self.rawData[:, col]
          percent_err = 100 * np.abs(decoded_vals - true_vals) / (np.abs(true_vals) + 1e-12)
          max_percent_err = np.max(percent_err)
          print("{:<25} {:>15.6f}".format(name, max_percent_err))
          col += 1
      print("-" * 40)

  def loadAutoencoderFromFile(self, fileName):
    self.vaeNet.load_state_dict(torch.load(fileName))
    self.vaeNet.eval()
    
  def trainAutoencoder(self, numEpochs, klFactor, savedNet, learningRate):
    opt = torch.optim.Adam(self.vaeNet.parameters(), learningRate)
    convgHistory = {'reconLoss':[], 'klLoss':[], 'loss':[]}
    self.vaeNet.encoder.isTraining = True
    for epoch in range(numEpochs):
      opt.zero_grad()
      predData = self.vaeNet(self.scaledMaterialData)
      klLoss = klFactor*self.vaeNet.encoder.kl
      reconLoss =  ((self.scaledMaterialData - predData)**2).sum()
      loss = reconLoss + klLoss 
      loss.backward()
      convgHistory['reconLoss'].append(reconLoss)
      convgHistory['klLoss'].append(klLoss/klFactor) # save unscaled loss
      convgHistory['loss'].append(loss)
      opt.step()
      if(epoch%500 == 0):
        print('Iter {:d} reconLoss {:.3e} klLoss {:.3e} loss {:.3e}'.\
              format(epoch, reconLoss.item(), klLoss.item(), loss.item()))
     
    self.vaeNet.encoder.isTraining = False
    torch.save(self.vaeNet.state_dict(), savedNet)
    return convgHistory

  def getMaterialName(self, index):
      """
      Returns the material name for a given index.
      """
      if 0 <= index < len(self.materialNames):
          return self.materialNames[index]
      else:
          raise IndexError("Index out of range for material names.")
      
  def getValuesAtLatentPoints(self, attributeName, zPts):
      """
      Returns the attribute values for a given set of latent points.
      """
      decoded = self.vaeNet.decoder(zPts)
      material_properties = self.getMaterialProperties(decoded)
      return material_properties[attributeName].detach().numpy()  
  

  def getMaterialPropertyAtTemperature(self, name,  zPts, T):
      decoded = self.vaeNet.decoder(zPts)
      material_properties = self.getMaterialProperties(decoded)

      if name in ['E', 'Y']:
          M0 = material_properties[name + '0'].detach().numpy()
          M1 = material_properties[name + '1'].detach().numpy()
          M2 = material_properties[name + '2'].detach().numpy()
          M3 = material_properties[name + '3'].detach().numpy()
          M = logBezierInterpolation(T, M0, M1, M2, M3)
      elif name == 'K':
          M0 = material_properties[name + '0'].detach().numpy()
          M1 = material_properties[name + '1'].detach().numpy()
          M2 = material_properties[name + '2'].detach().numpy()
          M3 = material_properties[name + '3'].detach().numpy()
          M = bezierInterpolation(T, M0, M1, M2, M3)
      return M

  def getMaterialPropertyAtTemperatureTorch(self, name,  zPts, T):
      decoded = self.vaeNet.decoder(zPts)
      material_properties = self.getMaterialProperties(decoded)

      if name in ['E', 'Y']:
          M0 = material_properties[name + '0']
          M1 = material_properties[name + '1']
          M2 = material_properties[name + '2']
          M3 = material_properties[name + '3']
          M = logBezierInterpolation_torch(T, M0, M1, M2, M3)
      elif name == 'K':
          M0 = material_properties[name + '0']
          M1 = material_properties[name + '1']
          M2 = material_properties[name + '2']
          M3 = material_properties[name + '3']
          M = bezierInterpolation_torch(T, M0, M1, M2, M3)
      return M

  
  def getMaterialProperties(self, decoded):
    """
    Returns a dictionary of all denormalized and unlogged material properties.
    Keys are attribute names from materialAttributes.
    """
    properties = {}
    for name, attribute in self.materialAttributes.items():
        idx = attribute['idx']
        scaleMax = attribute['scaleMax']
        scaleMin = attribute['scaleMin']
        if name in ['E0', 'E1', 'E2', 'E3', 'Y0', 'Y1', 'Y2', 'Y3']:
            properties[name] = 10**(decoded[:, idx] * (scaleMax - scaleMin) + scaleMin)
        else:
            properties[name] = (decoded[:, idx] * (scaleMax - scaleMin) + scaleMin)
    return properties

  
  def getClosestRealMaterialIndex(self, zDesign):
    # Get the index of the closest real material in latent space to the given design latent vector
    with torch.no_grad():
      zReal = self.vaeNet.encoder(self.scaledMaterialData)  

    distances = torch.cdist(zDesign, zReal)
    # For each design, find the closest real material index
    closest_indices = torch.argmin(distances, dim=1)
    return closest_indices
 
 
  def getClosestRealMaterialZValues(self, zDesign):
    # Get the index of the closest real material in latent space to the given design latent vector
    with torch.no_grad():
      zReal = self.vaeNet.encoder(self.scaledMaterialData)  

    distances = torch.cdist(zDesign, zReal)
    # For each design, find the closest real material index
    closest_indices = torch.argmin(distances, dim=1)
    return zReal[closest_indices].detach().numpy()

  def materialDistance(self, zDesign,xDesign,gamma):
    # Decode latent vectors to material properties
    zDesign_tensor = torch.tensor(zDesign, dtype=torch.float32, requires_grad=True)
    decoded_design = self.vaeNet.decoder(zDesign_tensor)

    props_real = self.rawData
    props_design = self.getMaterialProperties(decoded_design)
    # Convert dictionary of tensors to a single tensor array for props_design
    props_design = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in props_design.values()], dim=1)
    props_real = torch.tensor(props_real) if not isinstance(props_real, torch.Tensor) else props_real
    
    # Compute normalized attribute-wise squared differences
    # props_design: (nDesigns, nAttributes), props_real: (nReal, nAttributes)
    # Expand dims for broadcasting
    design_exp = props_design.unsqueeze(1)  # (nDesigns, 1, nAttributes)
    real_exp = props_real.unsqueeze(0)      # (1, nReal, nAttributes)

    # Normalized squared difference: ((design - real)^2) / (real^2 + 1e-12)
    norm_diff = ((design_exp - real_exp) ** 2) / (real_exp ** 2 + 1e-12)  # (nDesigns, nReal, nAttributes)

    # Sum over attributes to get net distance
    net_distance = norm_diff.sum(dim=2)  # (nDesigns, nReal)
    # Find the minimum distance for each design (over all real materials)
    # Use standard min instead of p-norm to aggregate distances across real materials
    # net_distance: (nDesigns, nReal)
    min_distances, _ = torch.min(net_distance, dim=1)

    penalty = gamma * torch.mean(min_distances * xDesign)
    # Compute gradient of penalty w.r.t. zDesign
    penalty.backward()
    grad = zDesign_tensor.grad.detach().numpy()
    grad = grad.T.reshape(-1)
    return penalty.detach().numpy(), grad

  def materialAttributeDistance(self, attributeName, zDesign, xDesign, gamma):
    attributeId = list(self.materialAttributes.keys()).index(attributeName)
    # Decode latent vectors to material properties
    zDesign_tensor = torch.tensor(zDesign, dtype=torch.float32, requires_grad=True)
    decoded_design = self.vaeNet.decoder(zDesign_tensor)

    props_real = self.rawData
    props_design = self.getMaterialProperties(decoded_design)
    # Convert dictionary of tensors to a single tensor array for props_design
    props_design = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in props_design.values()], dim=1)
    props_real = torch.tensor(props_real) if not isinstance(props_real, torch.Tensor) else props_real
    
    # Compute normalized attribute-wise squared differences
    # props_design: (nDesigns, nAttributes), props_real: (nReal, nAttributes)
    # Expand dims for broadcasting
    design_exp = props_design[:, attributeId].unsqueeze(1)  # (nDesigns, 1)
    real_exp = props_real[:, attributeId].unsqueeze(0)      # (1, nReal)

    # Normalized squared difference: ((design - real)^2) / (real^2 + 1e-12)
    norm_diff = ((design_exp - real_exp) ** 2) / (real_exp ** 2 + 1e-12)  # (nDesigns, nReal)

    # Sum over attributes to get net distance
    net_distance = norm_diff  # (nDesigns, nReal)
    # Find the minimum distance for each design (over all real materials)
    # Use standard min instead of p-norm to aggregate distances across real materials
    # net_distance: (nDesigns, nReal)
    min_distances, _ = torch.min(net_distance, dim=1)

    penalty = gamma * torch.mean(min_distances * xDesign)
    # Compute gradient of penalty w.r.t. zDesign
    penalty.backward()
    grad = zDesign_tensor.grad.detach().numpy()
    grad = grad.T.reshape(-1)
    return penalty.detach().numpy(), grad
  
  def plotLSR(self, zRealPts, zDesignPts = None,xDesign=None):

    if zDesignPts is not None and xDesign is not None:
      mask = xDesign > 0.5
      if np.any(mask):
        plt.scatter(zDesignPts[mask, 0], zDesignPts[mask, 1], c='red', marker='o', s=20, label='Optimized Materials', alpha=0.2)
    plt.scatter(zRealPts[:, 0], zRealPts[:, 1], c='black', marker='*', s=200, label='Real Materials', alpha=0.4)
    for i, label in enumerate(self.materialNames):
        plt.text(zRealPts[i, 0] + 0.1, zRealPts[i, 1], str(label), fontsize=12, color='black', ha='center', va='bottom')
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    plt.legend(fontsize=10)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(True)
    plt.show()

  def plotLSRContours(self, attributeName, title=""):
    attributeId = list(self.materialAttributes.keys()).index(attributeName)
    zReal = self.training_latents
    n_points = 50
    z1 = np.linspace(-5, 5, n_points)
    z2 = np.linspace(-5, 5, n_points)
    Z1, Z2 = np.meshgrid(z1, z2)
    Z_grid = np.stack([Z1.ravel(), Z2.ravel()], axis=1)
    QOI = []
    with torch.no_grad():
        for z in Z_grid:
            z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
            decoded = self.vaeNet.decoder(z_tensor)
            decodedValues = self.getMaterialProperties(decoded)
            QOI.append(decodedValues[list(decodedValues.keys())[attributeId]].item())
    QOI = np.array(QOI).reshape(Z1.shape)
    plt.figure(figsize=(7.5, 6))
    contour = plt.contourf(Z1, Z2, QOI, levels=30, cmap='viridis')
    units = self.materialAttributes[list(self.materialAttributes.keys())[attributeId]]['unit']
    plt.colorbar(contour, label=list(self.materialAttributes.keys())[attributeId] + " (" + units + ")")
    plt.scatter(zReal[:, 0], zReal[:, 1], c='black', marker='*', s=200,  alpha=1.0)
    for i, label in enumerate(self.materialNames):
        plt.text(zReal[i, 0] + 0.1, zReal[i, 1], str(label), fontsize=12, color='black', ha='center', va='bottom')
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    plt.title(title)
    plt.show()


  def plotTemperatureVsMaterialProperty(self,attrName,semilogy=False):
    """Plot temperature vs material property for the given mesh."""
    # Extract material properties from the mesh
    
    zRealPts = self.vaeNet.encoder.z
    plt.figure()
    T = np.linspace(0, 1250, 300)
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
    for i in range(zRealPts.shape[0]):
      zPt = zRealPts[i, :].view(1, 2)
      M = self.getMaterialPropertyAtTemperature(attrName, zPt, T)
      marker = markers[i % len(markers)]  # Cycle through markers
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


  def getHeaviestMaterial(self):
    # Get real latent points for all materials
    with torch.no_grad():
        z_real = self.vaeNet.encoder(self.scaledMaterialData)
        decoded = self.vaeNet.decoder(z_real)
        decoded_properties = self.getMaterialProperties(decoded)

    # Find the index of the heaviest material
    density_values = decoded_properties['Density'].detach().cpu().numpy().flatten()
    heaviest_idx = np.argmax(density_values)
    heaviest_z = z_real[heaviest_idx].detach().cpu().numpy()
    return heaviest_z
  
  def getLightestMaterial(self):
    # Get real latent points for all materials
    with torch.no_grad():
      z_real = self.vaeNet.encoder(self.scaledMaterialData)
      decoded = self.vaeNet.decoder(z_real)
      decoded_properties = self.getMaterialProperties(decoded)

    # Find the index of the lightest material
    density_values = decoded_properties['Density'].detach().cpu().numpy().flatten()
    lightest_idx = np.argmin(density_values)
    lightest_z = z_real[lightest_idx].detach().cpu().numpy()
    return lightest_z


