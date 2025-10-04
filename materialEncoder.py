from networks import VariationalAutoencoder
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



class MaterialEncoder:
  def __init__(self,vae_params):
    self.nAttributes = 0
    self.vae_params = vae_params
    self.offset = 10 # offset to avoid log(0)

  def readExcel(self, excel_file):
    self.excel_file = excel_file
    self.scaledMaterialData, self.materialAttributes, self.materialNames, self.trainInfo = self.preprocessData()  
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
      # Extract material names from first column, starting from third row
      material_names = df.iloc[2:, 0].tolist()
      # Extract attribute values from second column onwards, starting from third row
      values = df.iloc[2:, 1:].to_numpy(dtype=float)

  
      
      min_vals = np.min(values, axis=0)
      log_values = np.log10(values - min_vals + self.offset)
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
  
  def unlognorm(self, x, scaleMax, scaleMin, minAdded):
    # Reverse the normalization and log transform
    return 10**(x * (scaleMax - scaleMin) + scaleMin) + minAdded - self.offset
  

  def materialDistance(self, zReal, zDesign):
    # Decode latent vectors to material properties
    zDesign_tensor = torch.tensor(zDesign, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
      decoded_real = self.vaeNet.decoder(torch.tensor(zReal, dtype=torch.float32))
    decoded_design = self.vaeNet.decoder(zDesign_tensor)
    props_real = self.getMaterialProperties(decoded_real)
    # Convert dictionary of tensors to a single array for props_real
    props_real = np.stack([v for v in props_real.values()], axis=1)
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
    penalty = min_distances.sum()
    penalty.backward()
    grad = zDesign_tensor.grad.detach().numpy()
   
    return penalty.detach().numpy(), grad

  def plotLSR(self, zReal, zDesign = None,xDesign=None):
   
    if zDesign is not None and xDesign is not None:
      mask = xDesign > 0.5
      if np.any(mask):
        plt.scatter(zDesign[mask, 0], zDesign[mask, 1], c='red', marker='o', s=20, label='Optimized Materials', alpha=0.2)
    plt.scatter(zReal[:, 0], zReal[:, 1], c='black', marker='*', s=200, label='Real Materials', alpha=1.0)
    for i, label in enumerate(self.materialNames['name']):
        plt.text(zReal[i, 0] + 0.1, zReal[i, 1], str(label), fontsize=12, color='black', ha='center', va='bottom')
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    plt.legend(fontsize=14)
    plt.show()

  def plotLSRContours(self, attributeId = 0, title=""):
    zReal = self.vaeNet.encoder.z.detach().numpy()
    n_points = 25
    z1 = np.linspace(-3, 3, n_points)
    z2 = np.linspace(-3, 3, n_points)
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
    plt.figure(figsize=(9.5, 8))
    contour = plt.contourf(Z1, Z2, QOI, levels=30, cmap='viridis')
    units = self.materialAttributes[list(self.materialAttributes.keys())[attributeId]]['unit']
    plt.colorbar(contour, label=list(self.materialAttributes.keys())[attributeId] + " (" + units + ")")
    plt.scatter(zReal[:, 0], zReal[:, 1], c='black', marker='*', s=200,  alpha=1.0)
    for i, label in enumerate(self.materialNames['name']):
        plt.text(zReal[i, 0] + 0.1, zReal[i, 1], str(label), fontsize=12, color='black', ha='center', va='bottom')
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    plt.title(title)
    plt.show()

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
        minAdded = attribute['minAdded']
        properties[name] = self.unlognorm(decoded[:, idx], scaleMax, scaleMin, minAdded)
    return properties
    
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

      # Get true values from Excel (unlogged, unnormalized)
      true_values = self.scaledMaterialData
      attribute_names = list(self.materialAttributes.keys())
      true_properties = {}
      for name in attribute_names:
          info = self.materialAttributes[name]
          idx = info['idx']
          scaleMax = info['scaleMax']
          scaleMin = info['scaleMin']
          minAdded = info['minAdded']
          # Reverse normalization and log transform
          true_properties[name] = self.unlognorm(true_values[:, idx], scaleMax, scaleMin, minAdded)

      print("-" * 40)
      print("{:<25} {:>15}".format("Attribute", "Max % Error"))
      print("-" * 40)
      for name in attribute_names:
          decoded_vals = decoded_properties[name]
          if hasattr(decoded_vals, "detach"):
              decoded_vals = decoded_vals.detach().cpu().numpy().flatten()
          else:
              decoded_vals = np.array(decoded_vals).flatten()
          true_vals = true_properties[name]
          if isinstance(true_vals, torch.Tensor):
              true_vals = true_vals.detach().cpu().numpy().flatten()
          else:
              true_vals = np.array(true_vals).flatten()
          percent_err = 100 * np.abs(decoded_vals - true_vals) / (np.abs(true_vals) + 1e-12)
          max_percent_err = np.max(percent_err)
          print("{:<25} {:>15.6f}".format(name, max_percent_err))
      print("-" * 40)
