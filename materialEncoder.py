from networks import VariationalAutoencoder
import torch
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon, Ellipse
import numpy as np
def unlognorm(x, scaleMax, scaleMin, minAdded):
    # Reverse the normalization and log transform
    return 10**(x * (scaleMax - scaleMin) + scaleMin) + minAdded - 10


class MaterialEncoder:

  def __init__(self, scaledMaterialData, materialAttributes, materialNames, vaeSettings):
    self.nMaterials = scaledMaterialData.shape[0]
    self.nAttributes = scaledMaterialData.shape[1]
    self.scaledMaterialData = scaledMaterialData
    self.materialAttributes =  materialAttributes
    self.materialNames = materialNames
    self.vaeSettings = vaeSettings
    self.vaeNet = VariationalAutoencoder(vaeSettings)

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
  
  def plotLSR(self, zReal, zDesign = None):
   
    if zDesign is not None:
        plt.scatter(zDesign[:, 0], zDesign[:, 1], c='red', marker='o', s=20, label='Optimized Materials', alpha=0.2)
    plt.scatter(zReal[:, 0], zReal[:, 1], c='black', marker='*', s=200, label='Real Materials', alpha=1.0)
    for i, label in enumerate(self.materialNames['name']):
        plt.text(zReal[i, 0] + 0.1, zReal[i, 1], str(label), fontsize=12, color='black', ha='center', va='bottom')
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    plt.legend(fontsize=14)
    plt.show()

  def plotLSRContours(self, attributeId = 0, title=""):
    zReal = self.vaeNet.encoder.z.detach().numpy()
    n_points = 100
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
    plt.legend(fontsize=14)
    plt.show()

  def getMaterialProperties(self, decoded):
    """
    Returns a dictionary of all denormalized and unlogged material properties.
    Keys are attribute names from materialAttributes.
    """
    properties = {}
    for name, info in self.materialAttributes.items():
        idx = info['idx']
        scaleMax = info['scaleMax']
        scaleMin = info['scaleMin']
        minAdded = info['minAdded']
        properties[name] = unlognorm(decoded[:, idx], scaleMax, scaleMin, minAdded)
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
          true_properties[name] = unlognorm(true_values[:, idx], scaleMax, scaleMin, minAdded)

      
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
    


