from networks import VariationalAutoencoder
import torch
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon, Ellipse
import numpy as np

class MaterialEncoder:

  def __init__(self, trainingData, dataInfo, dataIdentifier, vaeSettings,constraints=None):
    self.trainingData, self.dataInfo = trainingData, dataInfo
    self.dataIdentifier = dataIdentifier
    self.vaeSettings = vaeSettings
    self.vaeNet = VariationalAutoencoder(vaeSettings)
    if constraints is not None:
      self.constraints = constraints
    else:
      self.constraints = {'distance': {'isOn': False, 'center': np.array([ 0.39397555, -0.5732285 ]), 'a': 2.1778150329939, 'b': 1.4285939288782639, 'theta': 2.9699808951264286, 'delta': 0.0, 'beta': 20}}
  
  def loadAutoencoderFromFile(self, fileName):
    self.vaeNet.load_state_dict(torch.load(fileName))
    self.vaeNet.eval()
    
  def trainAutoencoder(self, numEpochs, klFactor, savedNet, learningRate):
    opt = torch.optim.Adam(self.vaeNet.parameters(), learningRate)
    convgHistory = {'reconLoss':[], 'klLoss':[], 'loss':[]}
    self.vaeNet.encoder.isTraining = True
    for epoch in range(numEpochs):
      opt.zero_grad()
      predData = self.vaeNet(self.trainingData)
      klLoss = klFactor*self.vaeNet.encoder.kl
      reconLoss =  ((self.trainingData - predData)**2).sum()
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
  
  def plotLatent(self, ltnt1, ltnt2, plotHull, annotateHead, saveFileName):
    clrs = ['purple', 'green', 'orange', 'pink', 'yellow', 'black', 'violet', 'cyan', 'red', 'blue']
    colorcol = self.dataIdentifier['classID']
    ptLabel = self.dataIdentifier['name']
    autoencoder = self.vaeNet
    z = autoencoder.encoder.z.to('cpu').detach().numpy()
    fig, ax = plt.subplots()

    for i in range(np.max(colorcol)+1): 
      zMat = np.vstack((z[colorcol == i,ltnt1], z[colorcol == i,ltnt2])).T
      ax.scatter(zMat[:, 0], zMat[:, 1], c = 'black', s = 4)#clrs[i]
      if(i == np.max(colorcol)): #removed for last class TEST
        break # END TEST
      if(plotHull):
        hull = ConvexHull(zMat)
        cent = np.mean(zMat, 0)
        pts = []
        for pt in zMat[hull.simplices]:
            pts.append(pt[0].tolist())
            pts.append(pt[1].tolist())
  
        pts.sort(key=lambda p: np.arctan2(p[1] - cent[1],
                                        p[0] - cent[0]))
        pts = pts[0::2]  # Deleting duplicates
        pts.insert(len(pts), pts[0])
        poly = Polygon(1.1*(np.array(pts)- cent) + cent,
                       facecolor= 'black', alpha=0.1, edgecolor = 'black')
        poly.set_capstyle('round')
        plt.gca().add_patch(poly)
        ax.annotate(self.dataIdentifier['className'][i], (cent[0], cent[1]), size = 16)
    for i, txt in enumerate(ptLabel):
      if(annotateHead == False or ( annotateHead == True and  i<26)):
        ax.annotate(txt, (z[i,ltnt1], z[i,ltnt2]), size = 12)

  #   plt.axis('off')
    # ticks = [-1.5, -1., -0.5, 0., 0.5, 1., 1.5]
    # ticklabels = ['-1.5', '-1', '-0.5', '0','0.5', '1', '1.5']
    # plt.xticks(ticks, ticklabels, fontsize=18)
    # plt.yticks(ticks, ticklabels, fontsize=18)
    plt.xlabel('z{:d}'.format(ltnt1), size = 18)
    plt.ylabel('z{:d}'.format(ltnt2), size = 18)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(saveFileName)
    
    return fig, ax
  
  def getMaterialProperties(self, decoded):
      
    def unlognorm(x, scaleMax, scaleMin):
      return 10**(x*(scaleMax-scaleMin) + scaleMin)
    
    youngModulus = unlognorm(decoded[:,self.dataInfo['ElasticModulus']['idx']], \
                              self.dataInfo['ElasticModulus']['scaleMax'],\
                              self.dataInfo['ElasticModulus']['scaleMin'])
    physicalDensity = unlognorm(decoded[:,self.dataInfo['MassDensity']['idx']],
                        self.dataInfo['MassDensity']['scaleMax'],
                        self.dataInfo['MassDensity']['scaleMin'])
    youngModulus = youngModulus*1e9 # convert from Pa
    physicalDensity = physicalDensity*(0.001/(1e-2)**3) # convert from g/cm^3 to kg/m^3
    
    return youngModulus, physicalDensity
    
  def getMaterialProperties_tempdependent(self, decoded):
    def unlognorm(x, scaleMax, scaleMin):
        return 10**(x*(scaleMax-scaleMin) + scaleMin)

    # MassDensity
    massDensity = unlognorm(
        decoded[:, self.dataInfo['MassDensity']['idx']],
        self.dataInfo['MassDensity']['scaleMax'],
        self.dataInfo['MassDensity']['scaleMin']
    )
    # Ea, Eb, Ec, Ed
    Ea = decoded[:, self.dataInfo['Ea']['idx']]
    Eb = decoded[:, self.dataInfo['Eb']['idx']]
    Ec = decoded[:, self.dataInfo['Ec']['idx']]
    Ed = decoded[:, self.dataInfo['Ed']['idx']]
    # Only denormalize if scaleMax != scaleMin
    for name, arr in zip(['Ea', 'Eb', 'Ec', 'Ed'], [Ea, Eb, Ec, Ed]):
        info = self.dataInfo[name]
        if info['scaleMax'] != info['scaleMin']:
            arr = arr * (info['scaleMax'] - info['scaleMin']) + info['scaleMin']
        # If not, arr is already correct
        if name == 'Ea':
            Ea = arr
        elif name == 'Eb':
            Eb = arr
        elif name == 'Ec':
            Ec = arr
        elif name == 'Ed':
            Ed = arr

    # Thermal Conductivity
    tc_info = self.dataInfo['ThermalConductivity']
    thermalConductivity = decoded[:, tc_info['idx']]
    if tc_info['scaleMax'] != tc_info['scaleMin']:
        thermalConductivity = thermalConductivity * (tc_info['scaleMax'] - tc_info['scaleMin']) + tc_info['scaleMin']
    # If not, use as is

    Ea = Ea * 1e9
    Eb = Eb * 1e9
    Ec = Ec * 1e9
    Ed = Ed * 1e9
    massDensity = massDensity * (0.001 / (1e-2)**3)  # g/cm^3 to kg/m^3
    # Thermal conductivity is already in W/mK in your sheet
    return Ea, Eb, Ec, Ed, massDensity, thermalConductivity
  def getMaterialProperties_structuralcost(self, decoded):
    def unlognorm(x, scaleMax, scaleMin):
      return 10**(x*(scaleMax-scaleMin) + scaleMin)

    youngModulus = unlognorm(decoded[:, self.dataInfo['ElasticModulus']['idx']],
                              self.dataInfo['ElasticModulus']['scaleMax'],
                              self.dataInfo['ElasticModulus']['scaleMin'])
    physicalDensity = unlognorm(decoded[:, self.dataInfo['MassDensity']['idx']],
                        self.dataInfo['MassDensity']['scaleMax'],
                        self.dataInfo['MassDensity']['scaleMin'])
    cost = unlognorm(decoded[:, self.dataInfo['Cost']['idx']],
                     self.dataInfo['Cost']['scaleMax'],
                     self.dataInfo['Cost']['scaleMin'])
    youngModulus = youngModulus*1e9 # convert from Pa
    physicalDensity = physicalDensity*(0.001/(1e-2)**3) # convert from g/cm^3 to kg/m^3
    return youngModulus, physicalDensity, cost
  def normalize_last_n(self, arr, n, min_val=-3, max_val=3):
    # Copy the original array to avoid modifying it in-place
    arr_copy = np.copy(arr)
    # Select the last n values and normalize them
    arr_copy[int(n)*-1:] = (arr_copy[int(n)*-1:] - min_val) / (max_val - min_val)
    return arr_copy

  # Unnormalizing function
  def unnormalize_last_n(self, arr, n, min_val=-3, max_val=3):
    # Copy the normalized array
    arr_copy = np.copy(arr)
    # Select the last n values and unnormalize them
    arr_copy[int(n)*-1:] = arr_copy[int(n)*-1:] * (max_val - min_val) + min_val
    return arr_copy
  
  def map_to_ellipse_torch_patch(self, arr, num_material_vars):
    last_n = int(num_material_vars/2)
    arrsize = arr.size - num_material_vars
    # print('arrsize', arrsize)
    # print('last_n', last_n)
    arr_copy = arr.copy()
    # arr_copy.retain_grad()
    z0 = arr_copy[arrsize:arrsize+last_n]
    z1 = arr_copy[arrsize+last_n:]



    cx = self.constraints['distance']['center'][0]
    cy = self.constraints['distance']['center'][1]
    a = self.constraints['distance']['a']
    b = self.constraints['distance']['b']
    theta = self.constraints['distance']['theta']

    # Uniform distribution in the ellipse
    r = np.sqrt(z0)
    phi = 2 * np.pi * z1

    # Unrotated ellipse coordinates
    x_e = r * a * np.cos(phi)
    y_e = r * b * np.sin(phi)

    # Apply rotation
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_r = x_e * cos_theta - y_e * sin_theta
    y_r = x_e * sin_theta + y_e * cos_theta

    # Translate to ellipse center
    x = x_r + cx
    y = y_r + cy

    # Update the input tensor in place
    arr_copy[arrsize:arrsize+last_n] = x
    arr_copy[arrsize+last_n:] = y

    return arr_copy

  def map_to_ellipse(self, arr):
    last_n = int(arr.size / 3)
    # print(last_n)
    arr_copy = arr.copy()
    # arr_copy.retain_grad()
    z0 = arr_copy[last_n:2 * last_n]
    z1 = arr_copy[2 * last_n:]



    cx = self.constraints['distance']['center'][0]
    cy = self.constraints['distance']['center'][1]
    a = self.constraints['distance']['a']
    b = self.constraints['distance']['b']
    theta = self.constraints['distance']['theta']

    # Uniform distribution in the ellipse
    r = np.sqrt(z0)
    phi = 2 * np.pi * z1

    # Unrotated ellipse coordinates
    x_e = r * a * np.cos(phi)
    y_e = r * b * np.sin(phi)

    # Apply rotation
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_r = x_e * cos_theta - y_e * sin_theta
    y_r = x_e * sin_theta + y_e * cos_theta

    # Translate to ellipse center
    x = x_r + cx
    y = y_r + cy

    # Update the input tensor in place
    arr_copy[last_n:2 * last_n] = x
    arr_copy[2 * last_n:] = y

    return arr_copy

  def map_to_ellipse_torch(self, arr):
    n = arr.size()
    last_n = int(n[0] / 3)
    print(last_n)
    arr_copy = arr.clone()
    arr_copy.retain_grad()
    # arr_copy.retain_grad()
    z0 = arr_copy[last_n:2 * last_n]
    z1 = arr_copy[2 * last_n:]



    cx = self.constraints['distance']['center'][0]
    cy = self.constraints['distance']['center'][1]
    a = self.constraints['distance']['a']
    b = self.constraints['distance']['b']
    theta = torch.tensor(self.constraints['distance']['theta'])

    # Uniform distribution in the ellipse
    r = torch.sqrt(z0)
    phi = 2 * torch.pi * z1

    # Unrotated ellipse coordinates
    x_e = r * a * torch.cos(phi)
    y_e = r * b * torch.sin(phi)

    # Apply rotation
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    x_r = x_e * cos_theta - y_e * sin_theta
    y_r = x_e * sin_theta + y_e * cos_theta

    # Translate to ellipse center
    x = x_r + cx
    y = y_r + cy

    # Update the input tensor in place
    arr_copy[last_n:2 * last_n] = x
    arr_copy[2 * last_n:] = y

    return arr_copy


  def functional_value(self,points_tensor):
    def unlognorm(x, scaleMax, scaleMin):
      return 10**(x*(scaleMax-scaleMin) + scaleMin)
    decoded = self.vaeNet.decoder(points_tensor)
    youngModulus = unlognorm(decoded[:,self.dataInfo['ElasticModulus']['idx']], 
                  self.dataInfo['ElasticModulus']['scaleMax'],
                  self.dataInfo['ElasticModulus']['scaleMin'])
    physicalDensity = unlognorm(decoded[:,self.dataInfo['MassDensity']['idx']],
              self.dataInfo['MassDensity']['scaleMax'],
              self.dataInfo['MassDensity']['scaleMin'])
    return youngModulus.detach().numpy().reshape((100,100)), physicalDensity.detach().numpy().reshape((100,100))

