import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

def set_seed(manualSeed):
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.manual_seed(manualSeed)
  torch.cuda.manual_seed(manualSeed)
  torch.cuda.manual_seed_all(manualSeed)
  np.random.seed(manualSeed)
  random.seed(manualSeed)

class Encoder(nn.Module):
    def __init__(self, encoderSettings):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(encoderSettings['inputDim'], encoderSettings['hiddenDim'])
        self.linear2 = nn.Linear(encoderSettings['hiddenDim'], encoderSettings['latentDim'])   # mu
        self.linear3 = nn.Linear(encoderSettings['hiddenDim'], encoderSettings['latentDim'])   # logvar

        self.kl = 0.0
        self.isTraining = False

    def forward(self, x):
        h = F.softplus(self.linear1(x))
        mu = self.linear2(h)
        logvar = self.linear3(h)
        std = torch.exp(0.5 * logvar)

        if self.isTraining:
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        self.kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar)
        return z

class Decoder(nn.Module):
  def __init__(self, decoderSettings):
    super(Decoder, self).__init__()
    self.linear1 = nn.Linear(decoderSettings['latentDim'], decoderSettings['hiddenDim'])
    self.linear2 = nn.Linear(decoderSettings['hiddenDim'], decoderSettings['outputDim'])

  def forward(self, z):
    z = F.softplus(self.linear1(z))
    z = torch.sigmoid(self.linear2(z)) # decoder op in range [0,1]
    return z
    
class VariationalAutoencoder(nn.Module):
  def __init__(self, vaeSettings):
    super(VariationalAutoencoder, self).__init__()
    self.encoder = Encoder(vaeSettings['encoder'])
    self.decoder = Decoder(vaeSettings['decoder'])

  def forward(self, x):
    z = self.encoder(x)
    return self.decoder(z)
  
#%%
class MaterialNetwork(nn.Module):
  def __init__(self, nnSettings):
    self.nnSettings = nnSettings
    self.outputDim = nnSettings['outputDim'];
    super().__init__();
    self.layers = nn.ModuleList();
    set_seed(1234);
    current_dim = nnSettings['inputDim'];
    for lyr in range(nnSettings['numLayers']): # define the layers
      l = nn.Linear(current_dim, nnSettings['numNeuronsPerLyr']);
      nn.init.xavier_normal_(l.weight);
      nn.init.zeros_(l.bias);
      self.layers.append(l);
      current_dim = nnSettings['numNeuronsPerLyr'];
    self.layers.append(nn.Linear(current_dim, self.outputDim));

  def forward(self, x):
    m = nn.LeakyReLU();
    for layer in self.layers[:-1]: # forward prop
      x = m(layer(x))

    opLayer = self.layers[-1](x)
    z = self.nnSettings['zMin'] + self.nnSettings['zRange']*torch.sigmoid(opLayer) # scale op [-1,1]
    return z
  

#%%
class TopologyNetwork(nn.Module):
  def __init__(self, nnSettings):
    self.inputDim = nnSettings['inputDim']; # x and y coordn of the point
    self.outputDim = nnSettings['outputDim']
    super().__init__()
    self.layers = nn.ModuleList()
    set_seed(1234)
    current_dim = self.inputDim
    for lyr in range(nnSettings['numLayers']): # define the layers
      l = nn.Linear(current_dim, nnSettings['numNeuronsPerLyr'])
      nn.init.xavier_normal_(l.weight)
      nn.init.zeros_(l.bias)
      self.layers.append(l)
      current_dim = nnSettings['numNeuronsPerLyr']
    self.layers.append(nn.Linear(current_dim, self.outputDim))
    self.bnLayer = nn.ModuleList()
    for lyr in range(nnSettings['numLayers']): # batch norm
      self.bnLayer.append(nn.BatchNorm1d(nnSettings['numNeuronsPerLyr']))

  def forward(self, x):
    m = nn.LeakyReLU()
    ctr = 0
    for layer in self.layers[:-1]: # forward prop
      x = m(self.bnLayer[ctr](layer(x)))
      ctr += 1
    opLayer = self.layers[-1](x)
    rho = 0.001 + torch.softmax(opLayer, dim = 1)
    return rho