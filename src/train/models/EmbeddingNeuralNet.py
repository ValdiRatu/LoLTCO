import numpy as np
import torch
from torch import nn

class EmbeddingNeuralNet(nn.Module):
    def __init__(self, championEmbeddingSize, roleEmbeddingSize, hiddenNNSize, nonLinearity = nn.ReLU):
      super(EmbeddingNeuralNet, self).__init__()
      numChamps = 163
      numRoles = 5

      # we use 1-indexing, we need to add 1 to the number of champions for padding because apparently this is common practice
      self.championEmbedding = nn.Embedding(numChamps+1, championEmbeddingSize)
      self.roleEmbedding = nn.Embedding(numRoles+1, roleEmbeddingSize)
      
      self.buildNN(championEmbeddingSize, roleEmbeddingSize, hiddenNNSize, nonLinearity)
    
    def buildNN(self, championEmbeddingSize, roleEmbeddingSize, hiddenNNSize, nonLinearity):
      # layerSizes = [championEmbeddingSize + roleEmbeddingSize] + hiddenNNSize + [1]
      layerSizes = [10*(championEmbeddingSize + roleEmbeddingSize)] + hiddenNNSize + [1]

      layers = []
      for inSize, outSize in zip(layerSizes[:-1], layerSizes[1:]):
        lin = nn.Linear(inSize, outSize)
        nn.init.xavier_uniform_(lin.weight, gain=np.sqrt(2))
        layers.append(lin)
        layers.append(nonLinearity())
      
      layers.pop(-1)
      layers.append(nn.Sigmoid())

      self.nn = nn.Sequential(*layers)

    def forward(self, X):
      champions = X[:,:10]
      roles = X[:,10:]

      champEmbeddings = self.championEmbedding(champions)
      roleEmbeddings = self.roleEmbedding(roles)

      combinedEmbeds = torch.cat((champEmbeddings, roleEmbeddings), dim=2)
      # combinedEmbedsSum = torch.sum(combinedEmbeds, dim=1)
      combinedEmbedsFlat = torch.flatten(combinedEmbeds, start_dim=1)
      test = self.nn(combinedEmbedsFlat)
      return test

    def hardPredict(self, X):
      return torch.round(self(X))

