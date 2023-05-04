import torch
import torch.nn as nn
import math
import numpy as np

numChamps = 163
numRoles = 5

class SetElementProcessNetwork(nn.Module):
  def __init__(self, championEmbeddingSize = 32, roleEmbeddingSize = 3, outputSize = 10, nonLinearity = nn.ReLU):
    super(SetElementProcessNetwork, self).__init__()
    self.championEmbedding = nn.Embedding(numChamps+1, championEmbeddingSize)
    self.roleEmbedding = nn.Embedding(numRoles+1, roleEmbeddingSize)

    halfInputSize = math.floor((championEmbeddingSize+roleEmbeddingSize)/2)
    nn1 = nn.Linear(championEmbeddingSize+roleEmbeddingSize, halfInputSize)
    nn2 = nn.Linear(halfInputSize, outputSize)

    # nn.init.xavier_uniform_(nn1.weight, gain=np.sqrt(2))
    # nn.init.xavier_uniform_(nn2.weight, gain=np.sqrt(2))
    self.nn = nn.Sequential(
      nn1,
      nonLinearity(),
      nn2
    )

  def forward(self, champions, roles):
    championEmbeddings = self.championEmbedding(champions)
    roleEmbeddings = self.roleEmbedding(roles)

    combinedEmbeds = torch.cat((championEmbeddings, roleEmbeddings), dim=2)
    return self.nn(combinedEmbeds)

class DeepSetNeuralNet(nn.Module):
  def __init__(self, championEmbeddingSize = 32, roleEmbeddingSize =3, phiOutputSize = 10, nonLinearity = nn.ReLU):
    super(DeepSetNeuralNet, self).__init__()
    self.phi = SetElementProcessNetwork(championEmbeddingSize, roleEmbeddingSize, phiOutputSize, nn.Sigmoid)

    nn1 = nn.Linear(phiOutputSize*2, 5)
    nn2 = nn.Linear(5, 1)
    # nn.init.xavier_uniform_(nn1.weight, gain=np.sqrt(2))
    # nn.init.xavier_uniform_(nn2.weight, gain=np.sqrt(2))
    self.nn = nn.Sequential(
      nn1,
      nonLinearity(),
      nn2,
      nn.Sigmoid()
    )

  def forward(self, X):
    champions = X[:,:10]
    roles = X[:,10:]

    blueChampions = champions[:,:5]
    blueRoles = roles[:,:5]

    redChampions = champions[:,5:]
    redRoles = roles[:,5:]

    blueTeamEmbeddings = self.phi(blueChampions, blueRoles)
    redTeamEmbeddings = self.phi(redChampions, redRoles)

    # rho (aggregation function) is just sum TODO: try other agg functions later 
    blueTeamAgg = torch.sum(blueTeamEmbeddings, dim=1)
    redTeamAgg = torch.sum(redTeamEmbeddings, dim=1)
    combinedAgg = torch.cat((blueTeamAgg, redTeamAgg), dim=1)

    return self.nn(combinedAgg)

  def hardPredict(self, X):
    return torch.round(self(X))