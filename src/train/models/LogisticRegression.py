import torch
from torch import nn

class LogisticRegression(nn.Module):
  def __init__(self, inputSize, outputSize=1):
    super(LogisticRegression, self).__init__()
    self.linear = nn.Linear(inputSize, outputSize)
      
  def forward(self, x):
    return torch.sigmoid(self.linear(x))
      
  def hardPredict(self, X):
    return torch.round(self(X))
