# based off of the following stackoverflow post: https://stackoverflow.com/a/73704579
import numpy as np

class EarlyStopper:
  def __init__(self, patience=1, delta=0):
    self.patience = patience
    self.delta = delta
    self.bestLoss = np.inf
    self.counter = 0
      
  def shouldStop(self, loss):
    if loss < self.bestLoss:
      self.bestLoss = loss
      self.counter = 0
    elif loss > self.bestLoss + self.delta:
      self.counter += 1
      if self.counter >= self.patience:
        return True
    return False