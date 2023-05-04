import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
class Trainer():
  def __init__(
    self,
    model,
    trainX,
    trainY,
    testX,
    testY,
    optimizer,
    criterion,
    batchSize=32,
    device = "cpu",
    lossUpdateInterval=500,
    printClassification=True,
    dtype=torch.get_default_dtype(),
    scheduler = None,
    earlyStopper = None
  ):  
    self.model = model
    self.optimizer = optimizer
    self.criterion = criterion
    self.device = device
    self.printClassification = printClassification
    self.lossUpdateInterval = lossUpdateInterval
    self.model.to(self.device)
    self.scheduler = scheduler
    self.earlyStopper = earlyStopper

    self.trainX = torch.as_tensor(trainX, dtype=dtype, device=self.device)
    self.trainY = torch.as_tensor(trainY, dtype=torch.get_default_dtype(), device=self.device)
    self.testX = torch.as_tensor(testX, dtype=dtype, device=self.device)
    self.testY = torch.as_tensor(testY, dtype=torch.get_default_dtype(), device=self.device)
    trainData = TensorDataset(self.trainX, self.trainY)
    testData = TensorDataset(self.testX, self.testY)

    self.trainLoader = DataLoader(trainData, batch_size=batchSize)
    self.testLoader = DataLoader(testData, batch_size=batchSize)

    self.trainLoss = list()
    self.testLoss = list()
    self.trainClassificationPercentage = list()
    self.testClassificationPercentage = list()
  
  def train(self, epochs):
    for epoch in range(epochs):
      trainLoss = self.trainEpoch()
      testLoss = self.testEpoch()

      if self.scheduler is not None:
        self.scheduler.step()
      
      if self.earlyStopper is not None:
        if self.earlyStopper.shouldStop(testLoss):
          print(f"early stopping after {epoch+1} epochs")
          break

      trainClassification = self.trainClassification()
      testClassification = self.testClassification()

      self.trainLoss.append(trainLoss)
      self.testLoss.append(testLoss)

      self.trainClassificationPercentage.append(trainClassification)
      self.testClassificationPercentage.append(testClassification)

      classificationAccString = ''
      if self.printClassification:
        classificationAccString = f"train classification: {trainClassification: .4f} test classification: {testClassification: .4f}"
      if (epoch+1) % self.lossUpdateInterval == 0:
        print(f"[epoch {epoch+1}/{epochs}] train loss: {trainLoss: .4f} test loss: {testLoss: .4f} {classificationAccString}")
      if (epoch+1) == epochs:
        print(f"[epoch {epoch+1}/{epochs}] train loss: {trainLoss: .4f} test loss: {testLoss: .4f} {classificationAccString}")

  def trainEpoch(self):
    self.model.train() # set model to training mode
    for X, y in self.trainLoader:
      # forward pass
      output = self.model(X)
      loss = self.criterion(output, y)

      # backward pass
      self.optimizer.zero_grad() # zero out gradients
      loss.backward() # calculate gradients
      self.optimizer.step() # update weights

    loss = self.criterion(self.model(self.trainX), self.trainY) # calculate loss after epoch
    return loss.item()
  
  def testEpoch(self):
    self.model.eval() # set model to evaluation mode
    with torch.no_grad():
      loss = self.criterion(self.model(self.testX), self.testY)
      return loss.item()
      # for X, y in self.testLoader:
      #   output = self.model(X)
      #   loss = self.criterion(output, y)
      #   val_loss += loss.item() 

      # return val_loss 
    
  def trainClassification(self):
    self.model.eval()
    acc = 0
    with torch.no_grad():
      y_hat = self.model.hardPredict(self.trainX)
      acc = (y_hat == self.trainY).sum().item() / len(self.trainY)
      return acc
      # for X, y in self.trainLoader:
      #   output = self.model.hardPredict(X)
      #   acc += (output == y).sum().item() / len(y)
      
      # return acc / len(self.trainLoader)
  
  def testClassification(self):
    self.model.eval()
    acc = 0
    with torch.no_grad():
      y_hat = self.model.hardPredict(self.testX)
      acc = (y_hat == self.testY).sum().item() / len(self.testY)
      return acc
      # for X, y in self.testLoader:
      #   output = self.model.hardPredict(X)
      #   acc += (output == y).sum().item() / len(y)
      
      # return acc / len(self.testLoader)