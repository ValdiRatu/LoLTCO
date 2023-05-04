import os
import sys
from pathlib import Path
os.chdir(Path(__file__).parent.resolve())

from sklearn.model_selection import train_test_split
import torch
import numpy as np
import json
import pickle 

from utils import (
  handle,
  run,
  main,
  saveModel,
  saveResults,
  plotLoss,
  plotClassification,
  getResults
)

from dataProcessing import (
    getDataOneHot,
    getDataEmbeddings,
    getTrainValData
)

from EarlyStopper import EarlyStopper

from models.LogisticRegression import LogisticRegression  
from models.EmbeddingNeuralNet import EmbeddingNeuralNet
from models.DeepSetNeuralNet import (
  DeepSetNeuralNet
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB

from Trainer import Trainer

DEFAULT_NUM_EPOCHS = 4890

@handle("logReg")
def logistic_regression():
  train, val = getTrainValData()
  X, y = getDataOneHot("match_data_subset_5k.pkl", data=train)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1273, random_state=42)

  lr = 0.00075
  wd = 0.0015
  n,d = X.shape
  
  model = LogisticRegression(inputSize=d)
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
  criterion = torch.nn.BCELoss()
  earlyStopper = EarlyStopper(patience=3, delta=0.0001)
  
  trainer = Trainer(
    model,
    X_train,
    y_train,
    X_test, 
    y_test,
    optimizer,
    criterion,
    batchSize=64,
    device="cuda",
    earlyStopper=earlyStopper
  )

  trainer.train(epochs=DEFAULT_NUM_EPOCHS)
  results = getResults(trainer)
  plotLoss(results, f"logReg-EARLY.png", f"Logistic Regression Loss (Early Stop) lr={lr} wd={wd}")
  saveResults(trainer, f"logReg-EARLY")
  saveModel(model, "log_reg_model_Early")



@handle("randomForest")
def random_forest():
  train, val = getTrainValData()
  X, y = getDataOneHot("match_data_subset_5k.pkl", data=train)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  criterion = 'gini'
  # model = RandomForestClassifier(criterion=criterion)
  # model.fit(X_train, y_train.flatten())
  # print(f"Random Forest with criterion {criterion} accuracy: {model.score(X_test, y_test.flatten())}")

  num_trees = [10, 50, 100, 200, 400]
  numTrails = 3 
  bestScore = 0
  bestModel = None
  for n in num_trees:
    testAcc = 0 
    trainAcc = 0
    for i in np.arange(numTrails):
      model = RandomForestClassifier(
        n_estimators=n,
        max_depth=5,
        criterion=criterion,
        bootstrap=False
      )
      model.fit(X_train, y_train.flatten())
      testAcc += model.score(X_test, y_test.flatten())
      trainAcc += model.score(X_train, y_train.flatten())
    print(f"Random Forest with {n} trees and criterion {criterion} train accuracy {trainAcc/numTrails: .4f},test accuracy: {testAcc/numTrails: .4f}")
    if testAcc/numTrails > bestScore:
      bestScore = testAcc/numTrails
      bestModel = model
  with open("../../models/random_forest.pkl", 'wb') as f:
    pickle.dump(bestModel, f)


@handle("naiveBayes")
def naive_bayes():
  train, val = getTrainValData()
  X, y = getDataOneHot("match_data_subset_5k.pkl", data=train)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1273, random_state=42)

  alpha = 15 
  bestScore = 0
  bestModel = None
  for alpha in [1, 3, 5, 15, 30, 50, 75, 100]:
    model = BernoulliNB(alpha = alpha)
    model.fit(X_train, y_train.flatten())
    score = model.score(X_test, y_test.flatten())
    if score > bestScore:
      bestScore = score
      bestModel = model
    print(f"Naive Bayes - alpha {alpha}, train accuracy: {model.score(X_train, y_train.flatten()): .4f}, test accuracy: {score: .4f}")
  
  with open("../../models/naive_bayes.pkl", 'wb') as f:
    pickle.dump(bestModel, f)

@handle("embeddedNet")
def embedded_net():
  train, val = getTrainValData()
  X, y = getDataEmbeddings("match_data_subset_1k.pkl", data=train)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1273, random_state=42)

  champEmbedSize = 50 
  roleEmbedSize = 3 
  hiddenNNSize = [100, 10]
  lr = 0.001
  wd = 0.0019

  for nonLinearity in [torch.nn.Sigmoid]:
    # model = EmbeddingNeuralNet(champEmbedSize, roleEmbedSize, hiddenNNSize, nonLinearity=torch.nn.ReLU)
    model = EmbeddingNeuralNet(champEmbedSize, roleEmbedSize, hiddenNNSize, nonLinearity=nonLinearity)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000, 4000], gamma=0.1)
    earlyStopper = EarlyStopper(patience=3, delta=0.0001)
    criterion = torch.nn.BCELoss()

    trainer = Trainer(
      model,
      X_train,
      y_train,
      X_test,
      y_test,
      optimizer,
      criterion,
      batchSize=64,
      device="cuda",
      dtype=torch.int,
      # earlyStopper=earlyStopper
    )
    nonLinearityName = nonLinearity.__name__

    trainer.train(epochs=DEFAULT_NUM_EPOCHS)
    results = getResults(trainer)
    saveResults(trainer, f"embeddedNet-FULL-{nonLinearityName}")
    plotLoss(results , f"embeddedNet-FULL-{nonLinearityName}.png", f"Embedded Net Loss (Early Stop) ({nonLinearityName})")
    plotClassification(results, f"embeddedNet-FULL-classification-{nonLinearityName}.png", f"Embedded Net (Early Stop) ({nonLinearityName})")
    print(f"Champion Embedding Size: {champEmbedSize}, Role Embed Size: {roleEmbedSize}, test accuracy: {trainer.testClassification(): .4f}")
    saveModel(model, f"embed_net_{nonLinearityName}_FULL")

@handle("deepSetNet")
def deep_set_net():
  train, val = getTrainValData()
  X, y = getDataEmbeddings("match_data_subset_5k.pkl", data=train)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1273, random_state=42)


  championEmbeddingSize = 32 
  roleEmbeddingSize = 3
  model = DeepSetNeuralNet(championEmbeddingSize=championEmbeddingSize,roleEmbeddingSize=roleEmbeddingSize, phiOutputSize=10, nonLinearity=torch.nn.Sigmoid)
  lr = 0.001
  wd = 0.00003

  optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
  earlyStopper = EarlyStopper(patience=4, delta=0.0003)
  criterion = torch.nn.BCELoss()

  trainer = Trainer(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    optimizer,
    criterion,
    batchSize=64,
    device="cuda",
    dtype=torch.int,
    earlyStopper=earlyStopper
  )

  trainer.train(epochs=DEFAULT_NUM_EPOCHS+1000)
  results = getResults(trainer)
  plotLoss(results , f"deepSetNet3-EARLY_lr_{lr}.png", "Deep Set Net Loss")
  plotClassification(results, f"deepSetNet3-EARLY-classification_lr_{lr}.png", "Deep Set Net")
  saveResults(trainer, f"deepSetNet3-EARLY_lr_{lr}")
  print(f"Deep Set Net test accuracy: {trainer.testClassification(): .4f}")
  saveModel(model, f"deep_set_net_EARLY3")

@handle("plotResults")
def plot_results():
  result_files = [
    "logReg-EARLY.json",
    "embed_net_loss_classification.json",
  ]
  file = result_files[0]
  with open (f"../../results/{file}", 'r') as f:
    results = json.load(f)
  
  plotClassification(results, f"logReg-EARLY-classification.png", "Logistic Regression (Early Stop)")

@handle("test")
def test():
  train, val = getTrainValData()
  X, y = getDataOneHot("match_data_subset_5k.pkl", data=train)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1273, random_state=42)
  print(np.where(X_train[5] == 1)[0])

   
if __name__ == "__main__":
    main()