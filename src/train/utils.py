import argparse
import torch
import json
import numpy as np
import os

_funcs = {}

def handle(number):
    def register(func):
        _funcs[number] = func
        return func

    return register


def run(question):
    if question not in _funcs:
        raise ValueError(f"unknown question {question}")
    return _funcs[question]()


def main():
    parser = argparse.ArgumentParser()
    questions = sorted(_funcs.keys())
    parser.add_argument(
        "questions",
        choices=(questions + ["all"]),
        nargs="+",
        help="A question ID to run, or 'all'.",
    )
    args = parser.parse_args()
    for q in args.questions:
        if q == "all":
            for q in sorted(_funcs.keys()):
                start = f"== {q} "
                print("\n" + start + "=" * (80 - len(start)))
                run(q)

        else:
            run(q)

def saveModel(model, name):
    torch.save(model.state_dict(), f"../../models/{name}.pt")

def loadModel(model, name):
    model.load_state_dict(torch.load(f"../../models/{name}.pt"))
    return model

def getResults(trainer):
    return {
        "train": trainer.trainLoss,
        "test": trainer.testLoss,
        "trainClassification": trainer.trainClassificationPercentage,
        "testClassification": trainer.testClassificationPercentage,
    }

def saveResults(trainer, name):
    results = getResults(trainer)
    print(f"saving results to {name}.json")
    with open(f"../../results/{name}.json", "w") as f:
        json.dump(results, f)

def plotLoss(results, filename=None, title="plot"):
    import matplotlib.pyplot as plt
    x = np.arange(len(results["train"])) # should have same length as test

    y_train = results["train"]
    y_test = results["test"]
    plt.plot(x, y_train, label="train loss")
    plt.plot(x, y_test, label="test loss")

    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.legend()

    if filename is not None:
        filename = os.path.join("../../plots", filename)
        print("saving ", filename)
        plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.show()

def plotClassification(results, filename=None, title="plot"):
    import matplotlib.pyplot as plt
    x = np.arange(len(results["train"])) # should have same length as test

    y_train = results["trainClassification"]
    y_test = results["testClassification"]
    plt.plot(x, y_train, label="train classification")
    plt.plot(x, y_test, label="test classification")

    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("classification accuracy")

    plt.legend()

    if filename is not None:
        filename = os.path.join("../../plots/", filename)
        print("saving ", filename)
        plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.show()
