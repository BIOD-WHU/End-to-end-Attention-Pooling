import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import torch
import torch.nn.functional as F

def plotConfusionMatrix(confusionMatrix, imagePath):
    plt.imshow(confusionMatrix, cmap=plt.cm.Blues)
    plt.colorbar()
    indices = range(len(confusionMatrix))
    classes = ["正常", "良性", "原位癌", "浸润性癌"]
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.xlabel("预测值")
    plt.ylabel("真实值")
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] - False
    normalize = False
    fmt = ".2f" if normalize else "d"
    thresh = confusionMatrix.max() / 2.
    for firstIndex in range(len(confusionMatrix)):
        for secondIndex in range(len(confusionMatrix)):
            plt.text(secondIndex, firstIndex, format(confusionMatrix[firstIndex][secondIndex], fmt),
                horizontalalignment="center",
                color="white" if confusionMatrix[firstIndex, secondIndex] > thresh else "black")
    plt.savefig(imagePath)
    plt.close()

def train(loader, model, criterion, optimizer):
    runningLoss = 0
    for i, (inputs, labels) in enumerate(loader):
        optimizer.zero_grad()
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()
    return runningLoss / len(loader)

def evaluate(loader, model, fold):
    predictions = []
    truths = []
    for i, (inputs, labels) in enumerate(loader):
        with torch.no_grad():
            inputs = inputs.cuda()
            outputs = model(inputs)
            predictions.append(np.argmax(F.softmax(outputs, dim=1).cpu().numpy(), axis=1))
            truths.append(labels.numpy())
    predictions = np.array(predictions).flatten()
    truths = np.array(truths).flatten()
    accuracy = metrics.accuracy_score(truths, predictions)
    precision = metrics.precision_score(truths, predictions, average="macro")
    recall = metrics.recall_score(truths, predictions, average="macro")
    f1score = metrics.f1_score(truths, predictions, average="macro")
    print("Fold: {}\tAccuracy: {}\tPrecision: {}\tRecall: {}\tF1 score: {}".format(fold, accuracy, precision, recall, f1score))
    confusionMatrix = metrics.confusion_matrix(truths, predictions)
    plotConfusionMatrix(confusionMatrix, "results/confusion_matrix_{}.png".format(fold))
