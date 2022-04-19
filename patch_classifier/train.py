import os
from unittest.mock import patch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from loading_data import ImageDataset

def evaluate(loader, model, batchSize):
    scores = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, (image, __, __) in enumerate(loader):
            image = image.cuda()
            output = model(image)
            predictions = F.softmax(output, dim=-1)
            scores[i * batchSize : i * batchSize + image.size(0)] = predictions[:, 1].detach().cpu()
    return scores

def train(loader, model, lossFn, optimizer):
    runningLoss = 0
    for i, (image, label) in enumerate(loader):
        optimizer.zero_grad()
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        loss = lossFn(output, label)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()
    return runningLoss / len(loader.dataset)

def saveFeature(loader, model, fold):
    imageNames = loader.dataset.images
    for i, (image, __, __) in enumerate(loader):
        image = image.cuda()
        feature = model(image).detach().cpu()
        imagePath = os.path.join("features/{}".format(fold), "{}.pki".format(imageNames[i]))
        torch.save(feature, imagePath)

def selectPatches(loader, model, fold):
    imageNames = loader.dataset.images
    for i, (image, __, patchName) in enumerate(loader):
        print("[{} / {}]".format(i, len(loader)))
        with torch.no_grad():
            image = image.cuda()
            features, outputs = model(image)
            scores = F.softmax(outputs, dim=1)[:, 1]
            sortedScores, indices = torch.sort(scores, descending=True)
            selected = []
            for num in range(10):
                selected.append((patchName[indices[num]], features[indices[num]].cpu(), sortedScores[num].item()))
            for num in range(10):
                selected.append((patchName[indices[-num]], features[indices[-num]].cpu(), sortedScores[-num].item()))
            torch.save(selected, "selected_patches/{}/{}.pki".format(fold, imageNames[i]))