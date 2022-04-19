import os
import random
import numpy as np
import pandas as pd
import PIL.Image as Image
import torch.utils.data as data
import torchvision.transforms as transforms

class ImageDataset(data.Dataset):
    def __init__(self, csvPath, transform=None):
        dataFrame = pd.read_csv(csvPath)
        self.images = list(dataFrame["name"])
        self.labels = list(dataFrame["label"])
        self.imageIndices = []
        self.patches = []
        for i, image in enumerate(self.images):
            patchDir = os.path.join("../BACH数据集/patches", image)
            for patch in os.listdir(patchDir):
                self.patches.append(patch)
                self.imageIndices.append(i)
        self.mode = "evaluate"
        self.transform = transform
        self.evalTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def makeTrainData(self, scores):
        imageIndices = np.array(self.imageIndices)
        order = np.lexsort((scores, imageIndices))
        index = np.empty(len(imageIndices), "bool")
        index[-10:] = True
        index[:-10] = (imageIndices[10:] != imageIndices[:-10])
        trainIndices = list(order[index])
        trainPatches = np.array(self.patches)[trainIndices]
        trainPatches = [os.path.join(self.images[i // 10], trainPatches[i]) for i in range(len(trainPatches))]
        trainLabels = self.labels
        trainData = list(zip(trainPatches, trainLabels))
        random.shuffle(trainData)
        self.trainPatches, self.trainLabels = zip(*trainData)

    def setMode(self, mode):
        self.mode = mode
    
    def __getitem__(self, index):
        if self.mode == "evaluate":
            patchPath = os.path.join("../BACH数据集/patches", os.path.join(self.images[self.imageIndices[index]], self.patches[index]))
            image = Image.open(patchPath)
            return self.evalTransform(image), self.labels[self.imageIndices[index]], self.patches[index]
        elif self.mode == "train":
            patchPath = os.path.join("../BACH数据集/patches", self.trainPatches[index])
            image = Image.open(patchPath)
            if self.trainLabels[index] == "Normal":
                label = 0
            else:
                label = 1
            return self.transform(image), label
        
    def __len__(self):
        if self.mode == "evaluate":
            return len(self.patches)
        elif self.mode == "train":
            return len(self.trainPatches) 