import os
import pandas as pd
import torch
import torch.utils.data as data

class ImageDataset(data.Dataset):
    def __init__(self, csvPath, scorePath):
        super(ImageDataset, self).__init__()
        csv = pd.read_csv(csvPath)
        self.imageNames = list(csv["name"])
        self.labels = list(csv["label"])
        self.scorePath = scorePath
    
    def __getitem__(self, index):
        imageName = self.imageNames[index]
        selectedPatches = torch.load(os.path.join(self.scorePath, imageName + ".pki"))
        feature= torch.stack([patch[1] for patch in selectedPatches])
        if self.labels[index] == "Normal":
            label = 0
        elif self.labels[index] == "Benign":
            label = 1
        elif self.labels[index] == "InSitu":
            label = 2
        elif self.labels[index] == "Invasive":
            label = 3
        return feature, label
    
    def __len__(self):
        return len(self.imageNames)