import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
from loading_data import ImageDataset
from train import evaluate, saveFeature, selectPatches
from train import train
from torchlars import LARS

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

class EvalModel(nn.Module):
    def __init__(self, loadPath):
        super(EvalModel, self).__init__()
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 2)
        model.load_state_dict(torch.load(loadPath)["state_dict"])
        self.encoder = nn.Sequential(*(list(model.children())[:-1]))
        self.fc = model.fc
    
    def forward(self, x):
        features = self.encoder(x).squeeze(-1).squeeze(-1)
        scores = self.fc(features)
        return features, scores

def main():
    for fold in range(5):
        """trainTransform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(contrast=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        trainDataset = ImageDataset("splits/train_{}.csv".format(fold), trainTransform)
        trainLoader = data.DataLoader(trainDataset, batch_size=64, shuffle=False, num_workers=12)

        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 2)
        model.cuda()

        lossFn = nn.CrossEntropyLoss()

        optimizer = LARS(optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6))
        
        losses = []
        epoches = []
        patience = 0
        lowestLoss = 1e7
        for epoch in range(100):
            trainDataset.setMode("evaluate")
            scores = evaluate(trainLoader, model, batchSize=64)
            trainDataset.makeTrainData(np.array(scores))
            trainDataset.setMode("train")
            loss = train(trainLoader, model, lossFn, optimizer)
            losses.append(loss)
            epoches.append(epoch + 1)
            print("Fold: {}\tEpoch: {}\tLoss: {}".format(fold, epoch + 1, loss))
            plt.plot(epoches, losses)
            plt.xlabel("训练轮数")
            plt.ylabel("损失函数")
            plt.savefig("results/fold_{}.png".format(fold))
            #for x, y in zip(epoches, losses):
                #plt.text(x, y, "%.4f"%y)
            plt.close()
            if loss < lowestLoss: 
                patience = 0
                lowestLoss = loss
                torch.save({"epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            "best_loss": losses[-1],
                            "optimizer": optimizer.state_dict()}, "results/model_{}.pth".format(fold))               
            else:
                patience += 1
            if patience >= 5:
                break
        
        evalTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        trainDataset = ImageDataset("splits/train_{}.csv".format(fold), evalTransform)
        valDataset = ImageDataset("splits/val_{}.csv".format(fold), evalTransform)
        trainLoader = data.DataLoader(trainDataset, batch_size=70, shuffle=False)
        valLoader = data.DataLoader(valDataset, batch_size=70, shuffle=False)
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 2)
        model.load_state_dict(torch.load("results/model_{}.pth".format(fold))["state_dict"])
        model = nn.Sequential(*(list(model.children())[:-1]))
        model = model.cuda()
        saveFeature(trainLoader, model, fold)
        saveFeature(valLoader, model, fold)"""
        evalTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        trainDataset = ImageDataset("splits/train_{}.csv".format(fold), evalTransform)
        valDataset = ImageDataset("splits/val_{}.csv".format(fold), evalTransform)
        trainLoader = data.DataLoader(trainDataset, batch_size=70, shuffle=False)
        valLoader = data.DataLoader(valDataset, batch_size=70, shuffle=False)
        model = EvalModel("results/model_{}.pth".format(fold))
        model = model.cuda()
        #selectPatches(trainLoader, model, fold)
        selectPatches(valLoader, model, fold)
        
    

if __name__ == "__main__":
    main()