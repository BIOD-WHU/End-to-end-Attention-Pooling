import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from loading_data import ImageDataset
from torchlars import LARS
from train import train, evaluate
from model import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    for fold in range(5):
        csvPath = "../../patch_classifier/splits/train_{}.csv".format(fold)
        scorePath = "../../patch_classifier/selected_patches/{}".format(fold)
        trainDataset = ImageDataset(csvPath, scorePath)
        trainLoader = data.DataLoader(trainDataset, batch_size=1, shuffle=True)

        #model = nn.Linear(2048, 4)
        model = Model()
        model.cuda()

        criterion = nn.CrossEntropyLoss().cuda()
        #optimizer = LARS(optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6))
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)

        losses = []
        epochs = []
        lowestLoss = 1e9
        patience = 0
        for epoch in range(500):
            epochs.append(epoch + 1)
            runningLoss = train(trainLoader, model, criterion, optimizer)
            losses.append(runningLoss)
            plt.plot(epochs, losses)
            plt.xlabel("训练轮数")
            plt.ylabel("损失函数")
            plt.savefig("results/loss_{}.png".format(fold))
            plt.close()

            if runningLoss < lowestLoss:
                patience = 0
                lowestLoss = runningLoss
                torch.save({"epoch": epoch + 1,
                            "state_dict": model.state_dict()}, "results/model_{}.pth".format(fold))
            else:
                patience += 1
            if patience >= 5:
                break
        
        csvPath = "../../patch_classifier/splits/val_{}.csv".format(fold)
        scorePath = "../../patch_classifier/selected_patches/{}".format(fold)
        valDataset = ImageDataset(csvPath, scorePath)
        valLoader = data.DataLoader(valDataset, batch_size=1, shuffle=False)
        #model = nn.Linear(2048, 4)
        model = Model()
        model.load_state_dict(torch.load("results/model_{}.pth".format(fold))["state_dict"])
        model.cuda()
        evaluate(valLoader, model, fold)

            

if __name__ == "__main__":
    main()