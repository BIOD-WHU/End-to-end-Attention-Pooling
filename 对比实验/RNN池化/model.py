import torch 
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc = nn.Linear(2048, 4)
    
    def forward(self, x):
        x = x.cuda().squeeze(0)
        h = torch.randn((1, 2048)).cuda()
        for i in range(x.size(0)):
            h = F.relu(self.fc1(x[i].unsqueeze(0)) + self.fc2(h))
        output = self.fc(h)
        return output