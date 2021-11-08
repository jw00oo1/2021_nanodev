import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.pooling import MaxPool2d
import torch.optim as optim
import torch.nn.init as init

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN,self).__init__()
        self.batch_size = args.batch_size
        self.layer = nn.Sequential(
            nn.Conv2d(1,16,5),  #output img size = floor((I-K+2P)/S+1)
            nn.ReLU(),
            nn.Conv2d(16,32,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64*3*3,100),
            nn.ReLU(),
            nn.Linear(100,10)
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(self.batch_size, -1)
        out = self.fc_layer(out)
        return out
