import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.pooling import MaxPool2d
import torch.optim as optim
import torch.nn.init as init

def conv_2_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    
    return model

def conv_3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )

    return model

class VGG(nn.Module):
    def __init__(self, args, base_dim, num_classes=2):
        super(VGG,self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),
            conv_2_block(base_dim, 2*base_dim),
            conv_3_block(2*base_dim, 4*base_dim),
            conv_3_block(4*base_dim, 8*base_dim),
            conv_3_block(8*base_dim, 8*base_dim),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*7*7, 100),
            nn.ReLU(True),
            nn.Linear(100,20),
            nn.ReLU(),
            nn.Linear(20,num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x