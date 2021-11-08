import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt

class convEncoder(nn.Module):
    def __init__(self, batch_size):
        super(convEncoder, self).__init__()
        self.batch_size = batch_size
        self.layer1 = nn.Sequential(
                        #batch x channel x width x height
                        #batch x (1->)16 x 28 x 28
                        nn.Conv2d(1,16,3,padding=1),                            # batch x 16 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(16),
                        #batch x (16->)32 x 28 x 28
                        nn.Conv2d(16,32,3,padding=1),                           # batch x 32 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32,64,3,padding=1),                           # batch x 32 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(2,2)                                       # batch x 64 x 14 x 14
        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,128,3,padding=1),                          # batch x 64 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(128,256,3,padding=1),                         # batch x 64 x 7 x 7
                        nn.ReLU()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(self.batch_size,-1)
        return out

class convDecoder(nn.Module):
    def __init__(self, batch_size):
        super(convDecoder, self).__init__()
        self.batch_size = batch_size
        self.layer1 = nn.Sequential(
                        #in_channel, out_channel, kernel_size, stride, padding, output_padding
                        #stride=2, padding=1, output_padding=1 makes feature map size double
                        #256 = last channel size of convEncoder
                        nn.ConvTranspose2d(256,128,3,2,1,1),                    # batch x 128 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.ConvTranspose2d(128,64,3,1,1),                       # batch x 64 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
                        nn.ConvTranspose2d(64,16,3,1,1),                        # batch x 16 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(16),
                        nn.ConvTranspose2d(16,1,3,2,1,1),                       # batch x 1 x 28 x 28
                        nn.ReLU()
        )

    def forward(self, x):
        out = x.view(self.batch_size, 256,-1)
        #dim_size = out.size()[2] ** (1/2)
        out = out.view(self.batch_size, 256, 7, 7)
        out = self.layer1(out)
        out = self.layer2(out)
        return out