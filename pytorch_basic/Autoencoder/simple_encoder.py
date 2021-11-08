import torch
import torch.nn as nn
from torch.nn.modules import batchnorm

class Autoencoder(nn.Module):
    def __init__(self, batch_size=32):
        super(Autoencoder,self).__init__()

        self.batch_size = batch_size
        self.encoder = nn.Linear(28*28,20)
        self.decoder = nn.Linear(20,28*28)

    def forward(self,x):
        x = x.view(self.batch_size,-1)
        encoded = self.encoder(x)
        out = self.decoder(encoded).view(self.batch_size, 1, 28,28)
        return out