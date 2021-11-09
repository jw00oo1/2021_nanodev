import torch
import torch.nn as nn
from .unet_patch import *

class uNet(nn.Module):
    def __init__(self, in_dim, n_classes, bilinear=True):
        super(uNet, self).__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.conv_in = convBlock(in_dim, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        #bilinear를 통해 upsample할 경우 채널이 감소하지 않으므로 2배하지 않는다.
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024,512//factor,bilinear)
        self.up2 = Up(512,256//factor,bilinear)
        self.up3 = Up(256,128//factor,bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.conv_out = outConv(64, n_classes)
        
    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5,x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.conv_out(x)
        
        return out