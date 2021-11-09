import torch
import torch.nn as nn
import torch.nn.functional as F

class convBlock(nn.Module):
    def __init__(self, in_dim, out_dim, mid_dim = None, act_fc=nn.ReLU(inplace=True)):
        super().__init__()
        if not mid_dim:
            mid_dim = out_dim
        self.layer = nn.Sequential(
             nn.Conv2d(in_dim, mid_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(mid_dim),
                act_fc,
                nn.Conv2d(mid_dim, out_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_dim),
                act_fc
        )
    
    def forward(self, x):
        return self.layer(x)

class Down(nn.Module):
    def __init__(self, in_dim, out_dim, act_fc = None):
        super(Down, self).__init__()
        if act_fc is None:
            self.maxPool_conv = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                convBlock(in_dim, out_dim)
            )
        else:
            self.maxPool_conv = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                convBlock(in_dim, out_dim, act_fn=act_fc)
            )
    
    def forward(self, x):
        return self.maxPool_conv(x)
        
class Up(nn.Module):
    def __init__(self, in_dim, out_dim, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = convBlock(in_dim, out_dim, in_dim // 2)
        else:
            self.up = nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2)
            self.conv = convBlock(in_dim, out_dim)
            
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffx = x2.size()[2] - x1.size()[2]
        diffy = x2.size()[3] - x1.size()[3]
        
        #https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x1 = F.pad(x1, [diffx // 2, diffx - diffx//2, diffy // 2, diffy - diffy // 2])
        x = torch.cat([x2,x1], dim = 1)
        return self.conv(x)
    
class outConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(outConv, self).__init__()
        self.conv  = nn.Conv2d(in_dim, out_dim, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
        