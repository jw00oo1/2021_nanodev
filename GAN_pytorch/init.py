from numpy.lib.type_check import _imag_dispatcher
from numpy.random import normal
import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU, Tanh
from torch.nn.modules.linear import Linear

import torchvision.transforms as transforms
import torchvision.utils as utils

import numpy as np
from matplotlib import pyplot as plt

class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.n_latent = params.n_latent
        self.img_shape = (1,params.img_size,params.img_size)

        def fc_layer(n_in, n_out, normalize=True):
            layers = [nn.Linear(n_in, n_out)]
            if normalize:
                layers.append(nn.BatchNorm1d(n_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.1))
            return layers
        
        self.model = nn.Sequential(
            *fc_layer(self.n_latent,128,False),
            *fc_layer(128,256),
            *fc_layer(256,512),
            *fc_layer(512,1024),
            nn.Linear(1024,int(np.prod(self.img_shape))),
            nn.Tanh()
        )
        
    def forward(self, z):
        g_img = self.model(z)
        g_img = g_img.view(*self.img_shape)        #(g_img.size(0), *self.img_shape)
        return g_img

class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.img_shape = (1,params.img_size,params.img_size)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)),512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self, input_img):
        prediction = self.model(input_img.view(input_img.size(0), -1))
        return prediction

def init_weight(model):
    class_name = model.__class__.__name__

    if class_name.find('Linear') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def run_epoch(generator, discriminator, _optimizer_g, _optimizer_d, train_data_loader, device = 'cpu'):
    generator.train()
    discriminator.train()

    for feature_batch, label_batch in train_data_loader:
        feature_batch, label_batch = feature_batch.to(device), label_batch.to(device)
        img_size = feature_batch.size()

        z = torch.randn((img_size[0], img_size[2]*img_size[3]), device=device)

        _optimizer_d.zero_grad()
        
        #model(input) == model.forward(input)
        fake_data = generator(z)

        p_fake = discriminator(fake_data)
        p_real = discriminator(feature_batch.view(feature_batch.size(0),-1))

        #-1 for gradient ascending
        d_loss = torch.mean(-torch.log(p_real) - torch.log(1.-p_fake))

        #if loss based on cross entropy
        #criterion = nn.BCEloss()
        #d_loss = criterion(p_real, torch.ones_like(p_real)).to(device)
        #           + criterion(p_fake, torch.zeros_like(p_fake)).to(device)

        d_loss.backward()
        _optimizer_d.step()

        _optimizer_g.zero_grad()

        p_fake = generator(torch.randn((img_size[0], img_size[2]*img_size[3]), device=device))
        g_loss = -torch.log(p_fake).mean()

        g_loss.backward()
        _optimizer_g.step()

def evaluate(generator, discriminator, test_data_loader, device='cpu'):
    p_real, p_fake = 0. , 0.
    generator.eval()
    discriminator.eval()

    for feature_batch, label_batch in test_data_loader:
        img_size = feature_batch.size()
        feature_batch, label_batch = feature_batch.to(device), label_batch.to(device)
        noise = torch.randn((img_size[0], img_size[2]*img_size[3]))

        with torch.autograd.no_grad():
            p_real += (torch.sum(discriminator(feature_batch.view(-1,28*28))).item()) / 10000.
            p_fake += (torch.sum(discriminator(generator(noise.view(img_size[0],img_size[2]*img_size[3])))).item()) / 10000.

    return p_real, p_fake

def plot_img(img):
    #img size = (mini_batch, 1, 28, 28)
    img_grid = utils.make_grid(img.cpu().detach(), nrow=2, normalize= True).permute(1,2,0)
    plt.imshow(img_grid, cmap='gray')
    plt.show()