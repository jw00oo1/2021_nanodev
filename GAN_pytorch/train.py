import os
import sys
dir = os.path.join(os.getcwd(),'GAN_pytorch')
sys.path.append(dir)

from init import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pylab as plt

import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs",type=int, default=200, help="number of epochs")
parser.add_argument("--batch_size",type=int, default=32, help="size of batches")
parser.add_argument("--lr",type=float, default=1e-4, help="learning rate")
parser.add_argument("--img_size", type=int, default=28, help="size of input image")
parser.add_argument("--n_latent",type=int, default=100, help="dimension of latent vector")
args = parser.parse_args()
print(args)

learning_rate = args.lr
epochs = args.n_epochs
batch_size = args.batch_size
latent_dim = args.n_latent
img_size = args.img_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = os.path.join(os.getcwd(),'mnist_data')

# if not os.path.exists('mnist_data'):
#     os.mkdir('mnist_data')
mnist_train = datasets.MNIST(root=data_dir,
                            train=True,
                            transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5],[0.5])]),
                            download=True)
mnist_test = datasets.MNIST(root=data_dir,
                            train=False,
                            transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5],[0.5])]),
                            download=True)

train_data_loader = DataLoader(dataset=mnist_train,
                        batch_size=args.batch_size,
                        shuffle=True)
test_data_loader= DataLoader(dataset=mnist_test,
                        batch_size=args.batch_size,
                        shuffle=True)

#loss
criterion = nn.BCELoss()

#init generator & discriminator
generator = Generator(args).to(device)
discriminator = Discriminator(args).to(device)
init_weight(generator)
init_weight(discriminator)

#optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
p_real_history = []
p_fake_history = []

for epoch in range(epochs):
    run_epoch(generator, discriminator,optimizer_g,optimizer_d,train_data_loader,device)

    p_real, p_fake = evaluate(generator,discriminator,test_data_loader,device)
    
    if ((epoch+1) % 50 == 0):
        print(f"{epoch+1} : p_real {p_real} %, p_fake {p_fake}")
        noise = torch.randn((8,1,img_size*img_size))
        plot_img(generator(noise).view(-1,1,img_size,img_size))