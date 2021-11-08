import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter

num_data = 1000
num_epoch = 10000

x = init.uniform_(torch.Tensor(num_data,1),-15,15)
noise = init.normal_(torch.FloatTensor(num_data,1),std=1)
y = (x**2) +3
y_noise = 2*(x+noise)+3

model = nn.Sequential(
    nn.Linear(1,6),
    nn.ReLU(),
    nn.Linear(6,10),
    nn.ReLU(),
    nn.Linear(10,6),
    nn.ReLU(),
    nn.Linear(6,1)
)

loss_func = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.0002)
loss_array = []

label = y_noise

writer = SummaryWriter()

for i in range(num_epoch):
    optimizer.zero_grad()
    output = model(x)

    loss = loss_func(output, label)
    loss.backward()
    optimizer.step()
    writer.add_scalar('loss',loss.data, i)

    if i%1000==0:
        print(loss.data)

writer.close()