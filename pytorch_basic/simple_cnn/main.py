import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import argparse
import cnn_model as cnn
from tqdm import tqdm
import numpy as np

def run():
    torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="setting hyperparameters for simple CNN")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=int, default=0.0002)
    parser.add_argument("--num_epoch", type=int, default=10)

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.lr
    n_epoch = args.num_epoch

    if not os.path.isdir("./mnist_data"):
        os.mkdir("mnist_data")
        
    mnist_train = dset.MNIST("./mnist_data", train=True, transform=transforms.ToTensor(), target_transform=None,  download=True)
    mnist_test = dset.MNIST("./mnist_data", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = cnn.CNN(args).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_arr = []
    acc_arr = []
    for i in range(n_epoch):
        epoch_loss, epoch_acc = np.array([]), np.array([])
        for img, label in tqdm(train_loader):
            x = img.to(device)
            y_ = label.to(device)

            optimizer.zero_grad()
            output = model.forward(x)

            _, predict = torch.max(output,1)
            epoch_acc = np.append(epoch_acc, ((y_ == predict).sum().float() / batch_size).cpu())

            loss = loss_func(output, y_)
            loss.backward()
            optimizer.step()

            epoch_loss = np.append(epoch_loss, loss.detach().cpu().numpy())

        #detach??? graph?????? ????????? ????????? tensor??? return. graph ?????? ????????? ??????????????? ?????? ????????? ???????????? ???????????? ?????????
        #.cpu()detach()??? cpu??? ????????? edge??? ????????? (??? ?????????????????? ???), ??? ?????? ????????? ??? ????????? ?????? ?????????
        loss_arr.append(epoch_loss.mean())
        acc_arr.append(epoch_acc.mean())
        print("train loss = {}, train acc = {}".format(loss_arr[-1], acc_arr[-1]))

    correct = 0
    total = 0

    #???????????? ???????????? ???????????? ???????????? ????????? ??????
    with torch.no_grad():
        for img, label in tqdm(test_loader):
            x = img.to(device)
            y_ = label.to(device)

            output = model.forward(x)
            _, output_idx = torch.max(output,1)

            total += label.size(0)
            correct += (output_idx == y_).sum().float()

        print("test_acc : {}".format(100*correct/total))

#dataloader?????? num_worker??? ????????? ?????? main module ?????? ??????
if __name__ == '__main__':
    run()