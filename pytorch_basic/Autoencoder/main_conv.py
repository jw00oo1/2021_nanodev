import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import conv_encoder as ConvEncoder
import tqdm, os, argparse

parser = argparse.ArgumentParser(description="setting params of encoder")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=int, default=0.0002)
parser.add_argument("--num_epoch", type=int, default=2)
args = parser.parse_args()

batch_size = args.batch_size
lr = args.lr
num_epoch = args.num_epoch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
current = os.getcwd()

if __name__ == '__main__':

    #/dir로 하면 dir을 root로 보므로 상대경로 ./표시 or / 없이! 아니면 C:/와 join 됌
    data_dir = os.path.join(current, "mnist_data")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    mnist_train = dset.MNIST(data_dir, train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
    mnist_test = dset.MNIST(data_dir, train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

    train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size, shuffle=True,num_workers=0,drop_last=True)
    test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size, shuffle=False,num_workers=0,drop_last=True)

    print('working with {}...'.format(device))

    encoder = ConvEncoder.convEncoder(batch_size).to(device)
    decoder = ConvEncoder.convDecoder(batch_size).to(device)
    parameters = list(encoder.parameters()) + list(decoder.parameters())

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(parameters, lr=lr)

    #중간에 저장한 모델이 있으면 이어서 학습
    model_dir = os.path.join(current, "Autoencoder/model")
    try:
        encoder, decoder = torch.load(os.path.join(model_dir, "conv_autoencoder.pkl"))
        print("\n-----------model reload...---------")
    except:
        print("\n-----------model create...---------")
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        pass

    loss_arr = []
    for i in range(num_epoch):
        for j, [img, label] in enumerate(tqdm.tqdm(train_loader)):
            optimizer.zero_grad()
            x = img.to(device)

            output = encoder.forward(x)
            output = decoder.forward(output)
            loss = loss_func(output, x)
            loss.backward()
            optimizer.step()

            if j % 10 == 0:
                torch.save([encoder, decoder], os.path.join(model_dir, "./conv_autoencoder.pkl"))
            if j % 1000 == 0:
                print("\nloss : {}".format(loss.item()))
                ####
                loss_arr.append(loss.cpu())

    with torch.no_grad():
        for i in range(1):
            for j, [img, label] in enumerate(tqdm.tqdm(test_loader)):
                x = img.to(device)

                optimizer.zero_grad()
                output = encoder.forward(x)
                output = decoder.forward(output)

                if j % 1000 == 0:
                    print("\nloss : {}".format(loss.item()))
                    out_img = torch.squeeze(output.cpu().data)
                    print(out_img.size())

                    fig = plt.figure()
                    rows, cols, idx = 3, 2, 1
                    for k in range(rows):
                        ax = fig.add_subplot(rows, cols, idx)
                        ax.imshow(torch.squeeze(img[k]).numpy(),cmap='gray')
                        ax.set_xticks([]), ax.set_yticks([])
                        ax = fig.add_subplot(rows, cols, idx+1)
                        ax.imshow(out_img[k].numpy(),cmap='gray')
                        ax.set_xticks([]), ax.set_yticks([])
                        idx += 2
                    plt.savefig(os.path.join(current,'./Autoencoder/convEncoder_test.png'))
                    plt.show()
