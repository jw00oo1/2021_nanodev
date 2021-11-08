import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import rnn_model as rnn


n_hidden = 35
lr = 0.01
epochs = 1000

string = "hello pytorch. how long can a rnn cell remember?"
chars = "abcdefghijklmnopqrstuvwxyz ?!.,:;01"
char_list = [i for i in chars]
n_letters = len(char_list)

def onehot_encoding(string):
    start = np.zeros(shape=(n_letters), dtype=int)
    end = np.zeros(shape=(n_letters), dtype=int)
    start[-2] = 1
    end[-1] = 1
    for i in string:
        idx = char_list.index(i)
        onehot = np.zeros(shape=(n_letters), dtype=int)
        onehot[idx] = 1
        start = np.vstack([start, onehot])
    output = np.vstack([start, end])

    return output

def onehot_decoding(encoding):
    onehot = torch.Tensor.numpy(encoding)
    return char_list[onehot.argmax()]

if __name__ == '__main__':

    rnn = rnn.RNN(n_letters, n_hidden, n_letters)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)

    one_hot = torch.from_numpy(onehot_encoding(string)).type_as(torch.FloatTensor())

    for i in range(epochs):
        rnn.zero_grad()
        total_loss = 0
        hidden = rnn.init_hidden()

        for j in range(one_hot.size()[0]-1):
            input_ = one_hot[j:j+1,:]
            target = one_hot[j+1]

            output, hidden = rnn.forward(input_, hidden)
            loss = loss_func(output.view(-1), target.view(-1))
            total_loss += loss
            input_ = output

        total_loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(total_loss)

    start = torch.zeros(1, len(char_list))
    start[:,-2] = 1

    with torch.no_grad():
        hidden = rnn.init_hidden()
        input_ = start
        output_string = ""

        for i in range(len(string)):
            output, hidden = rnn.forward(input_, hidden)
            output_string += onehot_decoding(output.data)
            input_ = output

    print(output_string)