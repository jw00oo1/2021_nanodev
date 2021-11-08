import os
import wget
import torch
import torch.nn as nn
import unidecode
import string
import random
import re
import time, math
import lstm_model as Models
import sys, argparse

all_characters = string.printable
num_epochs = 1500
print_every = 100
plot_every = 10
chunk_len = 200
hidden_size = 100
batch_size = 1
num_layers = 1
embedding_size = 70
lr = 0.002

def data_donwload():
    try:
        os.chdir("./simple_rnn")
    except:
        sys.exit()

    if not os.path.exists("./data"):
        os.mkdir("./data")
        data_path = "./data"
        data_url = "https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tinyshakespeare/input.txt"
        wget.download(data_url, out=data_path)

def random_chunk(file, file_len):
    start_idx = random.randint(0, file_len - chunk_len)
    end_idx = start_idx + chunk_len + 1
    return file[start_idx:end_idx]

def char_tensor(string):
    """
        input = word(string type)
        ex) char_tensor("abce") = torch.tensor([0,1,2,3])
    """
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return tensor

def random_training_set(file, file_len):
    chunk = random_chunk(file, file_len)
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

def test(model):
    start_str = "b"
    inp = char_tensor(start_str)
    hidden, cell = model.init_hidden()
    x = inp
    print(start_str, end="")

    for i in range(200):
        output, hidden, cell = model(x, hidden, cell)

        output_dist = output.data.view(-1).div(0.8).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        predicted_char = all_characters[top_i]

        print(predicted_char, end="")

        x = char_tensor(predicted_char)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="setting hyperparameters for time-series data")
    parser.add_argument("--model_name", type=str, default="lstm")
    args = parser.parse_args()

    model_name = args.model_name

    data_donwload()

    n_characters = len(all_characters)
    print(all_characters)
    print('num_chars = ', n_characters)

    file = unidecode.unidecode(open('./data/input.txt').read())
    file_len = len(file)
    print('file_len = ', file_len)

    if model_name == "lstm":
        model = Models.RNN(model_name, n_characters, embedding_size, hidden_size, n_characters, num_layers)  
    elif model_name == "gru":
        model = Models.RNN(model_name, n_characters, embedding_size, hidden_size, n_characters, num_layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    #train
    for i in range(num_epochs):
        inp, label = random_training_set(file, file_len)
        hidden, cell = model.init_hidden()

        loss = torch.FloatTensor([0])
        optimizer.zero_grad()

        for j in range(chunk_len-1):
            x = inp[j]
            y_ = label[j].view(-1).type(torch.LongTensor)   #unsqueeze(0).type(torch.LongTensor)
            y, hidden, cell = model(x,hidden, cell)
            loss += loss_func(y,y_)
        
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            print("\n{}\n".format(loss/chunk_len))
            test(model)
            print("\n\n")