import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, model_name, input_size, embedding_size, hidden_size, output_size, num_layers, batch_size = 1):
        super(RNN, self).__init__()
        self.model_name = model_name
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        if model_name == "lstm":
            self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers)
        elif model_name == "gru":
            self.rnn = nn.GRU(self.embedding_size, self.hidden_size, self.num_layers)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, cell=None):
        out = self.encoder(input.view(1,-1))
        if self.model_name == "lstm":
            out, (hidden, cell) = self.rnn(out, (hidden, cell))
        elif self.model_name == "gru":
            out, hidden = self.rnn(out, hidden)
        out = self.decoder(out.view(self.batch_size, -1))
        return out, hidden, cell

    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        cell = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        return hidden, cell