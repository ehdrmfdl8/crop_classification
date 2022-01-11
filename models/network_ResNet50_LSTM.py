import math
import torch
import torch.nn as nn
import numpy as np
from torchvision import models

class CNN_Encoder(nn.Module):
    def __init__(self, class_n, rate=0.1):
        super(CNN_Encoder, self).__init__()
        self.model = models.resnet50(pretrained=True)

    def forward(self, inputs):
        output = self.model(inputs)
        return output

class RNN_Decoder(nn.Module):
    def __init__(self, max_len, embedding_dim, num_features, class_n, rate):
        super(RNN_Decoder, self).__init__()
        self.lstm = nn.LSTM(max_len, embedding_dim)
        self.rnn_fc = nn.Linear(num_features * embedding_dim, 1000)
        self.final_layer = nn.Linear(1000 + 1000, class_n)
        self.dropout = nn.Dropout(rate)

    def forward(self, enc_out, dec_inp):
        hidden, _ = self.lstm(dec_inp)
        hidden = hidden.view(hidden.size(0), -1)
        hidden = self.rnn_fc(hidden)
        concat = torch.cat([enc_out, hidden], dim = 1) # enc_out + hidden
        fc_input = concat
        output = self.dropout(self.final_layer(fc_input))
        return output

class CNN2RNN(nn.Module):
    def __init__(self, max_len, embedding_dim, num_features, class_n, rate):
        super(CNN2RNN, self).__init__()
        self.cnn = CNN_Encoder(embedding_dim, rate)
        self.rnn = RNN_Decoder(max_len, embedding_dim, num_features, class_n, rate)

    def forward(self, img, seq):
        cnn_output = self.cnn(img)
        output = self.rnn(cnn_output, seq)

        return output

if __name__ == '__main__':
    max_len = 144
    embedding_dim = 512
    num_features = 9
    class_n = 32
    dropout_rate = 0.1
    model = CNN2RNN(max_len=max_len, embedding_dim=embedding_dim, num_features=num_features,
                class_n=class_n, rate=dropout_rate)
    print(model)

    img = torch.randn((1, 3, 256, 256))
    seq = torch.randn((1, 9, 144))
    x = model(img, seq)
    print(x.shape)