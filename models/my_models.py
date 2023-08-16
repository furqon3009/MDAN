
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
device = torch.device('cuda:1')
import random
from torch.autograd import Variable
# from utils import *
""" CNN Model """


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class CNN_RUL(nn.Module):
    def __init__(self, input_dim, hidden_dim,dropout):
        super(CNN_RUL, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout=dropout
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            Flatten(),
            nn.Linear(8, self.hidden_dim)) 
        self.regressor= nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),   
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim//2) ,  
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim//2, 1))    
    def forward(self, src):
        features = self.encoder(src)
        predictions = self.regressor(features)
        return predictions, features


""" LSTM Model """
class LSTM_RUL(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, bid, device):
        super(LSTM_RUL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bid = bid
        self.dropout = dropout
        self.device = device
        # encoder definition
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=self.bid)
        # regressor
        self.regressor= nn.Sequential(
            nn.Linear(self.hidden_dim+self.hidden_dim*self.bid, self.hidden_dim),   
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim//2) ,  
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim//2, 1))
    def forward(self, src):
        # input shape [batch_size, seq_length, input_dim]
        # outputs = [batch size, src sent len,  hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        encoder_outputs, (hidden, cell) = self.encoder(src)
        # select the last hidden state as a feature
        features = encoder_outputs[:, -1:].squeeze()
        predictions = self.regressor(features)
        return predictions, features


class Mixup_RUL(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, bid, device):
        #print('mixup_RUL')
        super(Mixup_RUL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bid = bid
        self.dropout = dropout
        self.device = device
        self.hidden_dim2 = 30
        self.output_dim = 14
        # encoder definition
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=self.bid)
        # regressor
        self.regressor= nn.Sequential(
            nn.Linear(self.hidden_dim+self.hidden_dim*self.bid, self.hidden_dim),   
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim//2) ,  
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim//2, 1))
       
        # regressor 2
        self.regressor2= nn.Sequential(
            nn.Linear(self.hidden_dim+self.hidden_dim*self.bid, self.hidden_dim2),   
            nn.ReLU())

        # regressor 3
        self.regressor3= nn.Sequential(
            nn.Linear(1, self.output_dim),   
            nn.ReLU())

    def set_hidden_dim2(self, hid_dim2):
        self.hidden_dim2 = hid_dim2

    def set_output_dim(self, out_dim):
        self.output_dim = out_dim

    def forward(self, src):
        # input shape [batch_size, seq_length, input_dim]
        # outputs = [batch size, src sent len,  hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        encoder_outputs, (hidden, cell) = self.encoder(src)
#         encoder_outputs = F.dropout(torch.relu(encoder_outputs), p=0.5, training=self.training)
        # select the last hidden state as a feature

        features = encoder_outputs[:, -1:].squeeze()
        predictions = self.regressor(features)


        return predictions, features

    def forward_regressor_only(self, features):
        predictions = self.regressor(features)

        return predictions

    def forward_regressor2(self, src2):
        self.set_hidden_dim2(30)
        self.set_output_dim(14)
        encoder_outputs, (hidden, cell) = self.encoder(src2)

        features = encoder_outputs[:, -1:].squeeze()

        predictions = self.regressor2(features)
        predictions = predictions.unsqueeze(-1)
        predictions = self.regressor3(predictions)

        return predictions


