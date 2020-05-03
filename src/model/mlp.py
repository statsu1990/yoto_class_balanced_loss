
"""
YOU ONLY TRAIN ONCE: LOSS-CONDITIONAL TRAINING OF DEEP NETWORKS
# https://openreview.net/pdf?id=HyxY6JHKwr
For YOTO models, we condition the last layer of each　convolutional block. 
The conditioning MLP has one hidden layer with 256 units on Shapes3D and 512 units on CIFAR-10. 
At training time we sample the β parameter from log-normal distribution on the interval [0.125, 1024.] 
for Shapes3D and on the interval [0.125, 512.] for CIFAR-10.

FiLM: Visual Reasoning with a General Conditioning Layer
# https://arxiv.org/pdf/1709.07871.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, n_input, n_output, hidden_neurons=(512,), dropout_rate=0.1):
        super(MLP, self).__init__()

        n_neurons = (n_input,) + hidden_neurons + (n_output,)

        self.layers = nn.ModuleList()
        for i in range(len(n_neurons) - 1):
            self.layers.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
            #self.layers.append(nn.BatchNorm1d(n_neurons[i+1]))

        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h = x

        for i in range(len(self.layers)-1):
            h = self.dropout(self.act(self.layers[i](h)))
        h = self.layers[-1](h)

        return h

class MultiheadMLP(nn.Module):
    def __init__(self, n_input, n_outputs=(16, 32), 
                 common_hidden_neurons=(64,), 
                 multi_head_hidden_neurons=((128, 16), (128, 32)),
                 dropout_rate=0.1):
        super(MultiheadMLP, self).__init__()

        n_head = len(n_outputs)

        # common layer
        if common_hidden_neurons is not None:
            com_neurons = (n_input,) + common_hidden_neurons
            self.com_layers = []
            for i in range(len(com_neurons) - 1):
                self.com_layers.append(nn.Linear(com_neurons[i], com_neurons[i+1]))
                #self.com_layers.append(nn.BatchNorm1d(com_neurons[i+1]))
                self.com_layers.append(nn.ReLU(inplace=True))
                self.com_layers.append(nn.Dropout(dropout_rate))
            self.com_layers = nn.Sequential(*self.com_layers)
        else:
            com_neurons = (n_input,)
            self.com_layers = None

        # multi head layer
        self.head_layers = nn.ModuleList()
        for ih in range(n_head):
            if multi_head_hidden_neurons is not None and multi_head_hidden_neurons[ih] is not None:
                h_neurons = (com_neurons[-1],) + multi_head_hidden_neurons[ih] + (n_outputs[ih],)
            else:
                h_neurons = (com_neurons[-1],) + (n_outputs[ih],)

            h_layers = []
            for i in range(len(h_neurons) - 1):
                h_layers.append(nn.Linear(h_neurons[i], h_neurons[i+1]))
                if i < len(h_neurons) - 2:
                    #h_layers.append(nn.BatchNorm1d(h_neurons[i+1]))
                    h_layers.append(nn.ReLU(inplace=True))
                    h_layers.append(nn.Dropout(dropout_rate))
            self.head_layers.append(nn.Sequential(*h_layers))

    def forward(self, x):
        if self.com_layers is not None:
            h = self.com_layers(x)
        else:
            h = x

        hs = []
        for ly in self.head_layers:
            hs.append(ly(h))

        return hs












