import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from gfootball.env.ssim import pytorch_ssim
import numpy as np
from torch.nn import functional as F

from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

# changing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = torch.finfo(torch.float).eps  # numerical logs

# batch_size set to 1 for loading it in VRNN, it should be set to 4 for training itself
batch_size = 1
z_dim = 256
z_dim_2 = 64
h_dim = 128
timespan = 2


class classifier(nn.Module):
    def __init__(self, h_dim, z_dim, n_layers, bias=False):
        super(classifier, self).__init__()

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        # Encoder
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, kernel_size=(7, 7), stride=(2, 2), padding=(1, 1)),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),

                                     nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 3), padding=(1, 0)),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),

                                     nn.Conv2d(32, 48, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
                                     nn.BatchNorm2d(48),
                                     nn.ReLU(),

                                     nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(3, 3), padding=(0, 0)),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU())

        # recurrence
        self.lstm = nn.LSTM(input_size=z_dim_2, hidden_size=h_dim, num_layers=self.n_layers, bias=True,
                            batch_first=True)

        self.fc = nn.Sequential(nn.Linear(z_dim, z_dim), nn.ReLU(),
                                nn.Linear(z_dim, z_dim_2), nn.ReLU())

        self.fc_2 = nn.Sequential(nn.Linear(timespan * h_dim, h_dim), nn.ReLU(),
                                  nn.Linear(h_dim, h_dim // 2), nn.ReLU(),
                                  nn.Linear(h_dim // 2, h_dim // 2), nn.ReLU(),
                                  nn.Linear(h_dim // 2, 2))
        # self.fc = nn.Sequential(nn.Linear(10*h_dim, 5*h_dim),nn.ReLU(),
        #                         nn.Linear(5*h_dim, 5*h_dim), nn.ReLU(),
        #                         nn.Linear(5*h_dim, h_dim), nn.ReLU(),
        #                         nn.Linear(h_dim, 3), nn.Sigmoid())

    def forward(self, x):
        input_shape = x.shape
        x = x.reshape(x.shape[0] * x.shape[1], 3, 64, 96)

        # input shape [b,t,w,h,c]
        # for t in range(x.size(1)):
        # encoder
        phi_x = self.encoder(x)

        phi_x = phi_x.reshape(batch_size, input_shape[1], 64 * 2 * 2)
        phi_x = self.fc(phi_x)
        # print('phixshape', phi_x.shape)
        # if t == 0:
        #     encoded_seq = phi_x
        # else:
        #     encoded_seq = torch.cat([encoded_seq, phi_x], 1)

        # recurrence
        out, _ = self.lstm(phi_x)
        # print(phi_x.shape)
        out = out.reshape(batch_size, timespan * h_dim)
        out = self.fc_2(out)
        # print('outshape', out.shape)
        # print(out.shape)
        return out
