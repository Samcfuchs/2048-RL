#!/home/sam/miniconda3/envs/datasci/bin/python

from torch import nn
import torch
from torch.nn import functional as F
import numpy as np


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.rand((x.shape[0], 4))


class DQN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.classes = 11

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(self.classes+1, 4, 3, 1, 1)
        self.conv2 = nn.Conv2d(4, 4, 3, 1, 1)
        self.pool = nn.MaxPool2d(3,1,1)

        self.linear1 = nn.Linear(64, hidden_size)
        self.out = nn.Linear(hidden_size, 4)
        self.softmax = nn.Softmax(dim=1)

        self.apply(self.init_weights)
    
    def forward(self, x):

        x = self.preprocess(x)

        z1 = self.relu(self.conv1(x))

        z2 = self.conv2(z1)
        z3 = self.relu(z2)

        z4 = self.relu(self.linear1(z3.reshape(-1,64)))
        z5 = self.softmax(self.out(z4))

        #print(z2.shape)

        return z5

    def preprocess(self, x):
        x = x.reshape(-1,4,4)

        x_log = torch.log2(x + (x==0).int())
        onehot = F.one_hot(x_log.long(), num_classes=self.classes)

        # Transpose into (Batch, Channel, Row, Column) order
        output = np.transpose(onehot, axes=[0,3,1,2])

        mask = output.sum(axis=1, keepdim=True)

        output = torch.cat((mask,output), dim=1)

        return output.float()
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

class SmartCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.classes = 11

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.conv0 = nn.Conv2d(12, 4, kernel_size=(1,2),bias=True)
        self.conv1 = nn.Conv2d(12, 4, kernel_size=(2,1), bias=False)

        self.linear = nn.Linear(96, 256)
        self.output = nn.Linear(256,4)

        self.apply(self.init_weights)
    
    def forward(self, x):

        x, flips = self.preprocess(x)

        zh = self.relu(self.conv0(x))
        zv = self.relu(self.conv1(x))

        z1 = torch.cat((zh.reshape(-1,48), zv.reshape(-1,48)), dim=1)
        # 96 features

        z2 = self.relu(self.linear(z1))
        output = self.softmax(self.output(z2))

        return self.unscramble(output, flips)


    def rotate_normalize(self, x):
        """ Normalize a playing board so that the highest value is in the top left """
        # remember axis 0 is the vertical axis
        #assert x.shape == (-1, 4, 4)

        topleft = x[:,0,  0]
        topright= x[:,0, -1]
        botleft = x[:,-1, 0]
        botright= x[:,-1, -1]

        corners = torch.stack((topleft, topright, botleft, botright), dim=1)
        
        ix = torch.argmax(corners, dim=1)

        # Flips are ordered (vertical, horizontal)
        flips = torch.stack(((ix / 2).floor() == 1, ix % 2 == 1), dim=1)
        #print(flips.shape)

        for i in range(x.shape[0]):

            # OPTIMIZE
            if flips[i,0]:
                x[i] = torch.flip(x[i], (0,))
            if flips[i,1]:
                x[i] = torch.flip(x[i], (1,))

        return x, flips
    

    def unscramble(self, output, flips):

        for i in range(output.shape[0]):
            vert, horiz = flips[i]
            if vert:
                output[i] = output[i, [1,0,2,3]] # Swap up/down
            if horiz:
                output[i] = output[i, [0,1,3,2]] # Swap right/left
        
        return output


    def preprocess(self, x):
        x = x.reshape(-1,4,4)
        batch_size = x.shape[0]

        x, flips = self.rotate_normalize(x)

        x_log = torch.log2(x + (x==0).int())
        onehot = F.one_hot(x_log.long(), num_classes=self.classes)

        # Transpose into (Batch, Channel, Row, Column) order
        output = torch.zeros((batch_size, 12, 4, 4))
        output[:,1:] = np.transpose(onehot, axes=[0,3,1,2])

        #mask = output.sum(axis=1, keepdim=True)
        mask = torch.sum(output, dim=(1,), keepdim=False)
        output[:,0] = mask

        #output = np.transpose(onehot, axes=[0,3,1,2])

        #output = torch.cat((mask,output), dim=1)

        return output.float(), flips
    
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

