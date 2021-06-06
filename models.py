from torch import nn
import torch
from torch.nn import functional as F
import numpy as np

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

        self.conv1 = nn.Conv2d(self.classes, 4, 3, 1, 1)
        self.conv2 = nn.Conv2d(4, 4, 3, 1, 1)
        self.pool = nn.MaxPool2d(3,1,1)

        self.linear1 = nn.Linear(64, hidden_size)
        self.out = nn.Linear(hidden_size, 4)

        self.apply(self.init_weights)
    
    def forward(self, x):

        x = self.preprocess(x)
        #print(x.shape)
        z1 = self.relu(self.conv1(x))
        z2 = self.conv2(z1)
        z3 = self.relu(self.pool(z2))
        z4 = self.relu(self.linear1(z3.reshape(-1,64)))
        z5 = self.out(z4)
        #print(z2.shape)

        return z5

    def preprocess(self, x):
        x = x.reshape(-1,4,4)

        x_log = torch.log2(x + (x==0).int())
        onehot = F.one_hot(x_log.long(), num_classes=self.classes)

        # Transpose into (Batch, Channel, Row, Column) order
        output = np.transpose(onehot, axes=[0,3,1,2])

        mask = output.sum(axis=1)

        return output.float()
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
