from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from Error.error import ErrorCorrection


class Net(nn.Module, ErrorCorrection):
    def __init__(self, activation, error_rate):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.activation = self.Activation(activation)
        self.LUT = self.LookUpTable(error_rate)

    def forward(self, x):
        x = self.errorcorrection1(x, 1)
        x = self.activation(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.errorcorrection1(x, 2)
        x = self.activation(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.type(torch.float)
        return F.log_softmax(x, dim=1)


def LeNet(args):
    net = Net(args.act, args.error)
    return net