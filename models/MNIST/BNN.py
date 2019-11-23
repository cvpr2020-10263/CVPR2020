import torch.nn as nn
from conf import modules
from Error.error import ErrorCorrection


class Model(nn.Module, ErrorCorrection):
    def __init__(self, args, error_rate):
        super(Model, self).__init__()

        self.conv1 = modules.BinaryConv2d(1, 16, kernel_size=5, padding=2)
        self.batch1 = nn.BatchNorm2d(16, momentum=args.momentum, eps=args.eps)
        self.pool1 = nn.MaxPool2d(2)
        self.tanh1 = modules.BinaryTanh()

        self.conv2 = modules.BinaryConv2d(16, 32, kernel_size=5, padding=2)
        self.batch2 = nn.BatchNorm2d(32, momentum=args.momentum, eps=args.eps)
        self.pool2 = nn.MaxPool2d(2)
        self.tanh2 = modules.BinaryTanh()

        self.fc = modules.BinaryLinear(7 * 7 * 32, 10)
        self.LUT = self.LookUpTable(error_rate)


    def forward(self, x):
        out = self.errorcorrection1(x, 1)
        out = self.batch1(out)
        out = self.pool1(out)
        out = self.tanh1(out)

        out = self.errorcorrection1(out, 2)
        out = self.batch2(out)
        out = self.pool2(out)
        out = self.tanh2(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def BNN(args):
    net = Model(args, args.error)
    return net