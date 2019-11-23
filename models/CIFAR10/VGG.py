'''VGG11/13/16/19 in Pytorch.'''


# VGG RELU


import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from Error.error import ErrorCorrection
from Activation import PReLU



cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



class VGG(nn.Module, ErrorCorrection):

    def __init__(self, activation, error_rate, num_class=100 ):
        super(VGG, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batch3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batch4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batch5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch7 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.batch8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch10 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch13 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        self.activation = self.Activation(activation)
        self.classifier = nn.Linear(512, 10)
        self.LUT = self.LookUpTable(error_rate)




    def forward(self, x):

        x = self.errorcorrection1(x, 1)
        x = self.batch1(x)
        x = self.activation(x)


        x = self.errorcorrection1(x, 2)
        x = self.batch2(x)
        x = self.activation(x)
        x = self.pool1(x)

        x = self.errorcorrection1(x, 3)
        x = self.batch3(x)
        x = self.activation(x)

        x = self.errorcorrection1(x, 4)
        x = self.batch4(x)
        x = self.activation(x)

        x = self.pool2(x)

        x = self.errorcorrection1(x, 5)
        x = self.batch5(x)
        x = self.activation(x)

        x = self.errorcorrection1(x, 6)
        x = self.batch6(x)
        x = self.activation(x)

        x = self.errorcorrection1(x, 7)
        x = self.batch7(x)
        x = self.activation(x)

        x = self.pool3(x)

        x = self.errorcorrection1(x, 8)
        x = self.batch8(x)
        x = self.activation(x)

        x = self.errorcorrection1(x, 9)
        x = self.batch9(x)
        x = self.activation(x)

        x = self.errorcorrection1(x, 10)
        x = self.batch10(x)
        x = self.activation(x)

        x = self.pool4(x)

        x = self.errorcorrection1(x, 11)
        x = self.batch11(x)
        x = self.activation(x)

        x = self.errorcorrection1(x, 12)
        x = self.batch12(x)
        x = self.activation(x)

        x = self.errorcorrection1(x, 13)
        x = self.batch13(x)
        x = self.activation(x)

        x = self.pool5(x)
        output = x
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:

                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

