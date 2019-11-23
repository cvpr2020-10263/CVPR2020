# VGG_16 RELU

import torch.nn as nn
import torch.nn.functional as F
from Error.error import ErrorCorrection


class ResNet(nn.Module, ErrorCorrection):

    def __init__(self, activation, error_rate):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch1 = nn.BatchNorm2d(64)

        '''Layer 1'''
        #Basic Block  0
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.batch2= nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.batch3 = nn.BatchNorm2d(64)


        #Basic Block 1
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1,bias = False)
        self.batch4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1,bias = False)
        self.batch5 = nn.BatchNorm2d(64)

        '''Layer 2'''
        #Block 0
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.batch6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.batch7 = nn.BatchNorm2d(128)

        self.rcon1 = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
        self.rb1 = nn.BatchNorm2d(128)

        #Block 1
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch9 = nn.BatchNorm2d(128)

        '''Layer 3'''
        #Block 0
        self.conv10 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.batch10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.batch11 = nn.BatchNorm2d(256)

        self.rcon2 = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
        self.rb2 = nn.BatchNorm2d(256)

        #Block1
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.batch13 = nn.BatchNorm2d(256)

        '''Layer 4'''
        #Block 0
        self.conv14 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.batch14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch15 = nn.BatchNorm2d(512)

        self.rcon3 = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.rb3 = nn.BatchNorm2d(512)

        #Block 1
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch16 = nn.BatchNorm2d(512)
        self.conv17 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch17 = nn.BatchNorm2d(512)

        self.linear = nn.Linear(512, 100)

        self.activation = self.Activation(activation)
        self.LUT = self.LookUpTable(error_rate)

    def forward(self, x):


        x = self.errorcorrection1(x, 1)
        x = self.batch1(x)
        x_0 = self.activation(x)


        x = self.errorcorrection1(x_0.clone(), 2)
        x = self.batch2(x)
        x = self.activation(x)
        x = self.errorcorrection1(x, 3)
        x = self.batch3(x)
        x = self.activation(x)
        x_1 = x + x_0

        x = self.errorcorrection1(x_1.clone(), 4)
        x = self.batch4(x)
        x = self.activation(x)
        x = self.errorcorrection1(x, 5)
        x = self.batch5(x)
        x = self.activation(x)
        x_2 = x + x_1

        x = self.errorcorrection1(x_2.clone(), 6)
        x = self.batch6(x)
        x = self.activation(x)
        x = self.errorcorrection1(x, 7)
        x = self.batch7(x)
        x = self.activation(x)
        x_2 = self.ResErrorCorrection(x_2, 1)
        x_2 = self.rb1(x_2)
        x_2 = self.activation(x_2)

        x_3 = x + x_2

        x = self.errorcorrection1(x_3.clone(), 8)
        x = self.batch8(x)
        x = self.activation(x)
        x = self.errorcorrection1(x, 9)
        x = self.batch9(x)
        x = self.activation(x)
        x_4 = x + x_3

        x = self.errorcorrection1(x_4.clone(), 10)
        x = self.batch10(x)
        x = self.activation(x)
        x = self.errorcorrection1(x, 11)
        x = self.batch11(x)
        x = self.activation(x)
        x_4 = self.ResErrorCorrection(x_4,2)
        x_4 = self.rb2(x_4)
        x_4 = self.activation(x_4)

        x_5 = x + x_4

        x = self.errorcorrection1(x_5.clone(), 12)
        x = self.batch12(x)
        x = self.activation(x)
        x = self.errorcorrection1(x, 13)
        x = self.batch13(x)
        x = self.activation(x)
        x_6 = x + x_5

        x = self.errorcorrection1(x_6.clone(), 14)
        x = self.batch14(x)
        x = self.activation(x)
        x = self.errorcorrection1(x, 15)
        x = self.batch15(x)
        x = self.activation(x)
        x_6 = self.ResErrorCorrection(x_6,3)
        x_6 = self.rb3(x_6)
        x_6 = self.activation(x_6)

        x_7 = x + x_6

        x = self.errorcorrection1(x_7.clone(), 16)
        x = self.batch16(x)
        x = self.activation(x)
        x = self.errorcorrection1(x, 17)
        x = self.batch17(x)
        x = self.activation(x)
        x = x + x_7

        out = F.avg_pool2d(x, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(args):
    net = ResNet(args.act, args.error)
    return net