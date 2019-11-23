
import torch.nn as nn
import torch.nn.functional as F
from Error.error import ErrorCorrection


class AlexNet(nn.Module, ErrorCorrection):

    def __init__(self, activation, error_rate):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )
        self.activation = self.Activation(activation)
        self.LUT = self.LookUpTable(error_rate)

    def forward(self, x):


        x = self.errorcorrection1(x, 1)
        x = self.activation(x)
        x = self.pool1(x)

        x = self.errorcorrection1(x, 2)
        x = self.activation(x)
        x = self.pool2(x)

        x = self.errorcorrection1(x, 3)
        x = self.activation(x)

        x = self.errorcorrection1(x, 4)
        x = self.activation(x)
        x = self.errorcorrection1(x, 5)
        x = self.activation(x)

        x = self.pool3(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x

def alexnet(activation, error_rate):
    net = AlexNet(activation, error_rate)
    return net