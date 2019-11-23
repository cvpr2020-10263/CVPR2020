import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class PRELU2(nn.Module):

    def __init__(self, num_parameters=1, init=6):
        super(PRELU2, self).__init__()
        self.weight1 = Parameter(torch.Tensor(num_parameters).fill_(init))
        self.weight2 = Parameter(torch.Tensor(num_parameters).fill_(12))



    def forward(self, x):
        x = x.type(torch.float)
        x = F.relu(x)
        x = torch.where(x >= self.weight2, ((self.weight2 + x) * 0.4), x)
        x = torch.where((x >= self.weight1) & (x <= self.weight2), (self.weight1 + x) * 0.5, x)

        return x


class PRELU3(nn.Module):

    def __init__(self, num_parameters=1, init=6):
        super(PRELU3, self).__init__()
        self.weight1 = Parameter(torch.Tensor(num_parameters).fill_(init))
        self.weight2 = Parameter(torch.Tensor(num_parameters).fill_(12))
        self.weight3 = Parameter(torch.Tensor(num_parameters).fill_(18))


    def forward(self, x):
        x = x.type(torch.float)
        x = F.relu(x)
        x = torch.where(x >= self.weight3, (self.weight3 + x) * 0.3, x)
        x = torch.where((x >= self.weight2) & (x <= self.weight3), ((self.weight2  + x) * 0.4), x)
        x = torch.where((x >= self.weight1) & (x <= self.weight2), (self.weight1 + x) * 0.5, x)

        return x


class PRELU4(nn.Module):

    def __init__(self, num_parameters=1, init=6):
        super(PRELU4, self).__init__()
        self.weight1 = Parameter(torch.Tensor(num_parameters).fill_(init))
        self.weight2 = Parameter(torch.Tensor(num_parameters).fill_(12))
        self.weight3 = Parameter(torch.Tensor(num_parameters).fill_(18))
        self.weight4 = Parameter(torch.Tensor(num_parameters).fill_(24))


    def forward(self, x):
        x = x.type(torch.float)
        x = F.relu(x)
        x = torch.where(x > self.weight4, (self.weight4 + x) * 0.2, x)
        x = torch.where((x >= self.weight3) & (x <= self.weight4), (self.weight3 + x) * 0.3, x)
        x = torch.where((x >= self.weight2) & (x <= self.weight3), ((self.weight2  + x) * 0.4), x)
        x = torch.where((x >= self.weight1) & (x <= self.weight2), (self.weight1 + x) * 0.5, x)

        return x


class PRELU5(nn.Module):

    def __init__(self, num_parameters=1, init=6):
        super(PRELU5, self).__init__()
        self.weight1 = Parameter(torch.Tensor(num_parameters).fill_(init))
        self.weight2 = Parameter(torch.Tensor(num_parameters).fill_(12))
        self.weight3 = Parameter(torch.Tensor(num_parameters).fill_(18))
        self.weight4 = Parameter(torch.Tensor(num_parameters).fill_(24))
        self.weight5 = Parameter(torch.Tensor(num_parameters).fill_(30))



    def forward(self, x):
        x = x.type(torch.float)
        x = F.relu(x)
        x = torch.where(x > self.weight5, (self.weight5 + x) * 0.1, x)
        x = torch.where((x >= self.weight4) & (x <= self.weight5), (self.weight4 + x) * 0.2, x)
        x = torch.where((x >= self.weight3) & (x <= self.weight4), (self.weight3 + x) * 0.3, x)
        x = torch.where((x >= self.weight2) & (x <= self.weight3), ((self.weight2  + x) * 0.4), x)
        x = torch.where((x >= self.weight1) & (x <= self.weight2), (self.weight1 + x) * 0.5, x)

        return x


class PRELU6(nn.Module):

    def __init__(self, num_parameters=1, init=6):
        super(PRELU6, self).__init__()
        self.weight1 = Parameter(torch.Tensor(num_parameters).fill_(init))
        self.weight2 = Parameter(torch.Tensor(num_parameters).fill_(12))
        self.weight3 = Parameter(torch.Tensor(num_parameters).fill_(18))
        self.weight4 = Parameter(torch.Tensor(num_parameters).fill_(24))
        self.weight5 = Parameter(torch.Tensor(num_parameters).fill_(30))
        self.weight6 = Parameter(torch.Tensor(num_parameters).fill_(36))




    def forward(self, x):
        x = x.type(torch.float)
        x = F.relu(x)
        x = torch.where(x > self.weight6, self.weight6 , x)
        x = torch.where((x >= self.weight5) & (x <= self.weight6), (self.weight5 + x) * 0.1, x)
        x = torch.where((x >= self.weight4) & (x <= self.weight5), (self.weight4 + x) * 0.2, x)
        x = torch.where((x >= self.weight3) & (x <= self.weight4), (self.weight3 + x) * 0.3, x)
        x = torch.where((x >= self.weight2) & (x <= self.weight3), (self.weight2 + x) * 0.4, x)
        x = torch.where((x >= self.weight1) & (x <= self.weight2), (self.weight1 + x) * 0.5, x)

        return x



class PRELU(nn.Module):


    def __init__(self, num_parameters=1, init=6):
        super(PRELU, self).__init__()
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))



    def forward(self, x):
        # print (self.weight)

        x = F.relu(x)
        x = x.type(torch.float)
        x = torch.where(x > self.weight, (self.weight + x) * 0.5, x)

        return x
