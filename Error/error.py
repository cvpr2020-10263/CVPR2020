import torch
import torch.nn as nn
import numpy as np
import math
import random
from Activation import PReLU
import os

#from train import error


#file = '/home/mlvcgpu/salman/MNIST/LUT1_' + str(bit_error) + '.npy'
'''
LUT = np.load('/home/mlvcgpu/salman/MNIST/LUT1_0.001.npy')
LUT = np.array((LUT))
'''

class ErrorCorrection():

    def LookUpTable(self, error_rate):
        path = './Error_Files/LUT1_' + str(error_rate) + '.npy'
        path = os.path.abspath(path)
        LUT = np.load(path)
        return LUT

    def Check12(self,x):
        x1 = x.clone()
        max_n = torch.max(x)
        numofbits = self.countBits(int(max_n))
        x1 = x1 * 2 ** (31 - numofbits)
        shap = x.shape
        x1 = torch.reshape(x1, (-1, 1))
        x1 = x1.detach()
        x1 = x1.cpu()
        x1 = x1.numpy()
        x1 = x1.astype(int)
        x1 = x1.astype(float)
        x1 = torch.from_numpy(x1).float()
        x1 = x1/ (2 ** (31 - numofbits))
        x1 = torch.reshape(x1, (shap[0], shap[1], shap[2], shap[3]))
        x1 = x1.cuda()
        x.data = x1.data
        return x


    def countBits(self,number):

        try:
            return int((math.log(number) /
                        math.log(2)) + 1);
        except:
            return 1

    def BitError(self, x):
        x1 = x.clone()
        max_n = torch.max(x)
        numofbits = self.countBits(int(max_n))
        global rat


        x1 = x1 * 2 ** (31 - numofbits)


        shap = x.shape
        parameters = shap[0] * shap[1] * shap[2] * shap[3]
        x1 = torch.reshape(x1, (-1, 1))
        x1 = x1.detach()
        x1 = x1.cpu()
        x1 = x1.numpy()
        x1 = x1.astype(int)

        try:
            number = random.randint(0, 30000000 + 1 - parameters)
        except:
            global NOPS
            NOPS = NOPS + 1
            # print ('Too big tensor to operate. Problem occurred')
            return x

        lut = self.LUT[number: number + parameters]
        lut = lut.reshape((-1, 1))
        x1 = np.bitwise_xor(x1, lut)

        x1 = x1.astype(float)
        x1 = torch.from_numpy(x1).float()

        x1 = x1/ (2 ** (31 - numofbits))
        x1 = torch.reshape(x1, (shap[0], shap[1], shap[2], shap[3]))
        x1 = x1.cuda()
        x.data = x1.data
        return x

    def Tonumpy(self, f3):
        f3 = f3.detach()
        f3 = f3.cpu()
        f3 = f3.numpy()
        return f3

    def Precision(self, x1, numofbits):

        x1 = x1 * 2 ** (31 - numofbits)

        return x1


    def RestorePrecision(self,x1,numofbits):

        x1 = x1 / (2 ** (31 - numofbits))
        return  x1



    def WeightCorruption(self, x, number):


        index = number
        weight = self.Conv_num(number)
        original_weight = weight.clone()
        max_n = torch.max(weight)
        numofbits = self.countBits(int(max_n))

        weight = weight * 2 ** (32 - numofbits)

        shap = weight.shape
        parameters = shap[0] * shap[1] * shap[2] * shap[3]
        x1 = weight
        x1 = torch.reshape(x1, (-1, 1))
        x1 = x1.detach()
        x1 = x1.cpu()
        x1 = x1.numpy()
        x1 = x1.astype(int)

        number = random.randint(0, 30000000 + 1 - parameters)
        lut = self.LUT[number: number + parameters]
        lut = lut.reshape((-1, 1))

        x1 = np.bitwise_xor(x1, lut)
        '''
        sign = np.sign(x1)
        check = sign * x1
        check = np.bitwise_and(check, 4294967040)
        x1 = check * sign
        '''
        x1 = x1.astype(float)

        x1 = torch.from_numpy(x1).float()
        x1 = x1 / (2 ** (32 - numofbits))

        x1 = torch.reshape(x1, (shap[0], shap[1], shap[2], shap[3]))
        x1 = x1.cuda()

        weight = x1

        self.WeightAssign(index,weight)

        x1 = self.BitError(x.clone())
        x = self.Convolution(x1.clone(), index)

        self.WeightAssign(index, original_weight)

        return x

    def WeightAssign(self, number,weight):


        if number == 1:
            #print (str(len(torch.nonzero(self.conv1.weight - weight))))
            weight = torch.nn.Parameter(weight)
            self.conv1.weight = weight
            #print (str(len(torch.nonzero(self.conv1.weight - weight))))


        elif number == 2:
            weight = torch.nn.Parameter(weight)
            self.conv2.weight = weight

        elif number == 3:
            weight = torch.nn.Parameter(weight)
            self.conv3.weight = weight

        elif number == 4:
            weight = torch.nn.Parameter(weight)
            self.conv4.weight = weight

        elif number == 5:
            weight = torch.nn.Parameter(weight)
            self.conv5.weight = weight

        elif number == 6:
            weight = torch.nn.Parameter(weight)
            self.conv6.weight = weight

        elif number == 7:
            weight = torch.nn.Parameter(weight)
            self.conv7.weight = weight

        elif number == 8:
            weight = torch.nn.Parameter(weight)
            self.conv8.weight = weight

        elif number == 9:
            weight = torch.nn.Parameter(weight)
            self.conv9.weight = weight

        elif number == 10:
            weight = torch.nn.Parameter(weight)
            self.conv10.weight = weight

        elif number == 11:
            weight = torch.nn.Parameter(weight)
            self.conv11.weight = weight

        elif number == 12:
            weight = torch.nn.Parameter(weight)
            self.conv12.weight = weight

        elif number == 13:
            weight = torch.nn.Parameter(weight)
            self.conv13.weight = weight


    def  errorcorrection(self,x1,x2,x3):

        a = x1.clone()

        max_n = torch.max(x1)
        numofbits = self.countBits(int(max_n))
        x1 = self.Precision(x1,numofbits)
        x2 = self.Precision(x2, numofbits)
        x3 = self.Precision(x3, numofbits)

        #x6 = self.Precision(x6, numofbits)




        x1 = self.Tonumpy(x1)
        x2 = self.Tonumpy(x2)
        x3 = self.Tonumpy(x3)

        #x6 = Tonumpy(x6)




        x1 = x1.astype(int)
        x2 = x2.astype(int)
        x3 = x3.astype(int)
        #x6 = x6.astype(int)


        #a1 = self.ToPositive(x1.copy(),x2.copy())
        #a2 = self.ToPositive(x1.copy(),x3.copy())
        #a3 = self.ToPositive(x2.copy(),x3.copy())

        a1 = np.bitwise_and(x1,x2)
        a2 = np.bitwise_and(x1,x3)
        a3 = np.bitwise_and(x2,x3)


        result = np.bitwise_or(a1, a2)
        result = np.bitwise_or(result, a3)

        #result = np.bitwise_or(f, result)

        diff = result - x3
        #print (np.count_nonzero(x1 - x3))

        #print (np.count_nonzero(diff))


        x1 = result
        x1 = x1.astype(float)

        x1 = torch.from_numpy(x1).float()
        x2 = torch.from_numpy(x2).float()


        x1 = self.RestorePrecision(x1, numofbits)
        x1 = x1.cuda()


        a.data = x1.data


        return a


    def Convolution(self, x, number):

        fn = getattr(self, 'conv' + str(number), None)
        x = fn(x)

        return x

    def Conv_num(self,number):

        weight = 'self.conv' + str(number) + '.weight'
        x = eval(weight)
        return x

    def errorcorrection1(self, x, number):

        while 1:

            xoriginal = self.Convolution(x.clone(), number)

            x1 = xoriginal.clone() #self.Convolution(x.clone(), number)

            #if ECL == 'no':
            #return x1

            x1 = self.BitError(x1)
            #return x1

            x2 = xoriginal.clone()
            x2 = self.BitError(x2)


            x3 = xoriginal.clone()
            x3 = self.BitError(x3)

            x = self.errorcorrection(x1, x2, x3)
            return x

    def ResErrorCorrection(self, x, number):

        xoriginal = self.ResConvl(x.clone(), number)
        x1 = xoriginal.clone()
        #return x1

        x1 = self.BitError(x1)
        #return x1


        x2 = xoriginal.clone()
        x2 = self.BitError(x2)

        x3 = xoriginal.clone()
        x3 = self.BitError(x3)

        x = self.errorcorrection(x1, x2, x3)
        return x

    def ResConvl(self, x, number):

        fn = getattr(self, 'rcon' + str(number), None)
        x = fn(x)

        return x

    def Activation (self,Activation):

        if Activation == 'PRELU1':

            return PReLU.PRELU()

        elif Activation == 'PRELU2':

            return PReLU.PRELU2()

        elif Activation == 'PRELU3':

            return PReLU.PRELU3()

        elif Activation == 'PRELU4':

            return PReLU.PRELU4()

        elif Activation == 'PRELU5':

            return PReLU.PRELU5()

        elif Activation == 'PRELU6':

            return PReLU.PRELU6()

        elif Activation == 'RELU':
            return nn.ReLU()

        elif Activation == 'RELU6':
            return nn.ReLU6()

        elif Activation == 'RRELU':
            return nn.RReLU()

        elif Activation == 'ELU':
            return nn.ELU()

        elif Activation == 'CELU':
            return nn.CELU()

        elif Activation == 'SELU':
            return nn.SELU()

        elif Activation == 'LRELU':
            return nn.LeakyReLU()

        elif Activation == 'PRELU':
            return nn.LeakyReLU()

