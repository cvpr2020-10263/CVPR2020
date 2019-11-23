'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models.CIFAR10 import VGG
from utils_CIFAR10 import progress_bar
from models.CIFAR10 import resnet18
from models.CIFAR10 import alexnet
global accu
def Model(net, activation, error_rate):
    if net == 'VGG':
        net = VGG.VGG(activation, error_rate)
        return net

    elif net == 'resnet':
        net = resnet18.resnet(activation, error_rate)
        return net

    elif net == 'alexnet':
        net = alexnet.alexnet(activation, error_rate)
        return net

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
parser.add_argument('-resume',type=str, default='yes', help='resume from checkpoint')
parser.add_argument('-act', type=str, default='PRELU1', help='Activation function to use')
parser.add_argument('-error', type=float, default=0.1, help='Error Rate')
parser.add_argument('-net', type=str, default='VGG', help='Model Name')
parser.add_argument('-ECL', type=str, default='no', help='Use Error Correction Layer')



args = parser.parse_args()
bit_error = args.error
ECL = args.ECL
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
#net = VGG.VGG(args.act)

net = Model(args.net, args.act, args.error)

net = net.to(device)
'''
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
'''
if args.resume == 'yes':
    # Load checkpoint.bbs
    print('==> Resuming from checkpoint..')
    #assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    path = './checkpoint/CIFAR10/' + args.net + '/' + args.net + '_' + args.act + '.t7'
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
path = './checkpoint/CIFAR10/' + args.net + '/' + args.net + '_' + args.act + '.t7'


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    global accu
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print ('The accuracy is:  ' + str(acc))
    accu = accu + acc

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, path)
        best_acc = acc


accu = 0
for epoch in range(start_epoch, start_epoch+3):

    #train(epoch)
    test(epoch)



accu = float(accu)
print (accu/3)