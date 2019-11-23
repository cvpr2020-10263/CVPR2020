''''''
from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models.MNIST import LeNet
from models.MNIST import BNN
import os
import torch.nn as nn
from torch.autograd import Variable
global accu
best_accu = 0



def LoadModel(args, model):
    if args.resume == 'yes':
        PATH = './checkpoint/MNIST/' + args.net + '/' + args.net + '_' + args.act + '.pt'
        PATH = os.path.abspath(PATH)
        model.load_state_dict(torch.load(PATH))
        return model.eval()

    else:
        return model


def Model(net, args):
    if net == 'lenet':
        net = LeNet.LeNet(args)
        return net

    elif net == 'bnn':
        net = BNN.BNN(args)
        return net

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        criterion = nn.CrossEntropyLoss()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        # loss = loss.type(float)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def LossFunction(args):

    if args.net == 'bnn':
        criterion = nn.CrossEntropyLoss()
        return criterion

    elif args.net == 'lenet':
        criterion = nn.NLLLoss()
        return criterion

def Prediction(args, output):
    if args.net == 'bnn':
        pred = output.data.max(1, keepdim=True)[1]
        return pred

    elif args.net == 'lenet':
        pred = output.argmax(dim=1, keepdim=True)
        return pred

def Optimizer(args, model):
    if args.net == 'bnn':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        return optimizer

    elif args.net == 'lenet':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        return optimizer


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            criterion = LossFunction(args)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = Prediction(args, output)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    accuracy = 100. * correct / len(test_loader.dataset)
    global best_accu

    if best_accu < accuracy:
        best_accu = accuracy
        print ('saving....')
        PATH = './checkpoint/MNIST/' + args.net + '/' + args.net + '_' + args.act + '.pt'
        torch.save(model.state_dict(), PATH)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-net', type=str, default='lenet', help='net type')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('-act', type=str, default='PRELU2', help='Activation function to use')
    parser.add_argument('--eps', type=float, default=1e-5, metavar='LR', help='learning rate,default=1e-5')
    parser.add_argument('-error', type=float, default=0.1, help='Error Rate')
    parser.add_argument('-resume', type=str, default='yes', help='Resume the training')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()


    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)#, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True)#, **kwargs)

    model = Model(args.net, args)
    model = model.to(device)
    model = LoadModel(args, model)

    optimizer = Optimizer(args, model)

    ''' 
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
    '''

    i = 0
    accu = 0
    while (i < 3):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))

        i = i + 1
        a = 100 * correct / total
        accu = accu + a

    accu = float(accu)
    accu = float(accu/3)
    print (accu)


if __name__ == '__main__':
    main()
    # print (max(max1))
