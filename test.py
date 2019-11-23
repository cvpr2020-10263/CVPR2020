#test.py
#!/usr/bin/env python3


""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
from torch.autograd import Variable
from conf import settings
from utils import get_network, get_test_dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet', help='net type')
    parser.add_argument('-weights', type=str, required=False, help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-act', type=str, default='RELU', help='Activation function to use')
    parser.add_argument('-error', type=float, default=0.01, help='Error Rate')
    parser.add_argument('-resume', type=str, default='yes', help='resume from checkpoint')

    args = parser.parse_args()

    net = get_network(args, use_gpu=args.gpu)
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    #net.load_state_dict(torch.load(args.weights), args.gpu)
    #print(net)
    #net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    c5 = 0
    c1 = 0
    i = 0
    accu = 0
    tim = 3
    while (i < tim):
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            #print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
            image = Variable(image).cuda()
            label = Variable(label).cuda()
            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()
            #compute top1
            correct_1 += correct[:, :1].sum()

        correct_5 = correct_5 / len(cifar100_test_loader.dataset)
        correct_1 = correct_1 / len(cifar100_test_loader.dataset)

        c5 = c5 + correct_5
        c1 = c1 + correct_1
        correct_1 = 0.0
        correct_5 = 0.0

        i += 1

    c5 = float(c5)
    c1 = float(c1)

    c5 = c5/ tim
    c1 = c1/ tim

    print()
    #print("Top 1 err: ", 1 - c1 / len(cifar100_test_loader.dataset))
    print('Top-1 Accuracy:' + str(c1))
    print()
    print('Top-5 Accuracy:' + str(c5))


    #print("Top 5 err: ", 1 - c5 / len(cifar100_test_loader.dataset))
    #print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))