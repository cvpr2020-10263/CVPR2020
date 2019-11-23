#!/usr/bin/env bash

echo 'Resnet'
#CUDA_VISIBLE_DEVICES=2 python3 train_CIFAR10.py -act PRELU4 -error 0.09 -net resnet
#CUDA_VISIBLE_DEVICES=2 python3 train_CIFAR10.py -act PRELU5 -error 0.09 -net resnet

echo 'alexnet'
#CUDA_VISIBLE_DEVICES=2 python3 train_CIFAR10.py -act PRELU3 -error 0.09 -net alexnet
#CUDA_VISIBLE_DEVICES=2 python3 train_CIFAR10.py -act PRELU4 -error 0.09 -net alexnet
#CUDA_VISIBLE_DEVICES=2 python3 train_CIFAR10.py -act PRELU5 -error 0.09 -net alexnet


echo 'VGG'
CUDA_VISIBLE_DEVICES=2 python3 train_CIFAR10.py -act PRELU3 -error 0.09 -net VGG
CUDA_VISIBLE_DEVICES=2 python3 train_CIFAR10.py -act PRELU4 -error 0.09 -net VGG
CUDA_VISIBLE_DEVICES=2 python3 train_CIFAR10.py -act PRELU5 -error 0.09 -net VGG


