#!/usr/bin/env bash


echo "Resnet RELU"

echo "ERROR 0.02"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act RELU -error 0.02 -net alexnet
echo "ERROR 0.04"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act RELU -error 0.04 -net alexnet
echo "ERROR 0.06"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act RELU -error 0.06 -net alexnet


echo "Alexnet PRELU2"

echo "ERROR 0.02"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU2 -error 0.02 -net alexnet
echo "ERROR 0.04"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU2 -error 0.04 -net alexnet
echo "ERROR 0.06"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU2 -error 0.06 -net alexnet


echo "Alexnet PRELU5"

echo "ERROR 0.02"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU5 -error 0.02 -net alexnet
echo "ERROR 0.04"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU5 -error 0.04 -net alexnet
echo "ERROR 0.06"
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act PRELU5 -error 0.06 -net alexnet
