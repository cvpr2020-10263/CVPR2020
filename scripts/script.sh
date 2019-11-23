#!/usr/bin/env bash

echo "ERROR 0.05 VGG"

echo "ReLU"
CUDA_VISIBLE_DEVICES=3 python3 train.py -act RELU -error 0.05 -net vgg16
echo "PReLU1"
CUDA_VISIBLE_DEVICES=3 python3 test.py -act PRELU1 -error 0.05 -net vgg16
echo "PReLU2"
CUDA_VISIBLE_DEVICES=3 python3 test.py -act PRELU2 -error 0.05 -net vgg16





eecho "ERROR 0.05 Resnet"

echo "ReLU"
CUDA_VISIBLE_DEVICES=3 python3 test.py -act RELU -error 0.05 -net resnet
echo "PReLU1"
CUDA_VISIBLE_DEVICES=3 python3 train.py -act PRELU1 -error 0.05 -net resnet
echo "PReLU2"
CUDA_VISIBLE_DEVICES=3 python3 test.py -act PRELU2 -error 0.05 -net resnet




echo "ERROR 0.05 Alexnet"
echo "ReLU"
CUDA_VISIBLE_DEVICES=3 python3 test.py -act RELU -error 0.05 -net alexnet
echo "PReLU1"
CUDA_VISIBLE_DEVICES=3 python3 test.py -act PRELU1 -error 0.05 -net alexnet
echo "PReLU2"
CUDA_VISIBLE_DEVICES=3 python3 test.py -act PRELU2 -error 0.05 -net alexnet



