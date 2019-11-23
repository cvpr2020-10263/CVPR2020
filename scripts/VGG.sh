#!/usr/bin/env bash

:'
echo "VGG"

echo "RELU"
CUDA_VISIBLE_DEVICES=3  python3 test.py -act RELU -error 0.01 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act RELU -error 0.02 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act RELU -error 0.03 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act RELU -error 0.04 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act RELU -error 0.05 -net vgg16

echo "PRELU2"
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU2 -error 0.01 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU2 -error 0.02 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU2 -error 0.03 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU2 -error 0.04 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU2 -error 0.05 -net vgg16

echo "PRELU5"
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU5 -error 0.01 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU5 -error 0.02 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU5 -error 0.03 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU5 -error 0.04 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU5 -error 0.05 -net vgg16

echo "PRELU3"
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU3 -error 0.01 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU3 -error 0.02 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU3 -error 0.03 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU3 -error 0.04 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU3 -error 0.05 -net vgg16

echo "PRELU4"
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU4 -error 0.01 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU4 -error 0.02 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU4 -error 0.03 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU4 -error 0.04 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU4 -error 0.05 -net vgg16

echo "PRELU1"
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU1 -error 0.01 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU1 -error 0.02 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU1 -error 0.03 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU1 -error 0.04 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU1 -error 0.05 -net vgg16



echo "PRELU6"
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU6 -error 0.01 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU6 -error 0.02 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU6 -error 0.03 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU6 -error 0.04 -net vgg16
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU6 -error 0.05 -net vgg16
'
echo "VGG RELU"
CUDA_VISIBLE_DEVICES=3  python3 test.py -act RELU -error 0.05 -net vgg16
echo "VGG PRELU2"
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU2 -error 0.05 -net vgg16
echo "VGG PRELU5"
CUDA_VISIBLE_DEVICES=3  python3 test.py -act PRELU5 -error 0.05 -net vgg16

echo "Resnet RELU"
CUDA_VISIBLE_DEVICES=1  python3 test.py -act RELU -error 0.05 -net resnet

echo "Resnet RELU5"

CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU5 -error 0.04 -net resnet
echo "Resnet RELU5"

CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU6 -error 0.04 -net resnet

echo "Alexnet RLU"

CUDA_VISIBLE_DEVICES=1  python3 test.py -act RELU -error 0.04 -net resnet
echo "Alexnet RLU3"

CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU3 -error 0.04 -net resnet
echo "Alexnet RLU5"

CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU5 -error 0.04 -net resnet

