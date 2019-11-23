#!/usr/bin/env bash

:'
echo "Resnet PRELU5"
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU5 -error 0.01 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU5 -error 0.02 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU5 -error 0.03 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU5 -error 0.04 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU5 -error 0.05 -net resnet

echo "Resnet PRELU6"
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU6 -error 0.01 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU6 -error 0.02 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU6 -error 0.03 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU6 -error 0.04 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU6 -error 0.05 -net resnet



echo "Resnet PRELU1"
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU1 -error 0.01 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU1 -error 0.02 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU1 -error 0.03 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU1 -error 0.04 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU1 -error 0.05 -net resnet

echo "Resnet PRELU2"
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU2 -error 0.01 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU2 -error 0.02 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU2 -error 0.03 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU2 -error 0.04 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU2 -error 0.05 -net resnet

echo "Resnet PRELU3"
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU3 -error 0.01 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU3 -error 0.02 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU3 -error 0.03 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU3 -error 0.04 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU3 -error 0.05 -net resnet

echo "Resnet PRELU4"
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU4 -error 0.01 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU4 -error 0.02 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU4 -error 0.03 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU4 -error 0.04 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act PRELU4 -error 0.05 -net resnet
'
echo "Resnet RELU"
CUDA_VISIBLE_DEVICES=1  python3 test.py -act RELU -error 0.01 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act RELU -error 0.02 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act RELU -error 0.03 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act RELU -error 0.04 -net resnet
CUDA_VISIBLE_DEVICES=1  python3 test.py -act RELU -error 0.05 -net resnet