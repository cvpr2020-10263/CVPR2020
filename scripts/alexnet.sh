#!/usr/bin/env bash
:'
echo "Alexnet RELU"

CUDA_VISIBLE_DEVICES=2 python3 test.py -act RELU -error 0.01 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act RELU -error 0.02 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act RELU -error 0.03 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act RELU -error 0.04 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act RELU -error 0.05 -net alexnet


echo "Alexnet PRELU2"

CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU2 -error 0.01 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU2 -error 0.02 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU2 -error 0.03 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU2 -error 0.04 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU2 -error 0.05 -net alexnet


echo "Alexnet PRELU5"

CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU5 -error 0.01 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU5 -error 0.02 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU5 -error 0.03 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU5 -error 0.04 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU5 -error 0.05 -net alexnet

'
echo "Alexnet PRELU3"

CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU3 -error 0.01 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU3 -error 0.02 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU3 -error 0.03 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU3 -error 0.04 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU3 -error 0.05 -net alexnet

:'

echo "Alexnet PRELU4"

CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU4 -error 0.01 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU4 -error 0.02 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU4 -error 0.03 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU4 -error 0.04 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU4 -error 0.05 -net alexnet


echo "Alexnet PRELU6"

CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU6 -error 0.01 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU6 -error 0.02 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU6 -error 0.03 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU6 -error 0.04 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU6 -error 0.05 -net alexnet

echo "Alexnet PRELU1"

CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU1 -error 0.01 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU1 -error 0.02 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU1 -error 0.03 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU1 -error 0.04 -net alexnet
CUDA_VISIBLE_DEVICES=2 python3 test.py -act PRELU1 -error 0.05 -net alexnet'