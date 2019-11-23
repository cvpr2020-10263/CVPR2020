#!/usr/bin/env bash
'''
echo 'CELU'
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act CELU -net VGG -error 0.02
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act CELU -net VGG -error 0.04
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act CELU -net VGG -error 0.06
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act CELU -net VGG -error 0.08
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act CELU -net VGG -error 0.1
'''

echo 'ELU'
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act ELU -net VGG -error 0.02
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act ELU -net VGG -error 0.04
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act ELU -net VGG -error 0.06
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act ELU -net VGG -error 0.08
CUDA_VISIBLE_DEVICES=1 python3 train_CIFAR10.py -act ELU -net VGG -error 0.1


echo 'LRELU'
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act LRELU -net VGG -error 0.02
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act LRELU -net VGG -error 0.04
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act LRELU -net VGG -error 0.06
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act LRELU -net VGG -error 0.08
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act LRELU -net VGG -error 0.1

echo 'PRELU'
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU -net VGG -error 0.02
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU -net VGG -error 0.04
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU -net VGG -error 0.06
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU -net VGG -error 0.08
CUDA_VISIBLE_DEVICES=0 python3 train_CIFAR10.py -act PRELU -net VGG -error 0.1
echo 'RELU6'
CUDA_VISIBLE_DEVICES=2 python3 train_CIFAR10.py -act RELU6 -net VGG -error 0.02
CUDA_VISIBLE_DEVICES=2 python3 train_CIFAR10.py -act RELU6 -net VGG -error 0.04
CUDA_VISIBLE_DEVICES=2 python3 train_CIFAR10.py -act RELU6 -net VGG -error 0.06
CUDA_VISIBLE_DEVICES=2 python3 train_CIFAR10.py -act RELU6 -net VGG -error 0.08
CUDA_VISIBLE_DEVICES=2 python3 train_CIFAR10.py -act RELU6 -net VGG -error 0.1

echo 'RRELU'
CUDA_VISIBLE_DEVICES=2 python3 train_CIFAR10.py -act RRELU -net VGG -error 0.02
CUDA_VISIBLE_DEVICES=2 python3 train_CIFAR10.py -act RRELU -net VGG -error 0.04
CUDA_VISIBLE_DEVICES=2 python3 train_CIFAR10.py -act RRELU -net VGG -error 0.06
CUDA_VISIBLE_DEVICES=2 python3 train_CIFAR10.py -act RRELU -net VGG -error 0.08
CUDA_VISIBLE_DEVICES=2 python3 train_CIFAR10.py -act RRELU -net VGG -error 0.1

echo 'SELU'
CUDA_VISIBLE_DEVICES=3 python3 train_CIFAR10.py -act SELU -net VGG -error 0.02
CUDA_VISIBLE_DEVICES=3 python3 train_CIFAR10.py -act SELU -net VGG -error 0.04
CUDA_VISIBLE_DEVICES=3 python3 train_CIFAR10.py -act SELU -net VGG -error 0.06
CUDA_VISIBLE_DEVICES=3 python3 train_CIFAR10.py -act SELU -net VGG -error 0.08
CUDA_VISIBLE_DEVICES=3 python3 train_CIFAR10.py -act SELU -net VGG -error 0.1
