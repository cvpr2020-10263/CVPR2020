#!/usr/bin/env bash

echo "RELU"
python train_MNIST.py -net lenet -act RELU -resume yes -error 0.05
python train_MNIST.py -net lenet -act RELU -resume yes -error 0.1
python train_MNIST.py -net lenet -act RELU -resume yes -error 0.15
python train_MNIST.py -net lenet -act RELU -resume yes -error 0.20
python train_MNIST.py -net lenet -act RELU -resume yes -error 0.25
python train_MNIST.py -net lenet -act RELU -resume yes -error 0.30

echo "PRELU1"
python train_MNIST.py -net lenet -act PRELU1 -resume yes -error 0.05
python train_MNIST.py -net lenet -act PRELU1 -resume yes -error 0.1
python train_MNIST.py -net lenet -act PRELU1 -resume yes -error 0.15
python train_MNIST.py -net lenet -act PRELU1 -resume yes -error 0.20
python train_MNIST.py -net lenet -act PRELU1 -resume yes -error 0.25
python train_MNIST.py -net lenet -act PRELU1 -resume yes -error 0.30


echo "PRELU2"
python train_MNIST.py -net lenet -act PRELU2 -resume no -error 0.05
python train_MNIST.py -net lenet -act PRELU2 -resume yes -error 0.1
python train_MNIST.py -net lenet -act PRELU2 -resume yes -error 0.15
python train_MNIST.py -net lenet -act PRELU2 -resume yes -error 0.20
python train_MNIST.py -net lenet -act PRELU2 -resume yes -error 0.25
python train_MNIST.py -net lenet -act PRELU2 -resume no -error 0.30



echo "PRELU3"
python train_MNIST.py -net lenet -act PRELU3 -resume yes -error 0.05
python train_MNIST.py -net lenet -act PRELU3 -resume yes -error 0.1
python train_MNIST.py -net lenet -act PRELU3 -resume yes -error 0.15
python train_MNIST.py -net lenet -act PRELU3 -resume yes -error 0.20
python train_MNIST.py -net lenet -act PRELU3 -resume yes -error 0.25
python train_MNIST.py -net lenet -act PRELU3 -resume yes -error 0.30


echo "PRELU4"
python train_MNIST.py -net lenet -act PRELU4 -resume yes -error 0.05
python train_MNIST.py -net lenet -act PRELU4 -resume yes -error 0.1
python train_MNIST.py -net lenet -act PRELU4 -resume yes -error 0.15
python train_MNIST.py -net lenet -act PRELU4 -resume yes -error 0.20
python train_MNIST.py -net lenet -act PRELU4 -resume yes -error 0.25
python train_MNIST.py -net lenet -act PRELU4 -resume yes -error 0.30


echo "PRELU5"
python train_MNIST.py -net lenet -act PRELU5 -resume yes -error 0.05
python train_MNIST.py -net lenet -act PRELU5 -resume yes -error 0.1
python train_MNIST.py -net lenet -act PRELU5 -resume yes -error 0.15
python train_MNIST.py -net lenet -act PRELU5 -resume yes -error 0.20
python train_MNIST.py -net lenet -act PRELU5 -resume yes -error 0.25
python train_MNIST.py -net lenet -act PRELU5 -resume yes -error 0.30

echo "PRELU6"
python train_MNIST.py -net lenet -act PRELU6 -resume yes -error 0.05
python train_MNIST.py -net lenet -act PRELU6 -resume yes -error 0.1
python train_MNIST.py -net lenet -act PRELU6 -resume yes -error 0.15
python train_MNIST.py -net lenet -act PRELU6 -resume yes -error 0.20
python train_MNIST.py -net lenet -act PRELU6 -resume yes -error 0.25
python train_MNIST.py -net lenet -act PRELU6 -resume yes -error 0.30

