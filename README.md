# CVPR2020

This repositry corresponds to CVPR 2020 submission with ID number of 10263.

# Project Dependencies
- Pytorch  
- Numpy


# Project Testing 
- Checkpoint folder contains the google drive link for all the pre-trained models.
- After downloading the model, copy the downloaded model into its respective folder
- ErrorFiles folder contains the google drive link for all the Bit-Error Rates Look-up-Table, you can create custom look-up-table from the LUT.py file. Errors are injected into the model using these Look-up-Tables.

# CIFAR100 Testing
- By running the test.py file, you can test the performance on CIFAR100 dataset
- Example code snippet is as follows

python3 test.py -act PRELU2 -error 0.05 -net resnet

# CIFAR10 Testing
- By running the train_CIFAR10.py file, you can test the performance on CIFAR10 dataset
- Example code snippet is as follows

python3 train_CIFAR10.py -act PRELU5 -error 0.02 -net VGG

# MNIST Testing

- By running the train_MNIST.py file, you can test the performance on MNIST dataset
- Example code snippet is as follows

python train_MNIST.py -net lenet -act PRELU3 -resume yes -error 0.1
