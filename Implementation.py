# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 14:39:10 2021

@author: alexi
"""
### Libraries ###
import torch 
import torch.nn as nn
from sklearn.datasets import fetch_openml
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt

### Creating objects ###
class ColorTransformer(nn.Module):
    def __init__(self):
        super(ColorTransformer, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 7, stride = 1, padding = 0)
        #self.Conv1.weight.data.fill_(0)
        self.MaxPool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.Relu1 = nn.ReLU()
        self.DropOut1 = nn.Dropout2d(p=0.5)
        self.Conv2 = nn.Conv2d(32, 32, kernel_size = 5, stride = 1, padding = 0)
        #self.Conv2.weight.data.fill_(0)
        self.MaxPool2 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.Relu2 = nn.ReLU()
        self.DropOut2 = nn.Dropout2d(p=0.5)
        self.Lin1 = nn.Linear(512, 100)
        #self.Lin1.weight.data.fill_(0) # only last layer need to have weights set to 0
        self.Relu3 = nn.ReLU()
        self.Lin2 = nn.Linear(100, 27)
        self.Lin2.weight.data.fill_(0)
        Bias = [1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0, 0]
        self.Lin2.bias = nn.Parameter(torch.tensor(Bias, dtype = torch.float64))
        
    def forward(self, x):
        x = self.Conv1(x)
        x = self.MaxPool1(x)
        x = self.Relu1(x)
        x = self.DropOut1(x)
        x = self.Conv2(x)
        x = self.MaxPool2(x)
        x = self.Relu2(x)
        x = self.DropOut2(x)
        x = x.view(-1, 512)
        x = self.Lin1(x)
        x = self.Relu3(x)
        x = self.Lin2(x)
        return x

class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 7, stride = 1, padding = 0) # output size : 32 x 26 x 26
        #self.Conv1.weight.data.fill_(0)
        self.MaxPool1 = nn.MaxPool2d(kernel_size = 2, stride = 2) # output size : 32 x 13 x 13
        self.Relu1 = nn.ReLU()
        self.DropOut1 = nn.Dropout2d(p=0.5)
        self.Conv2 = nn.Conv2d(32, 32, kernel_size = 5, stride = 1, padding = 0) # output size : 32 x 9 x 9
        #self.Conv2.weight.data.fill_(0)
        self.MaxPool2 = nn.MaxPool2d(kernel_size = 3, stride = 2) # output size : 32 x 4 x 4
        self.Relu2 = nn.ReLU()
        self.DropOut2 = nn.Dropout2d(p=0.5)
        self.Lin1 = nn.Linear(512, 100)
        #self.Lin1.weight.data.fill_(0)
        self.Relu3 = nn.ReLU()
        self.Lin2 = nn.Linear(100, 6)
        self.Lin2.weight.data.fill_(0)
        Bias = [1, 0, 0, 0, 1, 0]
        self.Lin2.bias = nn.Parameter(torch.tensor(Bias, dtype = torch.float64))
        
    def forward(self, x):
        x = self.Conv1(x)
        x = self.MaxPool1(x)
        x = self.Relu1(x)
        x = self.DropOut1(x)
        x = self.Conv2(x)
        x = self.MaxPool2(x)
        x = self.Relu2(x)
        x = self.DropOut2(x)
        x = x.view(-1, 512)
        x = self.Lin1(x)
        x = self.Relu3(x)
        x = self.Lin2(x)
        
        return x
    
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.stn = SpatialTransformer()
        self.ctn = ColorTransformer()
        
    def DoStn(self,x):
        # get matrix for space transformation
        theta = self.stn(x)
        # reshape to the correct dim
        theta = theta.view(-1,2,3)
        
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        
        return x
    
    def DoCtn(self,x):
        
        return x
    
    def forward(self, x):
        
        x = self.DoCtn(x)
        x = self.DoStn(x)
        
        return x
    
class FeatureExtractor_MNIST_SVHN(nn.Module):
    def __init__(self):
        super(FeatureExtractor_MNIST_SVHN, self).__init__()
        self.InstanceNorm = nn.InstanceNorm2d(3) # do not change input size
        self.Conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 5, stride = 1, padding = 2) # output size : 64 x 32 x 32
        self.BatchNorm1 = nn.BatchNorm2d(64) # same shape as input
        self.Activation1 = nn.ReLU()
        self.MaxPool1 = nn.MaxPool2d(kernel_size = 3, stride = 2) # outpur size : 64 x 15 x 15
        self.DropOut1 = nn.Dropout2d(p = 0.5)
        
        self.Conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 1, padding = 2) # output size : 64 x 15 x 15
        self.BatchNorm2 = nn.BatchNorm2d(64) # same shape as input
        self.Activation2 = nn.ReLU()
        self.MaxPool2 = nn.MaxPool2d(kernel_size = 3, stride = 2) # outpur size : 64 x 7 x 7
        self.DropOut2 = nn.Dropout2d(p = 0.5)
        
        self.Conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 1, padding = 2) # output size : 64 x 7 x 7
        self.BatchNorm3 = nn.BatchNorm2d(128) # same shape as input
        self.Activation3 = nn.ReLU()
        self.DropOut3 = nn.Dropout2d(p = 0.5)
        
        self.Lin1 = nn.Linear(3136,3072)
        self.BatchNorm4 = nn.BatchNorm2d(3072) # same shape as input
        self.Activation4 = nn.ReLU()
        self.DropOut4 = nn.Dropout2d(p = 0.5)
        
        
    def forward(self, x):
        x = self.InstanceNorm(x)
        x = self.Conv1(x)
        x = self.BatchNorm1(x)
        x = self.Activation1(x)
        x = self.MaxPool1(x)
        x = self.DropOut1(x)
       
        x = self.Conv2(x)
        x = self.BatchNorm2(x)
        x = self.Activation2(x)
        x = self.MaxPool2(x)
        x = self.DropOut2(x)
        
        x = self.Conv3(x)
        x = self.BatchNorm3(x)
        x = self.Activation3(x)
        x = self.DropOut3(x)


        x = x.view(-1, 3136)
        x = self.Lin1(x)
        x = self.BatchNorm4(x)
        x = self.Activation4(x)
        x = self.DropOut4(x)

        return x
    
class LabelClassifier_MNIST_SVHN(nn.Module):
    def __init__(self):
        super(LabelClassifier_MNIST_SVHN, self).__init__()
        self.Lin1 = nn.Linear(3072,2048)
        self.BatchNorm = nn.BatchNorm1d(2048)
        self.Activation = nn.ReLU()
        self.Lin2 = nn.Linear(2048,10)
        
    def forward(self, x):
        x = self.Lin1(x)
        x = self.BatchNorm(x)
        x = self.Activation(x)
        x = self.Lin2(x)
        
        return nn.Softmax(x) # all classifier end by a softmax cf article

### Main Program ###
### Parameters
learning_rate = 1e-6
Pad = nn.ZeroPad2d(padding = 2)

### Using GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



### Importing Data
mnist = fetch_openml('mnist_784',version=1, cache=True)
### Spliting dataset
X, y = mnist["data"] , mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train [shuffle_index]

# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h, n_out, batch_size = 10, 5, 1, 10

#ColorTransformer.to(device)

Model = ColorTransformer()

Model = Model.double()

Test = [np.reshape(X_train[0], (28,28)), np.reshape(X_train[1], (28,28))]
Test[0] = torch.from_numpy(Test[0]).unsqueeze_(0)
Test[1] = torch.from_numpy(Test[1]).unsqueeze_(0)
Test[0] = Test[0].repeat(3,1,1)
Test[1] = Test[1].repeat(3,1,1)
Test[0] = Pad(Test[0])
Test[1] = Pad(Test[1])
Test = torch.stack((Test[0], Test[1]))

A = Model(Test.double())
print(A)
print(len(A))
print(A[0].size())
#ColorTransformer.train(X_train, y_train)

#ColorTransformer.t
