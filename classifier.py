import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from IPython.display import clear_output

import torch
from torch.utils.data import Dataset
from PIL import Image


##############################################
# Residual block with two convolution layers.
##############################################

    
class Network(nn.Module):

    def __init__(self, input_shape):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.conv2_drop = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(5120, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 1)
        # self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 20*16*16)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        
        return (x)
     
    
class DynamicNetwork(nn.Module):
    def __init__(self, input_shape, conv_layers, kernel_size, out_channel_ratio, FC_layers):
        super(DynamicNetwork, self).__init__()

        in_channels, height, samples, width = input_shape
       
        model = []
        
        for i in range(conv_layers):
            out_channels = in_channels * out_channel_ratio
            model += [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2)
                ]
            in_channels = out_channels     

        out_size = (in_channels*height*width)/(2**(i+1))**2 #MAX POOL REDUCES H AND W BY 2 EACH TIME. THIS FACTORS OUT AS SQUARED(1/2*ITER)

        model += [
            nn.Flatten()
        ]
        
        in_channels = out_size
        FC_layer_diff = in_channels//FC_layers
        
        for j in range((FC_layers-1)):
            out_channels = in_channels - (FC_layer_diff)
            model += [
                nn.Linear(int(in_channels), int(out_channels)),
                nn.ReLU()
            ]
            # print(in_channels, out_channels)
            in_channels = out_channels   
        
        model += [
            nn.LazyLinear(1),
        ]
            
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class CNN3D(nn.Module):
    def __init__(self, input_shape, conv_layers, kernel_size, out_channel_ratio, FC_layers):
        super(CNN3D, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(2**1*64, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.LeakyReLU()
        self.relu = nn.LeakyReLU()
        # self.batch=nn.BatchNorm1d(128)
        # self.drop=nn.Dropout(p=0.15)        
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        
        return out