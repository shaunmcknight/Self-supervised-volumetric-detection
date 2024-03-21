import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from IPython.display import clear_output

import torch
from torch.utils.data import Dataset

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.c1=32
        self.c2=64
        self.c3=128
       
        self.k3_block1 = self._conv_layer_set(1, self.c1, 3)
        self.k3_block2 = self._conv_layer_set(self.c1, self.c2, 3)
        self.k3_block3 = self._conv_layer_set(self.c2, self.c3, 3)
        # self.k3_block4 = self._conv_layer_set(8, 16, 3)
        
        self.k7_block1 = self._conv_layer_set(1, self.c1, 7)
        self.k7_block2 = self._conv_layer_set(self.c1, self.c2, 7)
        self.k7_block3 = self._conv_layer_set(self.c2, self.c3, 7)
        
        self.k15_block1 = self._conv_layer_set(1, self.c1, 15)
        self.k15_block2 = self._conv_layer_set(self.c1, self.c2, 15)
        self.k15_block3 = self._conv_layer_set(self.c2, self.c3, 15)
        
        # self.conv_reduction = nn.Conv1d(8,8,kernel_size=8, stride=8)
        self.gap = nn.AvgPool1d(8)
        self.linear_hidden = nn.Linear(self.c3*3, 256)
        self.linear = nn.Linear(256, 1)
        
        
        
        self.k3_block1_std = self._conv_layer_set(1, self.c1, 3)
        self.k3_block2_std = self._conv_layer_set(self.c1, self.c2, 3)
        self.k3_block3_std = self._conv_layer_set(self.c2, self.c3, 3)
        # self.k3_block4 = self._conv_layer_set(8, 16, 3)
        
        self.k7_block1_std = self._conv_layer_set(1, self.c1, 7)
        self.k7_block2_std = self._conv_layer_set(self.c1, self.c2, 7)
        self.k7_block3_std = self._conv_layer_set(self.c2, self.c3, 7)
        
        self.k15_block1_std = self._conv_layer_set(1, self.c1, 15)
        self.k15_block2_std = self._conv_layer_set(self.c1, self.c2, 15)
        self.k15_block3_std = self._conv_layer_set(self.c2, self.c3, 15)
        
        # self.conv_reduction = nn.Conv1d(8,8,kernel_size=8, stride=8)
        self.gap_std = nn.AvgPool1d(8)
        self.linear_hidden_std = nn.Linear(self.c3*3, 256)
        self.linear_std = nn.Linear(256, 1)
        
    def _conv_layer_set(self, in_c, out_c, kernel_size):
        conv_layer = nn.Sequential(
            #feature extraction
            nn.Conv1d(in_c, out_c, stride=1, kernel_size=kernel_size, padding='same'),
            nn.LeakyReLU(),
            #downsample
            nn.Conv1d(out_c, out_c, stride=2, kernel_size=2),
            nn.LeakyReLU(),
            )
        return conv_layer
    

    def forward(self, x):
        # Set 1
        out_3 = self.k3_block3(self.k3_block2(self.k3_block1(x)))
        out_7 = self.k7_block3(self.k7_block2(self.k7_block1(x)))
        out_15 = self.k15_block3(self.k15_block2(self.k15_block1(x)))
        
        out=torch.cat((out_3, out_7, out_15), dim=1)
        # out=out.flatten()
        out = self.gap(out)
        # print(out.size())
        mean = self.linear(self.linear_hidden(out.squeeze().squeeze()))
        
        # Set 1
        out_3_std = self.k3_block3_std(self.k3_block2_std(self.k3_block1_std(x)))
        out_7_std = self.k7_block3_std(self.k7_block2_std(self.k7_block1_std(x)))
        out_15_std = self.k15_block3_std(self.k15_block2_std(self.k15_block1_std(x)))
        
        out_std=torch.cat((out_3_std, out_7_std, out_15_std), dim=1)
        # out=out.flatten()
        out_std = self.gap(out_std)
        # print(out.size())
        std = self.linear(self.linear_hidden(out_std.squeeze().squeeze()))
        return torch.abs(mean), torch.abs(std)