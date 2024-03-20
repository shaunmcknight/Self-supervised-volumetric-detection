import os
import numpy as np

import torchvision.transforms as transforms
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import glob
import random
from torch.utils.data import Dataset
from PIL import Image


########################################################
# Methods for 3D DataLoader
#
# 1's = defect
# 0's = no defect
#
#
########################################################

class CleanDataset(Dataset):
    def __init__(self, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        
        self.id007=np.load(r'C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/SEARCH NDE/Composites/Exp Data/Spirit Cell/New scans/ID007/ID007_hilbert_raster.npy')
        self.id007=self.preprocess_data(self.id007)
        self.id007=self.id007[:260,:501,:]+1 # shift data
        
        plt.imshow(self.id007[:,:,10], aspect='auto')
        plt.show()
        print('~~~~ Synthetic dataloader INFO ~~~~~')
        print('Number points train ~ ', 'Max/Min ', np.amax(self.id007), np.amin(self.id007))
        #length 500 time series
        
    def preprocess_data(self, data):
        # done after applying hilbert transform
        shifted_ut = np.zeros(data.shape)
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                # a_scan = data[i,:,j]
                # peak = np.argmax(a_scan)
                # shifted_a_scan = a_scan[peak::]
                # shifted_ut[i,0:(data.shape[1]-peak),j] = shifted_a_scan
                # shifted_ut[i,:,j] = shifted_ut[i,:,j]/np.nanmax(shifted_ut[i,:,j])
                shifted_ut[i,:,j] = data[i,:,j]/np.nanmax(data[i,:,j])
        return shifted_ut
    
    def random_erase(self, time_series, p=0.5, erase_value=0):
        """
        Randomly erase segments of a 1D time series.
    
        Parameters:
        - time_series (numpy array): The input 1D time series.
        - p (float): Probability of applying random erasing to each segment.
        - erase_value: The value used to replace the erased segment.
    
        Returns:
        - erased_time_series (numpy array): The time series with random erasing applied.
        """
    
        erased_time_series = time_series.copy()
    
        if np.random.rand() < p:
            erase_length = np.random.randint(1, 9)  # Random erase length from 1 to 8 inclusive
            start_idx = np.random.randint(0, len(time_series) - erase_length)  # Random start index
            erased_time_series[start_idx:start_idx + erase_length] = erase_value
                
        return erased_time_series

        
    def __getitem__(self, index):       
        index=np.unravel_index(index, (4,100,244))

        base_data=self.id007[index[0]*65:index[0]*65+65,index[1]*5,index[2]]
        
        if self.transform != None:
            if np.random.randint(0,2)==1:
                base_data = np.flip(base_data)
        
        data = base_data[:64]
        # data = data.unsqueeze(0)
        
        if self.transform != None:
            data = self.random_erase(data, p=0.5, erase_value=0)
            
        label = base_data[64]      
        return torch.from_numpy(np.copy(data)).unsqueeze(0), torch.from_numpy(np.copy(label)).unsqueeze(0)
        #At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().) 

    def __len__(self):
        elements=self.id007.shape[2]
        time_step=round(self.id007.shape[1]/5)
        series_step=65
        total_series=self.id007.shape[0]/series_step
        length=int(total_series*time_step*elements)
        print('length should be: 97600 but is: ', length) #may need to do -1 for indexing (probably not)
        return length
    
    

class TestDataset(Dataset):
    def __init__(self, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        
        self.id007=np.load(r'C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/SEARCH NDE/Composites/Exp Data/Spirit Cell/New scans/ID007/ID007_hilbert_raster.npy')
        self.id007=self.preprocess_data(self.id007)
        self.id007=self.id007[:260,:501,:]+1 # shift data
            
        print('~~~~ Synthetic dataloader INFO ~~~~~')
        print('Number points train ~ ', 'Max/Min ', np.amax(self.id007), np.amin(self.id007))
        #length 500 time series
        
    def preprocess_data(self, data):
        # done after applying hilbert transform
        shifted_ut = np.zeros(data.shape)
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                # a_scan = data[i,:,j]
                # peak = np.argmax(a_scan)
                # shifted_a_scan = a_scan[peak::]
                # shifted_ut[i,0:(data.shape[1]-peak),j] = shifted_a_scan
                # shifted_ut[i,:,j] = shifted_ut[i,:,j]/np.nanmax(shifted_ut[i,:,j])
                shifted_ut[i,:,j] = data[i,:,j]/np.nanmax(data[i,:,j])
        return shifted_ut
    
    def random_erase(self, time_series, p=0.5, erase_value=0):
        """
        Randomly erase segments of a 1D time series.
    
        Parameters:
        - time_series (numpy array): The input 1D time series.
        - p (float): Probability of applying random erasing to each segment.
        - erase_value: The value used to replace the erased segment.
    
        Returns:
        - erased_time_series (numpy array): The time series with random erasing applied.
        """
    
        erased_time_series = time_series.copy()
    
        if np.random.rand() < p:
            erase_length = np.random.randint(1, 9)  # Random erase length from 1 to 8 inclusive
            start_idx = np.random.randint(0, len(time_series) - erase_length)  # Random start index
            erased_time_series[start_idx:start_idx + erase_length] = erase_value
                
        return erased_time_series

        
    def __getitem__(self, index):       
        index=np.unravel_index(index, (4,100,244))

        base_data=self.id007[index[0]*65:index[0]*65+65,index[1]*5,index[2]]
        
        # if self.transform != None:
        #     if np.random.randint(0,2)==1:
        #         base_data = np.flip(base_data)
        
        data = base_data[:64]
        # data = data.unsqueeze(0)
        
        # if self.transform != None:
        #     data = self.random_erase(data, p=0.5, erase_value=0)
            
        label = base_data[64]      
        return torch.from_numpy(np.copy(data)).unsqueeze(0), torch.from_numpy(np.copy(label)).unsqueeze(0)
        #At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().) 

    def __len__(self):
        elements=self.id007.shape[2]
        time_step=round(self.id007.shape[1]/5)
        series_step=65
        total_series=self.id007.shape[0]/series_step
        length=int(total_series*time_step*elements)
        print('length should be: 97600 but is: ', length) #may need to do -1 for indexing (probably not)
        return length
