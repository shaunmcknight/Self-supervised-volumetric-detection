import numpy as np

import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


class CleanDataset(Dataset):
    def __init__(self, data, transforms=['flip', 'erase', 'noise'], label=None, stride=1):
        assert [np.isnan(data).any() for data in data], f"Input data contains NaN values"
        self.transform = transforms
        self.stride=stride
        self.data = [np.reshape(data, (data.shape[0], -1)) for data in data]
        self.data = np.concatenate(self.data, axis =1)
        # self.data = self.data+1 # shift data # removed as calculating distributions
        print(f'\n{label} data info: ')
        print('Data shape: ', self.data.shape, '\nNumber points train: ', self.__len__(),'\nMax/Min ', np.amax(self.data), np.amin(self.data))
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
    
    def random_erase(self, time_series, p=0.5, erase_value=0, erase_length=8):
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
            erase_length = np.random.randint(1, erase_length+1)  # Random erase length from 1 to 8 inclusive
            start_idx = np.random.randint(0, len(time_series) - erase_length)  # Random start index
            erased_time_series[start_idx:start_idx + erase_length] = erase_value
                
        return erased_time_series
    
    def add_noise(self, time_series, p=0.5, noise_mean=0, noise_std=0.2):
        """
        Add noise to a 1D time series sampled from a normal distribution.

        Parameters:
        - time_series (numpy array): The input 1D time series.
        - p (float): Probability of adding noise to each segment.
        - noise_mean (float): Mean of the normal distribution.
        - noise_std (float): Standard deviation of the normal distribution.

        Returns:
        - noisy_time_series (numpy array): The time series with noise added.
        """

        noisy_time_series = time_series.copy()
        noise_std = noise_std * np.nanmedian(time_series)

        if np.random.rand() < p:
            noise = np.random.normal(noise_mean, noise_std, noisy_time_series.shape)
            noisy_time_series+=noise

        return noisy_time_series

        
    def __getitem__(self, index):  
        num_windows_per_sequence = ((self.data.shape[0] - 65) // self.stride) + 1

        sequence_index = index // num_windows_per_sequence
        window_index = index % num_windows_per_sequence
        
        start_index = window_index * self.stride
        
        base_data = self.data[start_index:start_index+65, sequence_index]
        
        if 'flip' in str(self.transform):
            if np.random.rand() < 0.5:
                base_data = np.flip(base_data)
            
        data = base_data[:64]
        
        if 'erase' in str(self.transform):
            data = self.random_erase(data, p=0.5, erase_value=0)
        if 'noise' in str(self.transform):
            data = self.add_noise(data, p=0.5, noise_mean=0, noise_std=0.2)
            
        label = base_data[64]      
        return torch.from_numpy(np.copy(data)).unsqueeze(0), torch.from_numpy(np.copy(label)).unsqueeze(0)
        #At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().) 

    def __len__(self):
        sequence_length=self.data.shape[0] #overlapping series windows
        window_length=65
        num_windows = ((sequence_length - window_length) // self.stride) + 1
        total_windows=self.data.shape[1]*num_windows
        return total_windows
    
    
