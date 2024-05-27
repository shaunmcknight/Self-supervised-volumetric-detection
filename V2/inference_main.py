import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
import torch.nn as nn
from classifierv2 import CNN_shared as model
import skimage
from skimage.measure import label, regionprops
from torchinfo import summary

import time
from math import pi as PI
import os.path
from tqdm import tqdm

class InferenceDataset():
    def __init__(self, path, pad='edge'):
        # self.data = np.load(path)[:,0:,:]
        self.data = np.load(path)[10:300,0:550:1,30:290]
        # self.data = np.load(path)[:260,0:1000:10,:]
        self.original_data = self.data
        self.pad = pad 
        
        self.visualise_data()

        nan_indices = np.argwhere(np.isnan(self.data))
        if nan_indices.size > 0:
            print("NaN values are located at:")
            for index in nan_indices:
                print(index)
                raise SystemExit(0)
        
        # check what data axis is divisibble for 61 and move to first axis
        divisible_axis = [i for i, dim in enumerate(self.data.shape) if dim % 61 == 0]
        # self.data=np.swapaxes(self.data, 0, 2)
        self.data = np.moveaxis(self.data, divisible_axis[0], 2) if divisible_axis else self.data
        print('Data shape: ', self.data.shape)
        self.data_original_shape = self.data.shape
        self.data = np.pad(self.data,((64,0),(0,0),(0,0)), self.pad)
        print('Data shape: ', self.data.shape)

    def visualise_data(self):
        plt.figure()
        plt.plot(self.data[0,:,0])
        plt.show()
        
        plt.figure()
        plt.imshow(self.data[10,:,:], aspect='auto')
        plt.show()
        
        #print c_scan
        plt.figure()
        plt.imshow(np.amax(self.data[:,250:,:],1))
        plt.show()

    def flip_data(self):
        self.data = self.data[64:,:,:]
        self.data = np.flip(self.data, axis=0)
        self.data = np.pad(self.data,((64,0),(0,0),(0,0)), self.pad)
        
    def moving_average_3d(self, window_size=20):
        window = np.ones(window_size) / window_size
        self.data = np.apply_along_axis(lambda m: np.convolve(m, window, 'same'), axis=1, arr=self.data)


    def preprocess_data(self, data):
        # done after applying hilbert transform
        shifted_ut = np.zeros(data.shape)
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                a_scan = data[i,:,j]
                peak = np.argmax(a_scan)
                shifted_a_scan = a_scan[peak::]
                shifted_ut[i,0:(data.shape[1]-peak),j] = shifted_a_scan
                shifted_ut[i,:,j] = shifted_ut[i,:,j]/np.nanmax(shifted_ut[i,:,j])
                shifted_ut[i,:,j] = data[i,:,j]/np.nanmax(data[i,:,j])
        return shifted_ut
            
    def __getitem__(self, index):       
        base_data=self.data[index:index+65, :, :]           
        data = base_data[:64]
        label = base_data[64]      
        return torch.from_numpy(np.copy(data)), torch.from_numpy(np.copy(label))
        #At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().) 

    def __len__(self):
        return self.data.shape[0]-64
    
    
class Evaluator:
    def __init__(self, threshold=0.99):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.Net = None
        self.threshold = threshold
        self.volume_forward = None
        self.volume_backward = None
        self.volume_combined = None
        self.volume_thresholded = None

    def initialise_network(self, model_path):
        self.Net = model()
        self.Net.load_state_dict(torch.load(model_path))
        self.Net = self.Net.double()
        self.Net.to(self.device)
        self.Net = nn.DataParallel(self.Net)
        print(summary(self.Net.float(), input_size=(8, 1, 64)))


    def evaluate_frame(self, test_data, targets):
        '''
        Evaluate a single frame of data.

        Args:
            test_data (numpy.ndarray): The input test data.
            targets (torch.Tensor): The target values.

        Returns:
            segmented (numpy.ndarray): The segmented mask.
            test_clean (numpy.ndarray): The cleaned test data.
        '''

        input_shape = targets.shape
        
        targets = targets.flatten()

        test_clean=torch.Tensor(np.copy(targets)).to(self.device)
        test_clean=test_clean.float()

        #flatten test data
        test_data = np.reshape(test_data, (64,-1))
        test_data = test_data.swapaxes(1, 0)

        test_data = test_data.unsqueeze(1)
        segmented = torch.Tensor(np.zeros(targets.shape)).to(self.device)
        
        prev_time = time.time()
        
        threshold=np.full(targets.flatten().shape, self.threshold)
        threshold=torch.Tensor(threshold).to(self.device)
        self.Net.eval()        
        with torch.no_grad():
            test_data=test_data.to(self.device)
            targets=targets.to(self.device)
                
            scale, concentration = self.Net(test_data.float())
            try:
                dists=dist.Weibull(scale, concentration)
            except:
                print('ERROR Can not generate distributions')
                print()
                    
            #get means of dist
            out_means=dists.mean
            
            threshold_values=dists.icdf(threshold).squeeze()

            #mask if target is greater than threshold    
            mask=targets>threshold_values       
            #replace target with mean if target is greater than threshold
            test_clean[mask]=out_means[mask]
            #set segmented mask          
            segmented[mask]=1
        return segmented.reshape(input_shape), test_clean.reshape(input_shape)

    def evaluate_sweep(self, dataset):
        test_clean=None
        segmented_volume = torch.tensor(np.zeros(dataset.data_original_shape)).to(self.device)
        for i, batch in enumerate(tqdm(dataset, position=0, leave=True, desc = 'Processing Sweep')):
            data, labels = batch
            if i>0:
                #make last data the test_clean
                data[-1,:,:]=test_clean
            segmented, test_clean = self.evaluate_frame(data, labels)
            segmented_volume[i,:,:]=segmented
        return segmented_volume
    
    def threshold_area(self, volume, area_filter=5, pitch=None, area=None):
        if pitch:
            area_filter = area_filter / (pitch ** 2)
            print(f'Area Threshold {round(area_filter,2)} mm')
            
        binary = volume
        threholded_volume = np.zeros(binary.shape)

        for depth in tqdm(range(binary.shape[1]), desc='Areas Thresholding'):
            thresholded_slice = skimage.morphology.area_opening(volume[:, depth, :], area_threshold=area_filter, connectivity=2)
            threholded_volume[:, depth, :] = thresholded_slice
        return threholded_volume
    
    def visualise_data(self, volume):
        #print c_scan
        plt.figure()
        plt.imshow(np.amax(volume[:,:,:],1))
        plt.show()
        
        
    def get_areas(self, thresholded_volume):
        labeled_volume, num_labels = label(np.amax(thresholded_volume, axis=1), connectivity=2, return_num=True)
        properties = regionprops(labeled_volume)
        centers = [prop.centroid for prop in properties]
        areas = [prop.area for prop in properties]
        return areas, centers
            
    def get_diameters(self, areas, pitch=0.8):
        diameters = [np.sqrt(4*area/np.pi)*pitch for area in areas]
        widths = [np.sqrt(area)*pitch for area in areas]
        return diameters, widths

    def plot_diameters_on_cscan(self, volume, diameters, centers):
        plt.imshow(np.amax(volume, axis=1), cmap='gray')
        for center, diameter in zip(centers, diameters):
            plt.text(center[1], center[0], f'{diameter:.2f}', color='red')
        plt.show()

        
    def save_data(self, path, sample, threshold):
        np.save(os.path.join(path, sample, f'{threshold}_forward'), self.volume_forward)
        np.save(os.path.join(path, sample, f'{threshold}_backward'), self.volume_backward)
        np.save(os.path.join(path, sample, f'{threshold}_combined'), self.volume_combined)
        np.save(os.path.join(path, sample, f'{threshold}_thresholded'), self.volume_thresholded)
    
def main():
    #load data
    dataset = InferenceDataset(
        '/media/cue-server/ubuntuStorage/ShaunMcKnight/Data/lear/belfast/LEAR85_2_lear 85CH 1.RF_normalised_data.npy',
        pad = 'reflect')
    dataset.visualise_data()
    dataset.moving_average_3d(window_size=30)
    dataset.visualise_data()
    # dataset.data=dataset.preprocess_data(dataset.data)
    dataset.visualise_data()
    print('length: ', len(dataset))

    #process forward sweep
    evaluator = Evaluator(threshold = 0.99999)
    evaluator.initialise_network('/media/cue-server/ubuntuStorage/ShaunMcKnight/Self-supervised-volumetric-detection-2/V2/results/stride_2/1_best_model.pth')
    evaluator.volume_forward = evaluator.evaluate_sweep(dataset).cpu().numpy()
    evaluator.visualise_data(evaluator.volume_forward)

    #process backward sweep
    dataset.flip_data()
    dataset.visualise_data()
    evaluator.volume_backward = np.flip(evaluator.evaluate_sweep(dataset).cpu().numpy(),axis=0)
    evaluator.visualise_data(evaluator.volume_backward)
    
    #combine forward and backward sweep
    evaluator.volume_combined = np.add(evaluator.volume_forward[:,:,:], 
                                        evaluator.volume_backward[:,:,:])
    evaluator.visualise_data(evaluator.volume_combined)
    evaluator.volume_combined[evaluator.volume_combined<2]=0    
    evaluator.visualise_data(evaluator.volume_combined)
    
    #area thresholding
    evaluator.volume_thresholded = evaluator.threshold_area(evaluator.volume_combined, area_filter=(3**2), pitch=2.)
    evaluator.visualise_data(evaluator.volume_thresholded)
    
    areas, centers = evaluator.get_areas(evaluator.volume_thresholded)
    diameters, widths = evaluator.get_diameters(areas)
    print(widths)
    print(areas)
    evaluator.plot_diameters_on_cscan(evaluator.volume_thresholded, widths, centers)
    evaluator.plot_diameters_on_cscan(evaluator.volume_thresholded, areas, centers)

    #save results
    sample='lear_belfast_rectified_averaged'
    path=os.path.join(os.getcwd(), 'inference_results')
    evaluator.save_data(path=path , sample = sample, threshold = evaluator.threshold)
    np.save(os.path.join(path, sample,'test_volume'), dataset.original_data)

main()