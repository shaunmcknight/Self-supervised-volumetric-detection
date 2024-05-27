import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
from classifierv2 import CNN_shared as model
import skimage
import scienceplots
import time
plt.style.use(['science', 'ieee','no-latex', 'bright'])

class InferenceDataset(Dataset):
    def __init__(self, path, pad='edge'):
        self.data = np.load(path)
        self.pad = pad 

        nan_indices = np.argwhere(np.isnan(test_mapped))
        if nan_indices.size > 0:
            print("NaN values are located at:")
            for index in nan_indices:
                print(index)
                raise SystemExit(0)
        
        # check what data axis is divisibble for 61 and move to first axis
        self.data = np.moveaxis(self.data, np.argwhere(self.data.shape % 61 == 0)[0], 2)
        print('Data shape: ', self.data.shape)

        self.data = np.pad(self.data,((64,0),(0,0),(0,0)), self.pad)
        print('Data shape: ', self.data.shape)

    def visualise_data(self):
        #print c_scan
        plt.figure()
        plt.imshow(np.amax(self.data[:,200:,:],1))
        plt.show()

    def flip_data(self):
        self.data = self.data[64:,:,:]
        self.data = np.flip(self.data, axis=0)
        self.data = np.pad(self.data,((64,0),(0,0),(0,0)), self.pad)

    def preprocess_data(self, data):
        # done after applying hilbert transform
        shifted_ut = np.zeros(data.shape)
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):\
                # a_scan = data[i,:,j]
                # peak = np.argmax(a_scan)
                # shifted_a_scan = a_scan[peak::]
                # shifted_ut[i,0:(data.shape[1]-peak),j] = shifted_a_scan
                # shifted_ut[i,:,j] = shifted_ut[i,:,j]/np.nanmax(shifted_ut[i,:,j])
                shifted_ut[i,:,j] = data[i,:,j]/np.nanmax(data[i,:,j])
        return shifted_ut
            
    def __getitem__(self, index):       
        base_data=self.data[index:index+65, :, :]           
        data = base_data[:64]
        label = base_data[64]      
        return torch.from_numpy(np.copy(data)), torch.from_numpy(np.copy(label))
        #At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().) 

    def __len__(self):
        return self.data.shape[2]-65
    
    
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
        self.Net.to(self.device)

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

        input_shape = test_data.shape
        #flatten test data
        test_data = test_data.flatten()

        segmented = np.zeros(test_data.shape)
        test_clean=np.copy(test_data)
        
        prev_time = time.time()
        
        threshold=np.full(test_clean.shape[0], self.threshold)
        threshold=torch.Tensor(threshold).to(self.device).unsqueeze(1)
        self.Net.eval()        
        with torch.no_grad():
            test_data=test_data.to(self.device)
            targets=targets.to(self.device)
                
            scale, concentration = self.Net(test_data)
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
            segmented[test_data!=test_clean]=1
        inference_time = time.time()-prev_time
        print('Finished Validation, inference time: ', inference_time)
        return segmented.reshape(input_shape), test_clean.reshape(input_shape)

    def evaluate_sweep(self, dataset):
        segmented_volume = np.zeros(dataset.data.shape())
        for i, batch in enumerate(dataset):
            data, labels = batch
            if i>0:
                #make last data the test_clean
                data[:,:-1]=test_clean
            segmented, test_clean = self.evaluate_frame(data.unsqueeze(0), labels.unsqueeze(0))
            segmented_volume[i,:,:]=segmented
        return segmented_volume, 
    
    def threshold_area(self, volume, area_filter=5, pitch=None, area=None):
        if pitch:
            area_filter = area * (pitch ** 2)
            
        binary = volume
        threholded_volume = np.zeros(binary.shape)

        for depth in range(binary.shape[1]):
            thresholded_slice = skimage.morphology.area_opening(volume[:, depth, :], area_threshold=area_filter, connectivity=2)
            threholded_volume[:, depth, :] = thresholded_slice
        return threholded_volume
    
    def visualise_data(self, volume):
        #print c_scan
        plt.figure()
        plt.imshow(np.amax(volume[:,:,:],1))
        plt.show()

    
def main():
    #load data
    dataset = InferenceDataset('C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/SEARCH NDE/Composites/Exp Data/Spirit Cell/LearJet/LEAR_Belfast/LEAR_Belfast/.nka files/LEAR85_2_lear 85CH 1.RF_normalised_data.npy')
    dataset.visualise_data()

    #process forward sweep
    evaluator = Evaluator()
    evaluator.initialise_network('C:/GIT/Self-supervised-volumetric-detection/saved_models/dist/epoch_69_model.pth')
    evaluator.volume_forward = evaluator.evaluate_sweep(dataset)
    
    #process backward sweep
    dataset.flip_data()
    dataset.visualise_data()
    evaluator.volume_backward = evaluator.evaluate_sweep(dataset)

    #combine forward and backward sweep
    evaluator.volume_combined = np.add(evaluator.segmented_volume_forward[64:,:,:], evaluator.segmented_volume_backwards[64:,:,:])
    evaluator.visualise_data(evaluator.volume_combined)
    evaluator.volume_combined[evaluator.segmented_volume_combined<2]=0    
    
    #area thresholding
    evaluator.volume_thresholded = evaluator.threshold_area(evaluator.segmented_volume_combined, area_filter=5)
    evaluator.visualise_data(evaluator.volume_thresholded)

    #save results
    # np.save('results/segmented_volume', evaluator.volume_thresholded)
