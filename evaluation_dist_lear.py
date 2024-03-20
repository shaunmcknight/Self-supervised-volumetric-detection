import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'ieee','no-latex', 'bright'])

#helper functions
def get_error(pred, target):
    # differences = np.abs(target-pred)
    differences = target-pred
    return 100*(differences/target)

def preprocess_norm(data):
    # done after applying hilbert transform
    shifted_ut = np.zeros(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            shifted_ut[i,:,j] = data[i,:,j]/np.nanmax(data[i,:,j])
    return shifted_ut

def preprocess_align(data):
    # done after applying hilbert transform
    shifted_ut = np.zeros(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            a_scan = data[i,:,j]
            peak = np.argmax(a_scan[:300])
            shifted_a_scan = a_scan[peak::]
            shifted_ut[i,0:(data.shape[1]-peak),j] = shifted_a_scan
    return shifted_ut

def moving_average_3d(array, window_size):
    # Define a kernel for convolution
    kernel = np.ones(window_size) / window_size
    
    # Apply convolution along the second axis
    result = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=1, arr=array)
    
    return result

# root='C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/SEARCH NDE/Composites/Exp Data/Spirit Cell/LearJet/lear3/lear3_hilbert_raster.npy'
# root='C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/SPIRIT WORK FOLDER/SAMPLE DATABASE/Spirit data/ML Collaboration Strathclyde University/Program A/MRD-CBSP001 (1024 samples)/5MHz_hilbert_data.npy'
root='C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/SPIRIT WORK FOLDER/SAMPLE DATABASE/Spirit data/ML Collaboration Strathclyde University/Program A/MRD-CBSP001 (1024 samples)/5MHz_normalised_data.npy'
root='C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/SEARCH NDE/Composites/Exp Data/Spirit Cell/LearJet/LEAR_Belfast/LEAR_Belfast/.nka files/LEAR85_2_lear 85CH 1.RF_normalised_data.npy'
data=abs(np.load(root))

data=moving_average_3d(data, 20)

plt.figure()
plt.plot(data[10,:,10])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

print(data.shape)

data=preprocess_norm(data)
data=preprocess_align(data)


plt.figure()
plt.imshow(data[round(data.shape[0]/2),:,:], aspect='auto')
plt.show()

plt.figure()
plt.imshow(data[:,:,round(data.shape[2]/2)], aspect='auto')
plt.show()

plt.figure()
plt.imshow(np.amax(data[:,200:,:], axis=1))
plt.show()

test=np.copy(data)

plt.figure()
plt.imshow(np.amax(test[:,100:,:], axis=1))
plt.show()

# %% evaluate


import numpy as np
import itertools
import time
import datetime
import math
from pathlib import Path
import os
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch
from torchinfo import summary
from torch.utils.data import Dataset
import torch.distributions as dist

sys.path.append('C:\GIT\Self-supervised-volumetric-detection')

import classifier
# output_file_path = "console_output.txt"
# original_stdout = sys.stdout
# with open(output_file_path, "w") as f:
#     sys.stdout = f

class CleanDataset(Dataset):
    def __init__(self, data, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        
        self.id007=data
            
        print('~~~~ Synthetic dataloader INFO ~~~~~')
        print('Number points train ~ ', 'Max/Min ', np.amax(self.id007), np.amin(self.id007))
        #length 500 time series
              
    def __getitem__(self, index):       
        base_data=self.id007[index:index+65]
        
        data = base_data[:64]
        label = base_data[64]   
        
        return torch.from_numpy(np.copy(data)).unsqueeze(0), torch.from_numpy(np.copy(label)).unsqueeze(0)
        #At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().) 

    def __len__(self):
        return self.id007.size-65


def validationv2(
    test_data,
    Tensor,
    Net,
    threshold=0.99,
    plot=None
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    predictions = np.zeros(test_data.shape)
    segmented = np.zeros(test_data.shape)
    errors = np.zeros(test_data.shape)
    
    test_mapped=np.copy(test_data)
    prediction_mapped=[]
    
    prev_time = time.time()
    Net.eval()
    
    threshold=np.full(test_mapped.shape[0], threshold)
    threshold=torch.Tensor(threshold).to(device).unsqueeze(1)


    nan_indices = np.argwhere(np.isnan(test_mapped))
    if nan_indices.size > 0:
        print("NaN values are located at:")
        for index in nan_indices:
            print(index)
            raise SystemExit(0)
    # else:
    #     print("Test data does not contain NaN values.")
        
    with torch.no_grad():
        for i in range(0,test_mapped.shape[1]-65):
            batch=torch.Tensor(np.expand_dims(test_mapped[:,i:i+64],1))
            targets=np.squeeze(test_mapped[:,i+64])
            targets=torch.Tensor(targets).cpu().numpy()          
            batch=batch.to(device)
            
            out_means, out_std = Net(batch)
            try:
                dists=dist.Normal(out_means, out_std)
            except:
                print('ERROR Can not generate distributions')
                print()
                
            out_means=out_means.squeeze().cpu().numpy()

         
            threshold_values=dists.icdf(threshold).squeeze().cpu().numpy()
             
            mask=np.copy(targets)>np.copy(threshold_values)
            

            test_mapped[:,i+64][mask]=out_means[mask] # check these are working, strange broadcasting behavour
            # segmented[:,i+64]=mask
            predictions[:,i+64]=out_means
            errors[:,i+64]=get_error(out_means, targets)
            
            if plot:
                # if i>150:
                idx=0
                
                print(mask[idx])
                print('Target    : ',targets[idx])
                print('Prediction: ',out_means[idx])
                print('Threshold : ',threshold_values[idx])
                if mask[idx]==1:
                    print('Threshold exceeded!!!', mask[idx])
                plt.figure(figsize=(10,5))
                plt.plot(batch[idx,:,:].cpu().numpy().squeeze().squeeze(), label='Signal')
                plt.scatter(64, targets[idx], label='GT', marker='o')
                plt.scatter(64, out_means[idx], label='Mean prediction', marker='x')
                plt.scatter(64, threshold_values[idx], label='Threshold', marker='_')
                plt.legend()
                plt.xlabel('Samples')
                plt.ylabel('Amplitude')
                plt.show()
                
        segmented[test_data!=test_mapped]=1
    inference_time = time.time()-prev_time
    print('Finished Validation, inference time: ', inference_time)
    return errors, segmented, predictions, test_mapped, inference_time


#data loader
test_dataloader = DataLoader(
        CleanDataset(test, transforms_ = True),
        batch_size=2,
        shuffle=False,
        # num_workers=1,
)

# model_path = 'C:/GIT/Self-supervised-volumetric-detection/saved_models/exp_4/best_model_epoch_702_model.pt' #209 is model w/o adjustment up
model_path = 'C:/GIT/Self-supervised-volumetric-detection/saved_models/dist/epoch_69_model.pth'
top_model = classifier.CNN()
top_model.load_state_dict(torch.load(model_path))

# top_model = torch.load(model_path)
# print(top_model)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

if cuda:
    top_model = top_model.to('cuda')
    print('Cuda')

# test_main=np.copy(data[:530,50:900,100:360]) #lear our
test_main=np.copy(data[60:200,0:900,50:270]) #lear tecnhatom
# test_main=np.copy(data[10:100,:600,5:380]) #us data
# test_main=np.swapaxes(test_main, 0,2) #for learjet our
test=np.pad(test_main,((64,0),(0,0),(0,0)),'reflect')

print(test.shape)
plt.figure()
plt.imshow(np.amax(data[:,200:,:],1))
plt.clim(0,1)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(np.amax(test[:,:400,:],1))
# plt.clim(0,1)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(np.amax(test[:,400:,:],1))
# plt.clim(0,1)
plt.colorbar()
plt.show()


# # Forward

# In[13]:




# plt.imshow(np.amax(test[:,:,:], axis=1),aspect='auto')
# plt.show()

def evaluate_d_scans(test, thresh=0.99):
    segmented_volume=np.zeros(test.shape)
    errors_total=np.zeros(test.shape)

    plt.imshow(test[:,:,0])
    
    plt.figure()
    plt.plot(test[:,32,0])
    # plt.plot(test[140:,32,0])
    plt.show()
    
    start_time=time.time()
    
    for i in range(0,test.shape[1]): #13-16
        # test_slice=test[:,i,:]
        test_slice=test[:,i,:]
        errors, segmented, predictions, mapped_data, inference_time = validationv2(
                np.swapaxes(test_slice+1,0,1),#[20:23,:], #need d axes at the end
                Tensor=Tensor,
                Net=top_model,
                threshold=thresh,
                # threshold=0.9999999,
                # plot=True
        )
        segmented_volume[:,i,:]=np.swapaxes(segmented, 0,1)
        errors_total[:,i,:]=np.swapaxes(errors, 0,1)
        elapsed_time = time.time()-start_time
        total_time = elapsed_time/((i+0.0001)/segmented_volume.shape[1])
        
        print(f'{round(100*(i/segmented_volume.shape[1]), 2)}% complete ~ ETA: {round((total_time-elapsed_time)/60, 2)} Mins')
    print('FINISHED')
    print('Complete time: ', time.time() - start_time)
    return segmented_volume


#runs along shape[0], which is running from top to bottom of the image
segmented_forward = evaluate_d_scans(test, 0.999999)
test=np.pad(test_main,((0,64),(0,0),(0,0)),'reflect')
test=np.flip(test, axis=0)
# backwards
segmented_volume_flipped = evaluate_d_scans(test,0.999999)

both_ways=np.copy(segmented_forward[64:,:,:])
both_ways=np.add(np.flip(segmented_volume_flipped[64:,:,:], axis=0), both_ways)


plt.figure()
plt.imshow(np.amax(segmented_forward[:,:,:], axis=1),aspect='auto')
plt.colorbar()
# plt.savefig('forward.png')
plt.show()


plt.figure()
plt.imshow(np.amax(segmented_volume_flipped[:,:,:], axis=1),aspect='auto')
plt.colorbar()
# plt.savefig('backward.png')
plt.show()

plt.figure()
plt.imshow(np.amax(both_ways, axis=1),aspect='auto')
plt.colorbar()
# plt.savefig('forward_and_backward.png')
plt.show()

both_ways[both_ways<2]=0

plt.figure()
plt.imshow(np.amax(test_main[:,200:,:], axis=1),aspect='auto')
plt.colorbar()
# plt.savefig('forward_and_backward.png')
plt.show()

plt.figure()
plt.imshow(np.amax(both_ways, axis=1),aspect='auto')
plt.colorbar()
# plt.savefig('forward_and_backward.png')
plt.show()


# In[15]:


# np.save(r'C:\GIT\Self-supervised-volumetric-detection\results\US\Program A\MRD-CBSP001/both_segmented_volume_rectified_avg20',both_ways)
# np.save(r'C:\GIT\Self-supervised-volumetric-detection\results\US\Program A\MRD-CBSP001/test_data',test)
np.save('results/segmented_volume_learTechnatom_averageFilter',both_ways)
# np.('errors',errors_total)
np.save('test_learTechnatom_averageFilter',test)

# #%%

# sys.stdout = original_stdout


