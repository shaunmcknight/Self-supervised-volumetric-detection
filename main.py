# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 09:19:42 2022

@author: Shaun McKnight
"""

import torch
import numpy as np


from utils import *
from classifier import *
from train import *

def VisualiseData(dataloader):
    test_data = next(iter(dataloader))[0][0].squeeze().cpu().detach().numpy()
    
    # print('Type: ', type(test_data[:,150:450,:]), test_data[:,150:450,:].shape)
    plt.figure()
    plt.imshow(np.amax(test_data[:,150:450,:], axis=1))
    plt.show()
    
    fig = plt.figure().add_subplot(projection="3d")
    fig.set_xlabel("sample")
    fig.set_ylabel("Amplitude")
    fig.set_zlabel("Element no.")
    frame = 32
    for i in range(test_data[:,:,0:64].shape[2]):       
        xlist = [x for x in range(1, (len(test_data[frame,:, i])+1))]
        plt.plot(
            xs=xlist, ys=test_data[frame,:, i], zs=i, label='Raw ' + str(i))
        plt.title(("All A-scans of frame: " + str(frame)))
    plt.show()
    
def experimental(HP, transforms, exp_path, iteration):

    root_test = r"C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\PhD\Data\3D\data\test"
    root_train = r"C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\PhD\Data\3D\data\noised"

    synthetic_dataloader = DataLoader(
        SyntheticDataset(root_train, mode=hp.dataset_train_mode, transforms_ = transforms_),
        batch_size=hp.batch_size,
        shuffle=True,
        # num_workers=1,
    )
    
    experimental_dataloader = DataLoader(
        ExperimentalDataset(root_test, transforms_=transforms_),
        batch_size=hp.batch_size,
        shuffle=False,
    )
    
    split_train, split_valid = round(len(synthetic_dataloader.dataset)*0.8), round(
        len(synthetic_dataloader.dataset))-round(len(synthetic_dataloader.dataset)*0.8)
    
    train, valid = torch.utils.data.random_split(
        synthetic_dataloader.dataset, (split_train, split_valid))
    
    train_dataloader = DataLoader(
        train,
        batch_size=hp.batch_size,
        shuffle=True,
    )
    
    test_dataloader = DataLoader(
        valid,
        batch_size=hp.batch_size,
        shuffle=False,
    )
    
    VisualiseData(synthetic_dataloader)
    VisualiseData(experimental_dataloader)

    accuracy, precision, recall, f_score, cm = main(HP, 
         train_dataloader=train,
         validation_dataloader=valid, 
         test_dataloader=experimental_dataloader, 
         cuda_device=None,
         iteration = iteration)
    
    return accuracy, precision, recall, f_score, cm
    


#optimisation of experimental data
HP = {'n_epochs': 264,
      'batch_size': 4, #64
      'lr': 0.013870869810956584, #0.03,
      'momentum': 0.175764011181887,
      'early_stop': 1,
      'conv_layers': 3,
      'out_channel_ratio': 3,
      'FC_layers': 1
      }

hp = Hyperparameters(
    epoch=0,
    n_epochs=HP['n_epochs'],
    dataset_train_mode="train",
    dataset_test_mode="test",
    batch_size=2,#2**HP['batch_size'],
    lr=HP['lr'],
    momentum=HP['momentum'],
    img_size=64,
    channels=1,
    early_stop=HP['early_stop']
)


transforms_ = [
    # transforms.ToTensor(),
    # transforms.Normalize((0.5), (0.5)),
]

exp_path = "C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/PhD/Data/classifier/simple/experimental"

accuracies = []
precisions = []
recalls = []
f_scores = []
confusion_matrixes = []
trps = []
fprs = []

for i in range(1):

    print('Model iteration ~ ', i)
    
    """Experimental"""
    
    accuracy, precision, recall, f_score, cm = experimental(HP, transforms=transforms_, exp_path=exp_path, iteration = i)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f_scores.append(f_score)
    confusion_matrixes.append(cm)
    

print('')
print('~~~~~~~~~~~~~~~~~')
print("~ Mean results ~")
print('~~~~~~~~~~~~~~~~~')
print("")
print('Accuracy ~ mu {}. std {}. '.format(np.mean(accuracies),np.std(accuracies)))
print('Precision ~ mu {}. std {}. '.format(np.mean(precisions),np.std(precisions)))
print('Recall ~ mu {}. std {}. '.format(np.mean(recalls),np.std(recalls)))
print('F score ~ mu {}. std {}. '.format(np.mean(f_scores),np.std(f_scores)))

print('Confusion matrix')
cm = np.array(confusion_matrixes)
cm = np.mean(cm, axis = 0)
print(cm)

print('')
print('~~~~~~~~~~~~~~~~~')
print("~ Max results ~")
print('~~~~~~~~~~~~~~~~~')
print('')
print('Accuracy ~ Max {}. '.format(np.amax(accuracies)))
print('Precision ~ Max {}. '.format(np.amax(precisions)))
print('Recall ~ Max {}. '.format(np.amax(recalls)))
print('F score ~ Max {}. '.format(np.amax(f_scores)))



"""

TO DO:
    add in training lists for 100 iterations to get std and averages
    re-train GAN
    
"""
