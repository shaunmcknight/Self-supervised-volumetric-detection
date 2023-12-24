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

def VisualiseData(dataloader,title):
    example = next(iter(dataloader))
    data = example[0][0].squeeze().cpu().detach().numpy()
    label = example[1][0].squeeze().cpu().detach().numpy()
    
    print('Data: ', data.shape, data)
    print('Label: ', label.shape, label)
    
    plt.figure()
    plt.title(title)
    plt.plot(data)
    plt.scatter(65,label, marker='x', color='red')
    plt.legend(('Train', 'Test'))
    plt.show()
    
def experimental(HP, transforms, exp_path, iteration):    
    train_dataloader= DataLoader(
        CleanDataset(),
        batch_size=hp.batch_size,
        shuffle=False,
        # num_workers=1,
    )
    
    test_dataloader = DataLoader(
        TestDataset(),
        batch_size=hp.batch_size,
        shuffle=False,
        # num_workers=1,
    )
    
    split_train, split_valid = round(len(train_dataloader.dataset)*0.7), round(
        len(train_dataloader.dataset))-round(len(train_dataloader.dataset)*0.7)
        
    train, _ = torch.utils.data.random_split(
        train_dataloader.dataset, (split_train, split_valid),
        torch.Generator().manual_seed(42))
            
    _, test = torch.utils.data.random_split(
        test_dataloader.dataset, (split_train, split_valid),
        torch.Generator().manual_seed(42))
    
    split_valid, split_test = round(split_valid/2), split_valid-round(split_valid/2)
    
    test, valid = torch.utils.data.random_split(
        test, (split_test, split_valid), torch.Generator().manual_seed(42))
    
    train_dataloader = DataLoader(
        train,
        batch_size=hp.batch_size,
        shuffle=True,
    )
    
    valid_dataloader = DataLoader(
        valid,
        batch_size=hp.batch_size,
        shuffle=False,
    )

    test_dataloader = DataLoader(
        test,
        batch_size=hp.batch_size,
        shuffle=False,
    )
    
    VisualiseData(train_dataloader,'Train')
    VisualiseData(test_dataloader, 'Test')
    
    errors_percent, errors_priors, targets, prediction = main(HP, 
          train_dataloader=train_dataloader,
          validation_dataloader=valid_dataloader, 
          test_dataloader=test_dataloader, 
          cuda_device=None,
          iteration = iteration)
    
    return  errors_percent, errors_priors, targets, prediction
    


#optimisation of experimental data
HP = {'n_epochs': 1000, #300
      'batch_size': 512, #1024
      'lr': 0.001, #0.0005,
      'momentum': 0.175764011181887,
      'early_stop': 0,
      'conv_layers': 3,
      'out_channel_ratio': 3,
      'FC_layers': 1,
      'folder':'dist'
      }

hp = Hyperparameters(
    epoch=0,
    n_epochs=HP['n_epochs'],
    dataset_train_mode="train",
    dataset_test_mode="test",
    batch_size=HP['batch_size'],
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
    
    errors_percent, errors_priors, targets, prediction = experimental(HP, transforms=transforms_, exp_path=exp_path, iteration = i)

#     accuracies.append(accuracy)
#     precisions.append(precision)
#     recalls.append(recall)
#     f_scores.append(f_score)
#     confusion_matrixes.append(cm)
    

# print('')
# print('~~~~~~~~~~~~~~~~~')
# print("~ Mean results ~")
# print('~~~~~~~~~~~~~~~~~')
# print("")
# print('Accuracy ~ mu {}. std {}. '.format(np.mean(accuracies),np.std(accuracies)))
# print('Precision ~ mu {}. std {}. '.format(np.mean(precisions),np.std(precisions)))
# print('Recall ~ mu {}. std {}. '.format(np.mean(recalls),np.std(recalls)))
# print('F score ~ mu {}. std {}. '.format(np.mean(f_scores),np.std(f_scores)))

# print('Confusion matrix')
# cm = np.array(confusion_matrixes)
# cm = np.mean(cm, axis = 0)
# print(cm)

# print('')
# print('~~~~~~~~~~~~~~~~~')
# print("~ Max results ~")
# print('~~~~~~~~~~~~~~~~~')
# print('')
# print('Accuracy ~ Max {}. '.format(np.amax(accuracies)))
# print('Precision ~ Max {}. '.format(np.amax(precisions)))
# print('Recall ~ Max {}. '.format(np.amax(recalls)))
# print('F score ~ Max {}. '.format(np.amax(f_scores)))



# """

# TO DO:
#     add in training lists for 100 iterations to get std and averages
#     re-train GAN
    
# """
