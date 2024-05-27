from torch.utils.data import DataLoader
import numpy as np

import sys
import os
sys.path.append(r"C:\GIT\Self-supervised-volumetric-detection\V2")

from utilsv2 import *
from trainv2 import Trainer
from config import hp

def VisualiseData(dataloader,title):
    example = next(iter(dataloader))
    data = example[0][0].squeeze().cpu().detach().numpy()
    label = example[1][0].squeeze().cpu().detach().numpy()
    
    # print('Data: ', data.shape, data)
    # print('Label: ', label.shape, label)
    
    plt.figure()
    plt.title(title)
    plt.plot(data)
    plt.scatter(65,label, marker='x', color='red')
    plt.legend(('Train', 'Test'))
    plt.show()

def display_cscan(data, gate=100):
    plt.figure()
    plt.imshow(np.amax(data[:,gate:,:], axis=1), aspect='auto')
    plt.show()

def display_bscan(data, gate=100):
    plt.figure()
    plt.imshow(data[data.shape[0]//2,:,:], aspect='auto')
    plt.show()

def setup_dataloaders(training_data, validation_data, test_data):    
    transforms = ['flip', 'noise', 'erase']
    train_dataloader= DataLoader(
        CleanDataset(data=training_data, transforms=transforms, label='Train'),
        batch_size=hp.batch_size,
        shuffle=True,
    )

    valid_dataloader= DataLoader(
        CleanDataset(data=validation_data, transforms=None, label='Validation', stride=64),
        batch_size=hp.batch_size,
        shuffle=False,
    )
    
    test_dataloader= DataLoader(
        CleanDataset(data=test_data, transforms=None, label='Test', stride=1),
        batch_size=hp.batch_size,
        shuffle=False,
    )
    
    # VisualiseData(train_dataloader,'Train')
    # VisualiseData(test_dataloader, 'Test')
        
    return train_dataloader, valid_dataloader, test_dataloader 


base_dir = r'C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\SEARCH NDE\Composites\Exp Data\Spirit Cell\New scans'
training_arrays = ['ID007', 'ID006','ID005']
training_arrays = [np.load(os.path.join(base_dir, x, x+'_hilbert_raster.npy')) for x in training_arrays]
training_arrays = [x[0:260,::5,:] for x in training_arrays] # cut down number of frames to match
time_limits = [600, 450, 450, 350]   #set time limits to just past backwall
training_arrays = [x[:,:time_limits[i],:] for i, x in enumerate(training_arrays)] # apply echo limits 

# [print(x.shape) for x in training_arrays]
# [display_cscan(x) for x in training_arrays]

validation_arrays = ['ID008']
validation_arrays = [np.load(os.path.join(base_dir, x, x+'_hilbert_raster.npy')) for x in validation_arrays]
validation_arrays = [x[0:260,::5,:] for x in validation_arrays] # cut down number of frames to match
time_limits = [600]   #set time limits to just past backwall
validation_arrays = [x[:,:time_limits[i],:] for i, x in enumerate(validation_arrays)] # apply echo limits 
# [display_bscan(x) for x in validation_arrays]
# [display_cscan(x) for x in validation_arrays]

test_arrays = [np.load(r'C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\SEARCH NDE\Composites\Exp Data\Spirit Cell\Small CFRP Samples\ID010\ID010_hilbert.npy')]
test_arrays = [x[0:260,:,:] for x in test_arrays] # cut down number of frames to match
time_limits = [700]   #set time limits to just past backwall
test_arrays = [x[:,:time_limits[i],:] for i, x in enumerate(test_arrays)] # apply echo limits 
# [display_bscan(x) for x in test_arrays]
# [display_cscan(x) for x in test_arrays]
# [print(x.shape) for x in test_arrays]

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
    train_dataloader, validation_dataloader, test_dataloader = setup_dataloaders(
        training_data=training_arrays, validation_data=validation_arrays, test_data=test_arrays)
    break
    trainer = Trainer(train_dataloader, validation_dataloader, test_dataloader, i)
    trainer.train()
    trainer.test(trainer.Net, trainer.dataloader_test)
    # errors_percent, errors_priors, targets, prediction = main(
    #     train_dataloader, validation_dataloader, test_dataloader, iteration=i) 
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
