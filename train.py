import numpy as np
import itertools
import time
import datetime
import math
from pathlib import Path
import os


import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.distributions as dist
from torchinfo import summary

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scienceplots
from mpl_toolkits.axes_grid1 import ImageGrid

from IPython.display import clear_output
from sklearn import metrics 

from PIL import Image
import matplotlib.image as mpimg

from utils import *
from classifier import *

plt.style.use(['science', 'ieee','no-latex', 'bright'])

##############################################
# Defining all hyperparameters
##############################################


class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
class SMAPE(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true):
        absolute_percentage_error = torch.abs(y_true - y_pred) / ((torch.abs(y_true) + torch.abs(y_pred)) / 2)
        smape = 100.0 * torch.mean(absolute_percentage_error)
        return smape
        
class NegativeWeibullLoss(nn.Module):
    def __init__(self):
        super().__init__()
        #negative loss liklihood for negative binomial distribution
        
    def forward(self, scale, concentration, target):
        # Ensure concentration is positive
        concentration = torch.abs(concentration)
        scale = torch.abs(scale)

        # Define the Weibull distribution
        weibull_distribution = dist.Weibull(scale, concentration)

        # Calculate the negative log likelihood
        loss = -weibull_distribution.log_prob(target)

        # Take the mean across the batch
        loss = torch.mean(loss)

        return loss
            
class NegativeNormalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        #negative loss liklihood for negative binomial distribution
        
    def forward(self, loc, scale, target):
        # Ensure concentration is positive
        loc = torch.abs(loc)
        scale = torch.abs(scale)

        # Define the Weibull distribution
        normal_distribution = dist.Normal(loc, scale)

        # Calculate the negative log likelihood
        loss = -normal_distribution.log_prob(target)

        # Take the mean across the batch
        loss = torch.mean(loss)

        return loss
        

##############################################
# Final Training Function
##############################################

def train(
    train_dataloader,
    valid_dataloader,
    n_epochs,
    criterion,
    optimizer,
    scheduler,
    Tensor,
    early_stop,
    Net,
    iteration,
    folder
):
    losses = []
    validation_losses = []
    validation_steps = []
    
    best_val_loss = float('inf')
    best_model = None
    patience = 3
    epoch=0
    
    # TRAINING
    start_time = time.time()
    
    Net.train()
    # for epoch in range(hp.epoch, n_epochs):
    while True:
        epoch+=1
        training_loss=0        
        for i, batch in enumerate(train_dataloader):
            data, labels = batch
            data.type(Tensor)
            labels.type(Tensor)

            if torch.cuda.is_available():
                data = data.cuda()
                labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # outputs = Net(data)
            out_means, out_std = Net(data)
            # out_means = outputs[:,0]
            # out_std = outputs[:,1]
            
            try:
                loss = criterion(out_means.squeeze(), out_std.squeeze(), labels.squeeze())
            except:
                print('Is there nan in data? ', torch.isnan(data).any())
                print('Is there nan in label? ', torch.isnan(labels).any())
                print(out_means)
                print(out_std)
                raise SystemExit(0) 
            loss.backward()
            optimizer.step()

            time_taken = datetime.timedelta(
                seconds=(time.time() - start_time)
            )

            print(
                "\r[Iteration %d] [Epoch %d] [Batch %d/%d] [ loss: %f] Training time: %s"
                % (
                    iteration+1,
                    epoch,
                    i,
                    len(train_dataloader),
                    (loss.item()),
                    time_taken,
                )
            )
            
            training_loss+=loss.item()

        losses.append(abs(training_loss)/(i+1))
        scheduler.step()
        #evaluate validation model
        Net.eval()
        validation_loss_total=[]
        with torch.no_grad():
            for i, batch in enumerate(valid_dataloader):
                data, labels = batch
                data.type(Tensor)
                labels.type(Tensor)
            
                if torch.cuda.is_available():
                    data = data.cuda()
                    labels = labels.cuda()
                # zero the parameter gradients
                optimizer.zero_grad()
            
                # forward + backward + optimize
                # outputs = Net(data)
                out_means, out_std = Net(data)
                # out_means = outputs[:,0]
                # out_std = outputs[:,1]

                loss = criterion(out_means.squeeze(), out_std.squeeze(), labels.squeeze())
                
                validation_loss_total.append(abs(loss.item()))
        
            validation_losses.append(np.mean(validation_loss_total))
            validation_steps.append(epoch)
            
            print("\r[Epoch %d] [Validation loss %f]" 
                  %(epoch, np.mean(validation_loss_total)))
            
            if np.mean(validation_loss_total) < best_val_loss:
                best_val_loss = np.mean(validation_loss_total)
                best_model = Net.state_dict()  # Save the best model
                best_epoch=epoch
                epochs_without_improvement = 0
                print('New lowest loss')
            else:
                epochs_without_improvement += 1
                print('Epochs without improvement: {epochs_without_improvement}')
                
        # Check if early stopping condition is met
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break     

    save_path, _ = get_save_path(epoch, folder)
    torch.save(best_model, save_path)

    print('Finished Training')
    print('Total training time ', datetime.timedelta(
        seconds=time.time()-start_time))
    
    plot_losses(losses, loss_val=validation_losses, save_path=None)
    
    top_model=validation_steps[np.argmin(validation_losses)]
    print('Best performing model at epoch {}, with loss {}'.
          format(top_model, np.min(validation_losses)))
    
    return best_model

def validation(
    dataloader,
    Tensor,
    Net,
):
    targets = []
    predictions = []
    priors = []
    # TRAINING
    prev_time = time.time()
    Net.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            data, labels = batch
            data.type(Tensor)
            labels.type(Tensor)

            if torch.cuda.is_available():
                data = data.cuda()
                labels = labels.cuda()

            # outputs = Net(data)
            
            # out_means = torch.abs(outputs[:,0])
            # out_std = torch.abs(outputs[:,1])
            
            out_means, out_std = Net(data)
            # out_dists=dist.Weibull(out_means, out_std)
            out_dists=dist.Normal(out_means, out_std)
            
            [targets.append(labels[i].cpu().numpy()) for i in range(
                len(out_means.squeeze().cpu().numpy()))]
            
            [predictions.append(out_dists.mean[i].cpu().numpy()) for i in range(
                len(out_means.squeeze().cpu().numpy()))]
            
            [priors.append(data[i][-1][-1].cpu().numpy()) for i in range(
                len(out_means.squeeze().cpu().numpy()))]

    errors_percent=calculate_errors(np.squeeze(predictions), np.squeeze(targets))
    errors_priors=calculate_errors(np.squeeze(priors), np.squeeze(targets))
    print('Finished Validation')
    return errors_percent, errors_priors, targets, predictions

def validation_metrics(errors_percent, errors_priors, targets, prediction):
    print('')
    print('Priors (reference) error: ', round(np.mean(errors_priors),4),'% Mean', round(np.median(errors_priors),4),'% Median')
    print('Mean error: ', round(np.mean(errors_percent),4),'% Mean', round(np.median(errors_percent),4),'% Median')
    
    plt.figure()
    plt.scatter(x=targets, y=errors_percent, marker='x')
    plt.title('Errors % vs target')
    plt.xlabel('target')
    plt.ylabel('% error')
    plt.ylim((0,100))
    plt.show()
    
    plt.figure()
    plt.hist(errors_priors.flatten(), bins=10000, color='red', label='Priors')
    plt.hist(errors_percent.flatten(), bins=10000, color='blue', alpha=0.8, label='Model')
    plt.title('Histogram of errors')
    plt.xlabel('% error')
    plt.ylabel('Frequency')
    plt.xlim((0,10))
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.hist(errors_priors.flatten(), bins=10000, color='red')
    plt.title('Histogram of errors priors')
    plt.xlabel('% error')
    plt.ylabel('Frequency')
    plt.xlim((0,10))
    plt.show()
    
    plt.figure()
    plt.hist(errors_percent.flatten(), bins=10000, color='blue')
    plt.title('Histogram of errors model')
    plt.xlabel('% error')
    plt.ylabel('Frequency')
    plt.xlim((0,10))
    plt.show()

def calculate_errors(predictions, targets):
    targets=np.array(targets)
    predictions=np.array(predictions)
    differences = np.abs(targets-predictions)

    return 100*(differences/targets)

def get_save_path(indx, folder):
    CURRENT_DIR = os.getcwd()
    BASE_OUTPUT = "saved_models"
    MODEL_PATH = os.path.join(CURRENT_DIR, BASE_OUTPUT, folder, "epoch_{}_".format(indx) + "model.pt")
    MODEL_PATH_BEST = os.path.join(CURRENT_DIR, BASE_OUTPUT, folder, "best_model_epoch_{}_".format(indx) + "model.pt")
    return MODEL_PATH, MODEL_PATH_BEST
    
def plot_roc(true_list, pred_list_raw):
    fpr, tpr, thresholds = metrics.roc_curve(true_list,  pred_list_raw)
    auc = metrics.roc_auc_score(true_list, pred_list_raw)
    
    plt.figure()
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.show()
    

    print('true list len ', len(true_list))
    print('pred list len', len(pred_list_raw))
    print('TPR ', tpr)
    print('FPR', fpr)
    print('Thresholds', thresholds)
    print('Optimum threshold ', thresholds[np.argmax(tpr - fpr)])
    
    return fpr, tpr, auc, thresholds

def plot_losses(loss, loss_val=None, save_path=None):
    with plt.style.context(['science', 'ieee','no-latex', 'bright']):
        plt.rcParams.update({
        "font.family": "serif",   
        "font.serif": ["Times New Roman"],
        "font.size": 12})
        plt.figure(figsize = (7,5))
        plt.plot(loss, label='train')
        if loss_val:
            plt.plot(loss_val, label='validation')
        plt.title('Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.yscale("log")
        plt.legend()
        if save_path:
            plt.save(save_path)
        plt.show()
    
   
def main(HP, train_dataloader, validation_dataloader, test_dataloader, cuda_device, iteration = None):
    
    #[n_epochs, batch_size, lr, momentum, early_stop, conv_layers, out_channel_ratio, FC_layers]
    
    if cuda_device != None:
        torch.cuda.set_device(cuda_device)
 
    global hp

    hp = Hyperparameters(
        epoch=0,
        n_epochs=HP['n_epochs'],
        dataset_train_mode="train",
        dataset_test_mode="test",
        batch_size=HP['batch_size'],
        lr=HP['lr'],
        momentum=HP['momentum'],
        img_size=64,
        no_samples=1000,
        channels=1,
        early_stop=HP['early_stop']
    )

    ##############################################
    # SETUP, LOSS, INITIALIZE MODELS and OPTIMISERS
    ##############################################
    
    Net = CNN()
    
    # Network summary info
    print('Network')
    
    # print(Net)
    print(summary(Net.float(), input_size=(hp.batch_size, 1, 64)))
    
    criterion = NegativeNormalLoss()#NegativeWeibullLoss()#SMAPE()#nn.MSELoss()
    # optimizer = torch.optim.Adam(Net.parameters(), lr=hp.lr) #try sgd
    optimizer = torch.optim.SGD(Net.parameters(), lr=hp.lr) #try sgd
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=hp.lr/20, total_iters=50)
    
    Net = Net.double()
        
    cuda = True if torch.cuda.is_available() else False
    print("Using CUDA" if cuda else "Not using CUDA")
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    
    if cuda:
        Net = Net.cuda()
        criterion = criterion.cuda()
    
    
    ##############################################
    # Execute the Final Training Function
    ##############################################
    folder=HP['folder']

    top_model_epoch = train(
        train_dataloader=train_dataloader,
        valid_dataloader=validation_dataloader,
        n_epochs=hp.n_epochs,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        Tensor=Tensor,
        early_stop=hp.early_stop,
        Net = Net,
        iteration = iteration,
        folder=folder
    )
    
    Net.load_state_dict(top_model_epoch)

    if validation_dataloader != None:
        errors_percent, errors_priors, targets, prediction = validation(
            dataloader=validation_dataloader,
            Tensor=Tensor,
            Net=Net,
        )
    validation_metrics(errors_percent, errors_priors, targets, prediction)
    
    if validation_dataloader != None:
        errors_percent, errors_priors, targets, prediction = validation(
            dataloader=test_dataloader,
            Tensor=Tensor,
            Net=Net,
        )
    validation_metrics(errors_percent, errors_priors, targets, prediction)
    
    return errors_percent, errors_priors, targets, prediction

    

    #[n_epochs, batch_size, lr, momentum, early_stop, conv_layers, out_channel_ratio, FC_layers]
