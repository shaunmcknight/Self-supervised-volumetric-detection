import numpy as np
import time
import datetime
import os
import torch
import torch.distributions as dist
import torch.nn as nn
from torchinfo import summary

import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm
from sklearn import metrics 

from config import hp
from classifierv2 import CNN

plt.style.use(['science', 'ieee','no-latex', 'bright'])

##############################################
# Defining all hyperparameters
##############################################
        
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

        # Define the distribution
        normal_distribution = dist.Normal(loc, scale)

        # Calculate the negative log likelihood
        loss = -normal_distribution.log_prob(target)

        # Take the mean across the batch
        loss = torch.mean(loss)

        return loss

class Trainer:
    def __init__(self, dataloader_train, dataloader_validation, dataloader_test, iteration):
        self.iteration = iteration
        self.dataloader_train = dataloader_train
        self.dataloader_validation = dataloader_validation
        self.dataloader_test = dataloader_test
        self.epochs_without_improvement = None
        self.training_losses = []
        self.validation_losses = []

        self.Net = CNN()
        print(summary(self.Net, input_size=(hp.batch_size, 1, 64)))
        self.criterion = NegativeNormalLoss()
        self.optimizer = torch.optim.Adam(self.Net.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
        self.cuda = True if torch.cuda.is_available() else False
        print("Using CUDA" if self.cuda else "Not using CUDA")
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        if self.cuda:
            self.Net = self.Net.float().cuda()
            self.criterion = self.criterion.cuda()

    def train(self):
        # TRAINING
        self.start_time = time.time()
        self.Net.train()
        epoch=0
        best_val_loss = float('inf')
        self.early_stop = False
        while self.early_stop == False:
            epoch+=1
            training_loss=[]
            pbar = tqdm(self.dataloader_train, total=len(self.dataloader_train), desc=f'Epoch {epoch}')
            for batch in pbar:
                self.optimizer.zero_grad()
                data, labels = self.prepare_data(batch)
                out_means, out_std = self.Net(data)
                
                try:
                    loss = self.criterion(out_means.squeeze(), out_std.squeeze(), labels.squeeze())
                except Exception as error:  
                    print('Is there nan in data? ', torch.isnan(data).any())
                    print('Is there nan in label? ', torch.isnan(labels).any())
                    print(error)
                    exit()
                    
                loss.backward()
                self.optimizer.step()


                training_loss.append(loss.item())
                minutes, seconds = divmod(time.time() - self.start_time, 60)
                pbar.set_postfix({'loss': loss.item(), 'elapsed_time': f'{int(minutes)}m {int(seconds)}s'}, refresh=True)
                pbar.update()
                
            self.training_losses.append(np.mean(training_loss))

            validation_loss = self.test(self.Net, self.dataloader_validation) # Evaluate the model on the validation set after each epoch

            self.validation_losses.append(validation_loss)
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                best_model = self.Net.state_dict()  # Save the best model
                self.epochs_without_improvement = 0
                print('New lowest loss {} at epoch {}'.format(best_val_loss, epoch))
            else:
                self.epochs_without_improvement += 1
                print('Epochs without improvement: {epochs_without_improvement}')
                    
            self.early_stop = self.early_stop(epoch) 

        torch.save(best_model, os.join(hp.save_path, f'{str(self.iteration)}_best_model.pth'))
        self.Net.load_state_dict(best_model)
        print('Finished Training')
        self.print_training_time()
    
    def test(self, Net, datloader, label='Test'):
        prev_time = time.time()
        loss_total = []
        Net.eval()
        with torch.no_grad():
            for batch in tqdm(datloader, total=len(datloader), desc=f'{label}'):
                data, labels = self.prepare_data(batch)            
                out_means, out_std = Net(data)

                loss = self.criterion(out_means.squeeze(), out_std.squeeze(), labels.squeeze())
                
                loss_total.append(loss.item())
        loss_total = np.mean(loss_total)
        self.validation_losses.append(loss_total)
        self.print_training_time(prev_time, label)
        return loss_total

    def early_stop(self, epoch):
        if self.epochs_without_improvement >= hp.patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            return True
        return False

    def print_training_time(self, prev_time=None, label = ''):
        if prev_time:
            print('Total training time ', datetime.timedelta(seconds=time.time()-self.start_time))
        else:
            print(f'{label} time taken ', datetime.timedelta(seconds=time.time()-prev_time))

    def plot_training_loss(self):
        self.plot_losses(self.training_losses, loss_val=self.validation_losses, save_path=None)

    def prepare_data(self, batch):
        data, labels = batch
        data = data.type(self.Tensor)
        labels = labels.type(self.Tensor)
        if self.cuda:
            data = data.cuda()
            labels = labels.cuda()
        return data, labels
    
    def plot_losses(self, train_loss, loss_val=None, save_path=None):
        with plt.style.context(['science', 'ieee','no-latex', 'bright']):
            plt.rcParams.update({
            "font.family": "serif",   
            "font.serif": ["Times New Roman"],
            "font.size": 12})
            plt.figure(figsize = (7,5))
            plt.plot(train_loss, label='train')
            if loss_val:
                plt.plot(loss_val, label='validation')
            plt.title('Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale("log")
            plt.legend()
            if save_path:
                plt.savefig(save_path)
            plt.show()
