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
import torch
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
        

##############################################
# Final Training Function
##############################################

def train(
    train_dataloader,
    valid_dataloader,
    n_epochs,
    criterion,
    optimizer,
    Tensor,
    early_stop,
    Net,
    iteration,
    folder
):
    losses = []
    validation_losses = []
    validation_steps = []
    # TRAINING
    start_time = time.time()
    prev_time = time.time()
    
    Net.train()
    for epoch in range(hp.epoch, n_epochs):
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
            outputs = Net(data)

            loss = criterion(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()

            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + i
            batches_left = n_epochs * len(train_dataloader) - batches_done

            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time)
            )

            print(
                "\r[Iteration %d] [Epoch %d/%d] [Batch %d/%d] [ loss: %f] ETA: %s"
                % (
                    iteration+1,
                    epoch,
                    n_epochs,
                    i,
                    len(train_dataloader),
                    np.mean(loss.item()*hp.batch_size),
                    time_left,
                )
            )

            losses.append(np.mean(loss.item()*hp.batch_size))

            prev_time = time.time()

        if (np.mean(loss.item()*hp.batch_size)) < early_stop:
            break
        
        Net.eval()
        validation_loss_total=0
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
                outputs = Net(data)
            
                loss = criterion(outputs.squeeze(), labels.squeeze())
    
                validation_loss_total+=np.mean(loss.item()*hp.batch_size)
        
        validation_losses.append(validation_loss_total)
        validation_steps.append(epoch)
        
        save_path, _ = get_save_path(epoch, folder)
        torch.save(Net, save_path)

        

    print('Finished Training')
    print('Total training time ', datetime.timedelta(
        seconds=time.time()-start_time))

    plt.figure()
    plt.plot(losses)
    plt.title('Network Losses (Batch average)')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()
    
    plt.figure()
    plt.plot(validation_steps, validation_losses)
    plt.title('Validation Losses (Batch average)')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.yscale("log")
    # plt.ylim((0,10))
    plt.show()
    
    plt.figure()
    plt.plot(losses)
    # plt.ylim((0,1.0))
    plt.title('Network Losses limited (Batch average)')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.yscale("log")
    plt.show()
    
    top_model=validation_steps[np.argmin(validation_losses)]
    print('Best performing model at epoch {}, with loss {}'.
          format(top_model, np.min(validation_losses)))
    
    return top_model

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

            # forward + backward + optimize
            outputs = Net(data)
            
            [targets.append(labels[i].cpu().numpy()) for i in range(
                len(labels.squeeze().cpu().numpy()))]
            
            [predictions.append(outputs[i].cpu().numpy()) for i in range(
                len(outputs.squeeze().cpu().numpy()))]
            
            # [print(data[i][-1].shape) for i in range(
            #     len(outputs.squeeze().cpu().numpy()))]
            [priors.append(data[i][-1][-1].cpu().numpy()) for i in range(
                len(outputs.squeeze().cpu().numpy()))]

    errors_percent=calculate_errors(np.squeeze(predictions), np.squeeze(targets))
    errors_priors=calculate_errors(np.squeeze(priors), np.squeeze(targets))
    print('Finished Validation')
    return errors_percent, errors_priors, targets, predictions

def validation_metrics(errors_percent, errors_priors, targets, prediction):
    print('Reference error: ', round(np.mean(errors_priors),4),'% Mean', round(np.median(errors_priors),4),'% Median')
    print('Mean error: ', round(np.mean(errors_percent),4),'% Mean', round(np.median(errors_percent),4),'% Median')
    
    plt.figure()
    plt.scatter(x=targets, y=errors_percent, marker='x')
    plt.title('Errors % vs target')
    plt.xlabel('target')
    plt.ylabel('% error')
    plt.ylim((0,20))
    plt.show()
    
    plt.figure()
    plt.hist(errors_priors.flatten(), bins=10000, color='red')
    plt.hist(errors_percent.flatten(), bins=10000, color='blue', alpha=0.8)
    plt.title('Histogram of errors')
    plt.xlabel('% error')
    plt.ylabel('Frequency')
    plt.xlim((0,10))
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
    
def test(dataloader, description, disp_CM, Net, Tensor):
    true_list = []
    pred_list = []
    pred_list_raw = []
    Net.eval()
    Net.cpu()  # cuda()
    images = []
    labels = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, labels = batch
            images.type(Tensor)
            labels.type(Tensor)

            if torch.cuda.is_available():
                images = images.cpu()  # cuda()
                labels = labels.cpu()  # cuda()

            for i in range(len(labels.numpy())):
                true_list.append(labels.numpy()[i])

            output_raw = Net(images)
            output = torch.sigmoid(output_raw)
            pred_tag = torch.round(output)
            
            [pred_list.append(pred_tag[i]) for i in range(
                len(pred_tag.squeeze().cpu().numpy()))]
            
            [pred_list_raw.append(output[i]) for i in range(
                len(output.squeeze().cpu().numpy()))]
            
    pred_list = [a.squeeze().tolist() for a in pred_list]
    pred_list_raw = [a.squeeze().tolist() for a in pred_list_raw]

    true_list = np.array(true_list)
    pred_list = np.array(pred_list)
    pred_list_raw = np.array(pred_list_raw)

    correct = np.sum(true_list == pred_list)
    total = np.shape(true_list)
    
    accuracy = correct/total

    print('')
    print('~~~~~~~~~~~~~~~~~')
    print(description)
    print('Prediciton Accuracy: ', (accuracy)*100)

    print('Confusion matrix || {}'.format(description))
    cm = metrics.confusion_matrix(true_list, pred_list)
    print(cm)

    if disp_CM == True:
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['No defect', 'Defect'])
        disp.plot()
        disp.ax_.set_title(description)
        plt.show()        
        

    precision, recall, f_score, support = metrics.precision_recall_fscore_support(
        true_list, pred_list)

    print('Precision ', precision[1])
    print('Recall ', recall[1])
    print('F score ', f_score[1])

    return true_list, pred_list, pred_list_raw, accuracy, precision[1], recall[1], f_score[1], cm

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
    
    criterion = SMAPE()#nn.MSELoss()
    optimizer = torch.optim.Adam(Net.parameters(), lr=hp.lr)
    
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
        Tensor=Tensor,
        early_stop=hp.early_stop,
        Net = Net,
        iteration = iteration,
        folder=folder
    )
    
    top_model_path, _ = get_save_path(top_model_epoch, folder)
    top_model = torch.load(top_model_path)
    
    #resave bets model
    _, best_model_path = get_save_path(top_model_epoch, folder)
    torch.save(top_model, best_model_path)


    if validation_dataloader != None:
        errors_percent, errors_priors, targets, prediction = validation(
            dataloader=validation_dataloader,
            Tensor=Tensor,
            Net=top_model,
        )
    validation_metrics(errors_percent, errors_priors, targets, prediction)
    
    if validation_dataloader != None:
        errors_percent, errors_priors, targets, prediction = validation(
            dataloader=test_dataloader,
            Tensor=Tensor,
            Net=top_model,
        )
    validation_metrics(errors_percent, errors_priors, targets, prediction)
    
    return errors_percent, errors_priors, targets, prediction

    

    #[n_epochs, batch_size, lr, momentum, early_stop, conv_layers, out_channel_ratio, FC_layers]
