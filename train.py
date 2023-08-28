import numpy as np
import itertools
import time
import datetime
import math
from pathlib import Path


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


##############################################
# Final Training Function
##############################################

def train(
    train_dataloader,
    n_epochs,
    criterion,
    optimizer,
    Tensor,
    early_stop,
    Net,
    iteration
):
    losses = []
    # TRAINING
    prev_time = time.time()
    for epoch in range(hp.epoch, n_epochs):
        for i, batch in enumerate(train_dataloader):
            images, labels = batch
            print(type())
            images.type(Tensor)
            labels.type(Tensor)

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = Net(images)
            loss = criterion(outputs.squeeze(), labels)
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

    print('Finished Training')

    plt.figure()
    plt.plot(losses)
    plt.title('Network Losses (Batch average)')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


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


def showFailures(true_list, pred_list, dataloader):
    list_incorrect_defects = []
    list_incorrect_clean = []
    for i, item in enumerate(pred_list):
        if item != true_list[i]:
            if item == 1:
                list_incorrect_defects.append(
                    dataloader.dataset[i][0].squeeze())
            elif item == 0:
                list_incorrect_clean.append(
                    dataloader.dataset[i][0].squeeze())
            else:
                print('ERROR')

    if len(list_incorrect_defects) > 0:
        plotGrid(list_incorrect_defects,
                 'Grid of incorrect defect predicitions')

    if len(list_incorrect_clean) > 0:
        plotGrid(list_incorrect_clean,
                 'Grid of incorrect clean predicitions')


def plotGrid(img_list, title):
    grid_dim = math.floor((len(img_list))**(1/2))
    fig = plt.figure()
    plt.suptitle(title)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     # creates 2x2 grid of axes
                     nrows_ncols=(grid_dim, math.ceil(len(img_list)/grid_dim)),
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for ax, im in zip(grid, img_list):
        ax.imshow(im)
        ax.axis('off')
    plt.show()
        
def visualiseFeatures(model, image, target_class):
    #from
    #https://www.coderskitchen.com/guided-backpropagation-with-pytorch-and-tensorflow/
    
    # backprop = Backprop(model)
        
    # # guided_gradients = backprop.calculate_gradients(image, target_class, guided = True)
    
    # backprop.visualize(image, 1, guided = True)
    
    def plot_maps(img1, img2,vmin=0.3,vmax=0.5, mix_val=1.5):
        # f = plt.figure(figsize=(15,45))
        plt.figure(figsize=(6,2))
        plt.subplot(1,3,1)
        plt.imshow(img2, cmap = "viridis")
        # plt.title('Image')
        plt.grid(False)
        # plt.axis("off")
        plt.subplot(1,3,2)
        plt.imshow(img1,vmin=vmin, vmax=vmax, cmap="viridis")
        # plt.title('Grad-CAM')
        # plt.axis("off")
        plt.grid(False)
        plt.subplot(1,3,3)
        # plt.imshow(img1*mix_val+img2/mix_val, cmap = "gray" )
        plt.imshow(img1*mix_val+img2, cmap = "viridis" )
        # plt.title('Mixed image')
        # plt.axis("off")
        plt.grid(False)
        plt.show()
        
    def norm_flat_image(img):
        grads_norm = img[0]
        grads_norm = grads_norm.detach().numpy().transpose(1, 2, 0)
    
        grads_norm = (grads_norm - np.min(grads_norm))/ (np.max(grads_norm)- np.min(grads_norm))
        return grads_norm
    
    def relu_hook_function(module, grad_in, grad_out):
        if isinstance(module, torch.nn.ReLU):
            return (torch.clamp(grad_in[0], min=0.),)
    
    for i, module in enumerate(model.modules()):
        if isinstance(module, torch.nn.ReLU):
            # print(model.named_modules())
            module.register_backward_hook(relu_hook_function)
            
    image.requires_grad = True
    # forward/inference
    out = model(image).backward()
    # best_id = decode_output(out)
    # backprop
    # out[0, target_class].backward()
    grads = image.grad
    
    plot_maps(norm_flat_image(grads),norm_flat_image(image))

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
    
def plot_threshold(fpr, tpr, thresholds):
    plt.figure()
    plt.plot(thresholds,fpr,label="FPR")
    plt.plot(thresholds,tpr,label="TPR")
    plt.ylabel('True Positive Rate')
    plt.xlabel('Threshold')
    plt.xlim(0,1)
    plt.legend()
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
    
    input_shape = (hp.channels, hp.img_size, hp.no_samples, hp.img_size)
    
    # Net = Network(input_shape)
    
    Net = CNN3D(input_shape, conv_layers=HP['conv_layers'],
                         kernel_size=3, out_channel_ratio=HP['out_channel_ratio'],
                         FC_layers=HP['FC_layers'])
    
    # Network summary info
    print('Network')
    
    # print(Net)
    print(summary(Net.float(), input_size=(hp.batch_size, hp.channels, hp.img_size, hp.no_samples, hp.img_size)))
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(Net.parameters(), lr=hp.lr, momentum=hp.momentum)
    
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
    
    train(
        train_dataloader=train_dataloader,
        n_epochs=hp.n_epochs,
        criterion=criterion,
        optimizer=optimizer,
        Tensor=Tensor,
        early_stop=hp.early_stop,
        Net = Net,
        iteration = iteration
    )
        
    if validation_dataloader != None:
        test(validation_dataloader, 'Validation Test', disp_CM=True, Net = Net, Tensor=Tensor)
    true_list, pred_list, pred_list_raw, accuracy, precision, recall, f_score, cm = test(test_dataloader,
                                'Experimental Test', disp_CM=True, Net = Net, Tensor=Tensor)
    fpr, tpr, auc, thresholds = plot_roc(true_list, pred_list_raw)
    plot_threshold(fpr, tpr, thresholds)
    
    plt.figure()
    plt.scatter(true_list, pred_list_raw, marker ='x')
    plt.show()
    
    # showFailures(true_list, pred_list, test_dataloader)
    
    img_idx = 2
    for img_idx in range(2):
        data = next(iter(test_dataloader))
        img = data[0][img_idx].unsqueeze(1)
        target_class = int(data[1][img_idx]) 
        target_class = 1
        visualiseFeatures(Net, img, target_class)
    
    return accuracy, precision, recall, f_score, cm

    #[n_epochs, batch_size, lr, momentum, early_stop, conv_layers, out_channel_ratio, FC_layers]
