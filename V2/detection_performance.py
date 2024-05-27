from utils import *
from visualise_volume import *
import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots

plt.style.use(['no-latex'])  

def plot_accuracies():
    thresholds = ['0.9999999', '0.999999', '0.99999', '0.9999', '0.999', '0.99']

    forward = [
        [16.85, 14.79, 7.98],
        [9.2,8.28,5.34],
        [4.66,4.28,3.42],
        [2.17,2.33,2.33],
        [1.31,1.33,1.68],
        [4.95,5.73,4.34]
    ]
    backward = [
        [12.40,21.55,10.49],
        [7.54,13.30,6.91],
        [3.83,5.94,3.99],
        [2.02,2.76,2.34],
        [1.36,1.39,1.71],
        [4.79,3.81,3.60]

    ]
    combined = [
        [39.47,55.56,22.73],
        [28.85,42.37,16.67],
        [12.83,20.49,10.79],
        [5.68,7.99,5.88],
        [1.89,2.32,2.85],
        [1.51,1.30,1.62,]

    ]
    thresholded = [
        [100,100,100],
        [93.75,100,93.75],
        [83.33,96.15,88.24],
        [71.43,92.59,78.95],
        [65.22,89.29,84.62],
        [50,96.15,60]
    ]

    # Calculate the mean for each method
    methods = ['Forward\nSweep', 'Backward\nSweep', 'Combined', 'Area\nThreshold']

    plt.figsize = (5, 5)
    #set font size
    plt.rcParams.update({'font.size': 14})
    #set font style
    plt.rcParams.update({'font.family': 'Times New Roman'})    # Plot a line for each threshold
    for i, threshold in enumerate(thresholds):
        means = [np.mean(method[i]) for method in [forward, backward, combined, thresholded]]
        plt.plot(methods, means, label=threshold, marker='x', linestyle='--')

    plt.ylabel('Mean Sample Detection Accuracy (%)')
    plt.legend(title='Thresholds')
    plt.savefig(r'C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\PhD\Publications\Journal\SSL\media\detection.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_accuracies()
    # exit()
    # sample = 'ID012/combined'
    sample = 'ID009'
    # sample = 'ID018'
    threshold = 0.99
    path = r'C:\GIT\Self-supervised-volumetric-detection\V2\inference'
    volume = (np.load(os.path.join(path, sample, 'test_volume.npy')))
    forward = (np.load(os.path.join(path, sample, f'{str(threshold)}_forward.npy')))
    backward = (np.load(os.path.join(path, sample, f'{str(threshold)}_backward.npy')))
    combined = (np.load(os.path.join(path, sample, f'{str(threshold)}_combined.npy')))
    thresholded = (np.load(os.path.join(path, sample, f'{str(threshold)}_thresholded.npy')))
    sample = str.lower(sample.split('/')[0])
    if sample == 'id018':
        volume = volume[30:, :, :]
        forward = forward[30:, :, :]
        backward = backward[30:, :, :]
        combined = combined[30:, :, :]
        thresholded = thresholded[30:, :, :]
    elif sample == 'id009':
        pass
    elif sample == 'id012':
        volume = np.swapaxes(volume[:488, :105, 12:124], 0, 2)
        forward = forward[12:124, :105, :488]
        backward = backward[12:124, :105, :488]
        combined = combined[12:124, :105, :488]
        thresholded = thresholded[12:124, :105, :488]

    print(volume.shape, forward.shape, backward.shape, combined.shape, thresholded.shape)

    forward_mask = np.amax(forward, axis=1)
    backward_mask = np.amax(backward, axis=1)
    combined_mask = np.amax(combined, axis=1)
    thresholded_mask = np.amax(thresholded, axis=1)

    steps = {'Forward Sweep':forward_mask, 
             'Backward Sweep':backward_mask, 
             'Combined Sweeps':combined_mask, 
             'Area Thresholded':thresholded_mask}

    metrics = [

    ]

    db_18 = db_masks(sample, volume)
    db_18.get_coords()
    db_18.gen_db_mask()
    db_18.get_centroids()
    db_18.get_sizes()

    # db_18.plot()
    db_18.plot(forward_mask)
    db_18.gen_true_mask()
    db_18.plot(db_18.true_mask)
    
    for step_label, step in steps.items():
        print(step_label)
        ious = db_18.get_iou(step)
        accuracy, thresholds = db_18.get_accuracy_vs_threshold(ious)
        print("")

    # ious = db_18.get_iou(db_18.db_mask)
    # db_18.plot(db_18.db_mask)

    # db_18.plot_precision_recall(ious)
    # db_18.plot_roc(ious)
    # db_18.plot_precision_recall_vs_threshold(ious)
    # db_18.plot_tp_fp_vs_threshold(ious)
    # accuracy, thresholds = db_18.get_accuracy_vs_threshold(ious)
    # metrics.append({
    #         'thresholds': thresholds,
    #         'accuracy': accuracy,
    #         'label': 'Accuracy'
    #     })
    # f1s, thresholds = db_18.get_f1_vs_threshold(ious)
    # metrics.append({
    #         'thresholds': thresholds,
    #         'accuracy': f1s,
    #         'label': 'F1'
    #     })
    
    # db_18.plot_metrics_vs_threshold(metrics)