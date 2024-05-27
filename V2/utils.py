import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score
import math

class db_masks():
    def __init__(self, sample='id009', volume=None):
        self.sample = sample
        self.volume = volume  
        self.db_mask = np.zeros((volume.shape[0], volume.shape[2]))
        self.true_mask = np.zeros((volume.shape[0], volume.shape[2]))
        self.coords = None
        self.centers = None
        self.sizes = None
        #get centroids
        #get true size
        #return centroids, db mask, true mask size
                
    def gen_6db_threshold(self, idx):
        volume = self.volume[idx[0]:idx[0]+25,idx[2]:idx[2]+5,idx[1]:idx[1]+25]
        c_scan = np.amax(volume, axis=1)
        db_threshold = np.ones(c_scan.shape)
        db_threshold[c_scan<(np.max(c_scan))*0.5]=0
        return db_threshold

    def gen_true_mask(self):
        for i, center in enumerate(self.centers):
            width = self.sizes[i]
            if self.sample == 'id009' or self.sample == 'id018':
                self.gen_circle(center, width)
            if self.sample == 'id012':
                self.gen_square(center, width)
    
    def gen_circle(self, center, diameter):
        x, y = center[1], center[0]
        diameter = diameter / 0.8

        # Create a grid of coordinates
        Y, X = np.ogrid[:self.true_mask.shape[0], :self.true_mask.shape[1]]

        # Create a mask where the condition is True
        mask = (X - x)**2 + (Y - y)**2 <= (diameter/2)**2
        self.true_mask[mask] = 1
    
    def gen_square(self, center, width):
        x, y = (center[0], center[1])
        width = width/0.8
        half_width = width / 2

        lower_x = int(round(x - half_width))
        upper_x = int(round(lower_x + width))
        lower_y = int(round(y - half_width))
        upper_y = int(round(lower_y + width))

        self.true_mask[lower_x:upper_x, lower_y:upper_y] = 1
        
    def get_centroids(self):
        labeled_volume, num_labels = label(self.db_mask, connectivity=2, return_num=True)
        properties = regionprops(labeled_volume)
        centers = [prop.centroid for prop in properties]
        self.centers = centers

    def get_sizes(self):
        #coords in format [x,y,z]
        print(self.sample)
        if self.sample == 'id009':
            self.sizes = [
                9,6,3,
                9,6,3,
                9,6,3,
                9,6,3,
                9,6,3,                 
            ]
        if self.sample == 'id018':
            self.sizes = [
                9,7,4,6,3,
                9,7,6,4,3,
                9,7,4,6,3,
                9,7,6,4,3,
                9,7,4,6,3,
            ]
        if self.sample == 'id012':
            self.sizes = [
                6,6,6,6,6,6,6,6,6,6,6,6,6,6,6
            ]

    def get_MAE_gt(self):
        if self.sample == 'id009' or self.sample == 'id018':
            labeled_volume, _ = label(self.true_mask, connectivity=2, return_num=True)
            properties = regionprops(labeled_volume)
            areas = [prop.area for prop in properties]
            diameters = [0.8*2*(area/math.pi)**0.5 for area in areas]
            print(diameters)
            differences = [abs(sorted(diameters)[i] - sorted(self.sizes)[i]) for i in range(len(diameters))]   
            MAE = np.mean(differences) #diameter mean error in mm
        elif self.sample == 'id012':
            labeled_volume, _ = label(self.true_mask, connectivity=2, return_num=True)
            properties = regionprops(labeled_volume)
            areas = [prop.area for prop in properties]
            widths = [0.8*(area)**0.5 for area in areas]
            print(diameters)
            differences = [abs(sorted(widths)[i] - sorted(self.sizes)[i]) for i in range(len(widths))]   
            MAE = np.mean(differences) #diameter mean error in mm
        return MAE
            
    def get_coords(self):
            #coords in format [x,y,z]
            if self.sample == 'id009':
                self.coords = [
                    [50, 21, 21],     
                    [90, 20, 31],     
                    [130, 20, 40],     
                    [175, 20, 50],     
                    [220, 20, 60],  

                    [50, 80, 21],     
                    [90, 80, 31],     
                    [130, 80, 40],     
                    [175, 80, 52],     
                    [220, 80, 60],  

                    [50, 140, 22],     
                    [90, 140, 31],     
                    [130, 140, 40],     
                    [175, 140, 52],     
                    [220, 140, 63],     
                ]
            if self.sample == 'id018':
                self.coords = [
                    [20, 20, 60],     
                    [64, 20, 50],     
                    [110, 20, 40],     
                    [151, 20, 30],     
                    [194, 20, 20],  

                    [20, 80, 60],     
                    [64, 80, 50],     
                    [105, 83, 40],     
                    [151, 80, 30],     
                    [194, 80, 20], 

                    [20, 142, 60],     
                    [64, 142, 50],     
                    [105, 142, 40],     
                    [151, 142, 30],     
                    [194, 142, 20],  

                    [20, 206, 60],     
                    [64, 206, 50],     
                    [105, 206, 40],     
                    [151, 206, 30],     
                    [194, 206, 20],  

                    [20, 270, 60],     
                    [64, 270, 50],     
                    [105, 270, 40],     
                    [151, 270, 30],     
                    [194, 270, 20],  
                ]
            if self.sample == 'id012':
                self.coords = [
                    [0, 20, 60],     
                    [37, 20, 40],     
                    [73, 20, 22],     

                    [16, 138, 21],     
                    [51, 138, 48],     
                    [88, 138, 76],     

                    [0, 236, 76],     
                    [37, 236, 48],     
                    [73, 236, 21],     

                    [10, 327, 90],     
                    [44, 327, 59],     
                    [80, 327, 22],     

                    [10, 390, 22],     
                    [44, 390, 59],     
                    [80, 390, 90],  
                ]
        
    def gen_db_mask(self): 
        for coord in self.coords:
            db_threshold = self.gen_6db_threshold(coord)
            self.db_mask[coord[0]:coord[0]+25, coord[1]:coord[1]+25] = db_threshold

    def get_iou(self, predicted_mask = None):
        ious_total = []
        predictions = label(predicted_mask, connectivity=2)
        ground_truths = label(self.true_mask, connectivity=2)
        for i in range(1, predictions.max() + 1):
            ious = []
            prediction_mask = (predictions == i)
            for j in range(1, ground_truths.max() + 1):
                ground_truth = (ground_truths == j)    
                # iou = np.sum(np.logical_and(prediction_mask, ground_truth)) / np.sum(np.logical_or(prediction_mask, ground_truth))
                iou = np.sum(np.logical_and(prediction_mask, ground_truth))
                ious.append(iou)
                # plt.figure()
                # plt.imshow(prediction_mask)
                # plt.imshow(ground_truth, alpha=0.3)
                # plt.show()
            ious_total.append(max(ious))
        return np.array(ious_total)
                
    def plot_precision_recall(self, ious):
        ious = np.array(ious)
        truth = np.zeros(len(ious))

        truth[ious>0] = 1
        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(truth, ious)
        auprc = auc(recall, precision)
        # Plot precision-recall curve
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.show()

    def plot_roc(self, ious):
        ious = np.array(ious)
        truth = np.zeros(len(ious))
        truth[ious>0] = 1
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(truth, ious)
        aucroc = auc(fpr, tpr)
        # Plot ROC curve
        plt.plot(fpr, tpr, marker='.')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.grid(True)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.show()

    def plot_precision_recall_vs_threshold(self, ious):
        ious = np.array(ious)
        truth = np.zeros(len(ious))
        truth[ious>0] = 1

        fpr, tpr, thresholds = roc_curve(truth, ious)
        
        plt.plot(thresholds, tpr, "b--", label="TPR", linewidth=2)
        plt.plot(thresholds, fpr, "g-", label="FPR", linewidth=2)
        plt.xlabel("Threshold")
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def plot_tp_fp_vs_threshold(self, ious):
        ious = np.array(ious)
        truth = np.zeros(len(ious))
        truth[ious>0] = 1

        precisions, recalls, thresholds = precision_recall_curve(truth, ious)

        plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
        plt.xlabel("Threshold")
        # plt.plot([threshold_80_precision, threshold_80_precision], [0., 0.8], "r:")
        # plt.plot([-4, threshold_80_precision], [0.8, 0.8], "r:")
        # plt.plot([-4, threshold_80_precision], [recall_80_precision, recall_80_precision], "r:")
        # plt.plot([threshold_80_precision], [0.8], "ro") 
        # plt.plot([threshold_80_precision], [recall_80_precision], "ro")
        plt.grid(True)
        plt.legend()
        plt.show()

    def get_accuracy_vs_threshold(self, ious):
        ious = np.array(ious)
        truth = np.zeros(len(ious))
        truth[ious>0] = 1
        print(np.sum(truth), len(ious)) 
        thresholds = np.linspace(0, 1, 100)
        accuracies = []
        for threshold in thresholds:
            prediction = np.zeros(len(ious))
            prediction[ious>=threshold] = 1
            accuracy = np.sum(prediction == truth) / len(truth)
            accuracies.append(accuracy)

        pred_iou = ious
        pred_iou[pred_iou>0] = 1   

        print(f'Accuracy: {round(accuracies[0]*100,2)}%')
        print(f'False poitive: {np.sum(ious==0)}')

        # plt.plot(thresholds, accuracies, "b--", label="Accuracy", linewidth=2)
        # plt.xlabel("Threshold")
        # plt.grid(True)
        # plt.legend()
        # plt.show()

        return accuracies, thresholds
    
    def get_f1_vs_threshold(self, ious):
        ious = np.array(ious)
        truth = np.zeros(len(ious))
        truth[ious>0] = 1
        thresholds = np.linspace(0, 1, 100)
        f1s = []
        for threshold in thresholds:
            prediction = np.zeros(len(ious))
            prediction[ious>=threshold] = 1
            f1 = f1_score(truth, prediction)    
            f1s.append(f1)

        # plt.plot(thresholds, f1s, "b--", label="F1", linewidth=2)
        # plt.xlabel("Threshold")
        # plt.grid(True)
        # plt.legend()
        # plt.show()

        return f1s, thresholds
    
    def plot_metrics_vs_threshold(self, metrics=None):
        if metrics is not None:
            plt.figure()
            for metric in metrics:
                plt.plot(metric['thresholds'], metric['accuracy'], label=metric['label'], linewidth=2)
            plt.xlabel("Threshold")
            plt.grid(True)
            plt.legend()
            plt.show()


    def plot(self, mask=None):
        plt.figure()
        plt.imshow(np.amax(self.volume[:,25:,:], axis=1))
        if mask is not None:
            plt.imshow(mask, alpha=0.3)
        if self.centers:
            for center in self.centers:
                plt.plot(center[1], center[0], 'rx')
        plt.show()

    # plt.figure()
    # plt.imshow(self.volume[:,:,336])
    # plt.show()
    
if __name__ == '__main__':
    path = r'C:\GIT\Self-supervised-volumetric-detection\V2\inference\ID009'
    volume = (np.load(os.path.join(path, 'test_volume.npy')))
    db_9 = db_masks(sample = 'id009', volume = volume)
    # db_9.plot()
    # db_9.get_coords()
    # db_9.gen_db_mask()
    # db_9.get_centroids()
    # db_9.get_sizes()
    # db_9.gen_true_mask()
    # db_9.plot('true')

    path = r'C:\GIT\Self-supervised-volumetric-detection\V2\inference\ID018'
    volume = (np.load(os.path.join(path, 'test_volume.npy'))[30:, :, :])
    db_18 = db_masks('id018', volume)
    # db_18.get_coords()
    # db_18.gen_db_mask()
    # db_18.get_centroids()
    # db_18.get_sizes()
    # # db_18.plot()
    # db_18.gen_true_mask()
    # db_18.plot('true')
    # print([round(iou, 3) for iou in db_18.get_iou(db_18.db_mask)])
    # db_18.plot_precision_recall(np.array(db_18.get_iou(db_18.db_mask)))

    path = r'C:\GIT\Self-supervised-volumetric-detection\V2\inference\ID012\combined'
    volume = (np.load(os.path.join(path, 'test_volume.npy')))
    # volume = np.swapaxes(volume, 0, 2)
    volume = np.swapaxes(volume[:488, :105, 12:124], 0, 2)
    db_12 = db_masks('id012', volume)
    db_12.get_coords()
    db_12.gen_db_mask()
    db_12.get_centroids()
    db_12.get_sizes()
    db_12.gen_true_mask()
    db_12.plot(db_12.true_mask)
