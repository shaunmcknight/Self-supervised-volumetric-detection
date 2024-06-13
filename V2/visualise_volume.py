from PyQt5.QtWidgets import QGridLayout, QSlider, QMainWindow, QWidget, QLabel, QApplication, QVBoxLayout, QPushButton, QFileDialog, QLineEdit
from PyQt5.QtGui import QVector3D, QColor
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph import QtCore
import numpy as np
import sys
import os
import scienceplots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.measure import label, regionprops
from scipy import ndimage


class VisualiseVolume(QMainWindow):
    def __init__(self):
        super(VisualiseVolume, self).__init__()
        self.setWindowTitle("Volumetric Visualisation")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)  # Use QVBoxLayout instead of QGridLayout
        self.central_widget.setStyleSheet("background-color: black;")
        
        self.init_camera = 0

        # self.distance = 324  # Initial distance value
        # self.azim = -10
        # self.elev = 102
        # self.roll = -174
        # self.pos = QVector3D(20.65146255493164, 13.584249496459961, 29.060409545898438)

        self.distance = 630  # Initial distance value
        self.azim = 14-5
        self.elev = 70#116
        self.roll = 18
        self.pos = QVector3D(205.58238220214844, -116.35991668701172, -77.59249114990234)

        # Create 3D plot
        self.plot_c_scan_3d = gl.GLViewWidget()

        self.layout.addWidget(self.plot_c_scan_3d)  # Add the 3D plot first
        self.plot_c_scan_3d.setBackgroundColor(QColor('white'))  # Set the background color to white

        # Create sliders for elevation and azimuth
        self.slider_elev = QSlider(QtCore.Qt.Horizontal)
        self.slider_elev.setRange(-180, 180)  # Range from -90 to 90 degrees
        self.slider_elev.setValue(self.elev)  # Initial value
        self.slider_elev.valueChanged.connect(self.update_view)  # Connect to update_camera method

        self.slider_azim = QSlider(QtCore.Qt.Horizontal)
        self.slider_azim.setRange(-180, 180)  # Range from 0 to 360 degrees
        self.slider_azim.setValue(self.azim)  # Initial value
        self.slider_azim.valueChanged.connect(self.update_view)  # Connect to update_camera method

        self.slider_roll = QSlider(QtCore.Qt.Horizontal)
        self.slider_roll.setRange(-180, 180)  # Set the range of the slider to [-180, 180]
        self.slider_roll.setValue(self.roll)  # Set the initial value of the slider to 0
        self.slider_roll.valueChanged.connect(self.update_view)  # Connect the valueChanged signal to the update_view method

        self.opacity_slider = QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)  # Set the range of the slider to [-180, 180]
        self.opacity_slider.setValue(30)  # Set the initial value of the slider to 0

        # Add sliders to layout
        self.layout.addWidget(self.slider_elev)
        self.layout.addWidget(self.slider_elev)
        self.layout.addWidget(self.slider_azim)
        self.layout.addWidget(self.slider_roll)  # Add the slider to the grid layout
        self.layout.addWidget(self.opacity_slider)  # Add the slider to the grid layout

        # Add a button to hide the segmentation plot
        self.hide_segmentation_button = QPushButton('Hide/Show Segmentation')
        self.hide_segmentation_button.setStyleSheet("background-color: white; color: black;")
        self.hide_segmentation_button.clicked.connect(self.hide_segmentation)
        self.layout.addWidget(self.hide_segmentation_button)

        self.save_image_button = QPushButton('Save Image')
        self.save_image_button.setStyleSheet("background-color: white; color: black;")
        self.save_image_button.clicked.connect(self.save_image)
        self.layout.addWidget(self.save_image_button)

        # Set stretch factors
        self.layout.setStretch(0, 5)  # Make the 3D plot take up 3/4 of the space
        self.layout.setStretch(1, 1)  # Make the first slider take up 1/8 of the space
        self.layout.setStretch(2, 1)  # Make the second slider take up 1/8 of the space
        self.layout.setStretch(3, 1)  # Make the opacity slider take up 1/8 of the space
        self.layout.setStretch(4, 1)  # Make the button take up 1/8 of the space
 
 
    def innit_view(self):  
        # Calculate the changes in distance, elevation, azimuth, and roll
        delta_elev = self.init_camera - self.elev
        delta_azim = self.init_camera - self.azim
        delta_roll = self.init_camera - self.roll

        # Rotate the item in the 3D plot
        for item in self.plot_c_scan_3d.items:
            item.rotate(delta_elev, 1, 0, 0)  # Rotate around the x-axis
            item.rotate(delta_azim, 0, 1, 0)  # Rotate around the y-axis
            item.rotate(delta_roll, 0, 0, 1)  # Rotate around the z-axis

    def update_view(self):
        # Get the new distance, elevation, azimuth, and roll from the sliders
        new_elev = self.slider_elev.value()
        new_azim = self.slider_azim.value()
        new_roll = self.slider_roll.value()
    
        # Calculate the changes in distance, elevation, azimuth, and roll
        delta_elev = new_elev - self.elev
        delta_azim = new_azim - self.azim
        delta_roll = new_roll - self.roll
        print(new_elev, self.elev, delta_elev)

        self.pos = self.plot_c_scan_3d.opts["center"]
        self.distance = self.plot_c_scan_3d.opts["distance"]

        # Rotate the item in the 3D plot
        for item in self.plot_c_scan_3d.items:
            item.rotate(delta_elev, 1, 0, 0)  # Rotate around the x-axis
            item.rotate(delta_azim, 0, 1, 0)  # Rotate around the y-axis
            item.rotate(delta_roll, 0, 0, 1)  # Rotate around the z-axis
        
        print(f'Azimuth: {self.azim}, Elevation: {self.elev}, Roll: {self.roll}, Distance: {round(self.plot_c_scan_3d.opts["distance"])}, Position: {self.plot_c_scan_3d.opts["center"]}')

        # Update the distance, elevation, azimuth, and roll
        self.elev = new_elev
        self.azim = new_azim
        self.roll = new_roll

    def load_data(self, path, segmentation, segmentation_2):
        #[30:224, :80, :]
        #[0:480, :105, 12:]
        self.data = np.load(os.path.join(path, 'test_volume.npy'))#[:,0:1000:3,:]#[30:260, :, :]
        self.segmentation = np.load(os.path.join(path, segmentation))#[:,0:355:1,:]#[30:260, :, :]#[30:224, :80, :]
        self.data = np.swapaxes(self.data, 0, 2)
        self.segmentation = np.swapaxes(self.segmentation, 0, 2)
        # self.segmentation = np.swapaxes(self.segmentation, 0, 2)#[0:480, :105, 12:]
        if segmentation_2:
            self.segmentation_2 = np.load(os.path.join(path, segmentation_2))[30:, :80, :]
        # self.setup_new_plot()
        # Flip first 64 elements of the array along axis 1
        # self.data[:64] = np.flip(self.data[:64], axis=1)
        # self.segmentation[:64] = np.flip(self.segmentation[:64], axis=1)

    def setup_new_plot(self):
        self.data_4d = None
        self.segmentation_4d = None
        
        # self.data_4d = np.zeros(self.data.shape + (4,), dtype=np.ubyte)
        # self.data_4d[..., 3] = (self.data * 255).astype(np.ubyte)
        # color = (140, 0, 0)  
        # self.data_4d[..., :3] = color
        
        self.segmentation_4d = np.zeros(self.segmentation.shape + (4,), dtype=np.ubyte)
        rgba_value = (240, 228, 66, 255)  # (R, G, B, A)
        self.segmentation_4d[self.segmentation > 0] = rgba_value

        # Assuming self.data is already defined
        self.data_normalized = (self.data - self.data.min()) / (self.data.max() - self.data.min())  # Normalize data to [0, 1]
        colormap = plt.get_cmap('viridis')  # Choose a colormap
        data_rgba = colormap(self.data_normalized)  # Apply colormap to get RGBA

        self.data_4d = np.zeros(self.data.shape + (4,), dtype=np.ubyte)
        self.data_4d[..., :3] = (data_rgba[..., :3] * 255).astype(np.ubyte)  # Assign RGB values
        self.data_4d[..., 3] = (self.data * 255).astype(np.ubyte)  # Assign alpha values based on original data

        self.full_volume_item_1 = gl.GLVolumeItem(self.data_4d)
        self.plot_c_scan_3d.addItem(self.full_volume_item_1)

        # Add segmentation to the plot
        self.full_volume_item_2 = gl.GLVolumeItem(self.segmentation_4d, glOptions='additive')
        self.plot_c_scan_3d.addItem(self.full_volume_item_2)
        self.is_segmentation_visible = True

        # # Add a box around the volume
        box_item_1 = gl.GLBoxItem()
        box_item_1.setSize(*self.data_4d.shape[:-1])  
        box_item_1.setColor((255, 255, 255, 255)) 
        self.plot_c_scan_3d.addItem(box_item_1)
        self.opacity_slider.valueChanged.connect(self.update_opacity)
        self.plot_c_scan_3d.setCameraPosition(pos=self.pos, distance=self.distance)
        self.plot_c_scan_3d.update()  # Update the GLViewWidget
        self.innit_view()

    def update_opacity(self, value):
        opacity = value / 100.0
        a = np.copy(self.data_4d)
        a[...,3] = (self.data_4d[:,:,:,3] * opacity).astype(np.ubyte)
        self.full_volume_item_1.setData(a)

    def hide_segmentation(self):
        if self.is_segmentation_visible:
            self.full_volume_item_2.setVisible(False)
            self.is_segmentation_visible = False
        else:
            self.full_volume_item_2.setVisible(True)
            self.is_segmentation_visible = True

    def save_image(self):
        # Open a QFileDialog to select the location and filename to save the image
        filename, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Image Files (*.png *.jpg *.bmp)")

        # If a filename was selected, save the image
        if filename:
            self.plot_c_scan_3d.makeCurrent()
            self.plot_c_scan_3d.grabFrameBuffer().save(filename)

def plot_cscan(data, depth=0, segmentation=None, path=None, tof=False):    
    with plt.style.context(['science', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman",
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(10, 6))
        #plot thresholds against mea and std as error bars
        if tof:
            plt.imshow(np.argmax(data[:, depth:, :], axis = 1))#
        else:
            plt.imshow(np.amax(data[:, depth:, :], axis = 1))
            plt.clim(0, 1)

        if segmentation is not None:
            #custom colour map to set the 0 values to 0 alpha values
            num_colors = 256

            # Create a custom RGBA array from transparent to red
            custom_colors = np.zeros((num_colors, 4))

            # Set RGB values to (240, 228, 66)
            custom_colors[:, 0] = 240 / 255.0  # Red channel
            custom_colors[:, 1] = 228 / 255.0  # Green channel
            custom_colors[:, 2] = 66 / 255.0  # Blue channel

            # Set RGB values to (240, 228, 66)
            custom_colors[:, 0] = 212 / 255.0  # Red channel
            custom_colors[:, 1] = 21 / 255.0  # Green channel
            custom_colors[:, 2] = 21 / 255.0  # Blue channel

            custom_colors[:, 3] = np.linspace(0, 1, num_colors)  # Alpha channel ranges from 0 (transparent) to 1 (opaque)

            # Set alpha to 0 for values less than 1
            custom_colors[:1, 3] = 0
            # Create a custom colormap from the RGBA array
            custom_cmap = mcolors.ListedColormap(custom_colors)
            plt.imshow(np.amax(segmentation[:, :, :], axis = 1), cmap=custom_cmap)
        plt.ylabel('Array Axis (Pixels)')
        plt.xlabel('Scan Axis (Pixels)')
        if path:
            plt.savefig(path, dpi=400, bbox_inches='tight')
        plt.show()

def plot_bscan(data, idx=0, segmentation = None, path=None):
    with plt.style.context(['science', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.figure(figsize=(5, 3))
        plt.imshow(data[idx,:,:], aspect='auto')
        if segmentation is not None:
            #custom colour map to set the 0 values to 0 alpha values
            num_colors = 256

            # Create a custom RGBA array from transparent to red
            custom_colors = np.zeros((num_colors, 4))

            # Set RGB values to (240, 228, 66)
            custom_colors[:, 0] = 255 / 255.0  # Red channel
            custom_colors[:, 1] = 0 / 255.0  # Green channel
            custom_colors[:, 2] = 0 / 255.0  # Blue channel

            custom_colors[:, 3] = np.linspace(0, 1, num_colors)  # Alpha channel ranges from 0 (transparent) to 1 (opaque)

            # Set alpha to 0 for values less than 1
            custom_colors[:1, 3] = 0
            # Create a custom colormap from the RGBA array
            custom_cmap = mcolors.ListedColormap(custom_colors)

            plt.imshow(segmentation[idx, :, :], cmap=custom_cmap, alpha=0.5)

        #plot vertical line at given points
        start_1 = 13
        start_2 = 75
        start_3 = 138.2
        plt.axvline(x=start_1, color='green', linestyle='--', alpha=1)
        plt.axvline(x=start_1+(9/0.8), color='green', linestyle='--', alpha=1)
        plt.axvline(x=start_2, color='green', linestyle='--', alpha=1)
        plt.axvline(x=start_2+(6/0.8), color='green', linestyle='--', alpha=1)
        plt.axvline(x=start_3, color='green', linestyle='--', alpha=1)
        plt.axvline(x=start_3+(3/0.8), color='green', linestyle='--', alpha=1)
        #plot thresholds against mea and std as error bars
        plt.xlabel('Array Axis (Pitch 0.8 mm)')
        plt.ylabel('Depth (0.1 μs)')
        if path:
            plt.savefig(path, dpi=400, bbox_inches='tight')
        plt.show()

def get_FP(thresholded, segmentation):
    threshold_count = np.count_nonzero(thresholded)
    segmentation_count = np.count_nonzero(segmentation)
    differenece = segmentation_count - threshold_count
    fn = differenece/(np.size(segmentation)-threshold_count)
    return fn, differenece/np.size(segmentation)

def get_speed_of_sound(a_scan): #there is an error here
    offset = int(500/10)
    sample_rate = 100e6/10
    thickness = 8.6e-3
    front_wall_idx = np.argmax(a_scan)
    back_wall_idx = np.argmax(a_scan[offset:])
    peak_to_peak_time = ((back_wall_idx+offset)-front_wall_idx)/sample_rate
    return 2*thickness/peak_to_peak_time
    
def get_depth(a_scan_exp, a_scan_seg, speed_of_sound):
    sample_rate = 100e6/10
    split_seg = np.split(a_scan_seg, np.where(np.diff(a_scan_seg) != 0)[0]+1)
    segmentation_index = np.where(a_scan_seg[:(len(split_seg[0])+len(split_seg[1]))] == 2)[0]
    # check for continuous sequential values
    segmentation_index = (segmentation_index[0] + segmentation_index[-1]) // 2
    # segmentation_index = np.argmax(a_scan_seg) + int(segmentation_index)
    distance = segmentation_index-np.argmax(a_scan_exp[:20]) #finds first, change to find mean
    # plt.plot(a_scan_exp)
    # plt.plot(a_scan_seg)
    # plt.show()
    peak_to_peak_time = distance/sample_rate
    return peak_to_peak_time*speed_of_sound*1e3/2

def get_depths(volume, segmented):
    labeled_volume, num_labels = label(np.amax(segmented, axis=1), connectivity=2, return_num=True)
    properties = regionprops(labeled_volume)
    centers = [prop.centroid for prop in properties]
    speed_of_sound = get_speed_of_sound(volume[5, :, 5])
    print(speed_of_sound)
    # depths = [get_depth(volume[int(center[0]), :, int(center[1])], segmented[int(center[0]), :, int(center[1])], speed_of_sound) for center in centers]
    depths = []
    return depths , centers

def plot_cscan_depths(volume, depths, centers):
    plt.imshow(np.amax(volume, axis=1), cmap='gray')
    for center, depths in zip(centers, depths):
        plt.text(center[1], center[0], f'{depths:.2f}', color='red')
    plt.show()

def gen_6db_threshold(volume, idx=0):
    volume = volume[:32,idx:idx+5,:32]
    c_scan = np.amax(volume, axis=1)
    db_threshold = np.ones(c_scan.shape)
    db_threshold[c_scan<(np.max(c_scan))/2]=0
    return db_threshold

def compare_centers(exp_volume, seg_volume):
    exp_volume=exp_volume[:, :, :]
    seg_volume=seg_volume[:, :, :]
    print(exp_volume.shape)
    idx = 20
    step = 50
    count = 0
    positions = [
        [10, 10],
        [10, 45],
        [10, 80],

        [130, 26],
        [130, 56],
        [130, 90],

        [225, 10],
        [225, 45],
        [225, 80],

        [315, 10],
        [315, 45],
        [315, 80],

        [385, 10],
        [385, 45],
        [385, 80],
    ]

    idxs = [
        61,
        40,
        22,

        22,
        46,
        74,

        74,
        46,
        22,

        88,
        56,
        22,

        22,
        56,
        88

    ]



    for i, pos in enumerate(positions):
        print(pos)
        # plt.imshow(np.amax(exp_volume[pos[0]:pos[0]+34, 25:, pos[1]:pos[1]+34], axis=1))
        print(exp_volume[pos[0]:pos[0]+34, 25:, pos[1]:pos[1]+34].shape)
        test_db=gen_6db_threshold(exp_volume[pos[0]:pos[0]+34, :, pos[1]:pos[1]+34], idx=idxs[i])
        test_seg=np.amax(seg_volume[pos[0]:pos[0]+34, :, pos[1]:pos[1]+34], axis=1)
        dist = compare_centroids(test_db, test_seg)
        plt.imshow(test_db, cmap='gray', alpha=0.5)
        plt.imshow(test_seg, alpha=0.5)
        #display the image dist
        plt.text(0, 0, f'{dist:.2f}', color='red')
        plt.show()
        count+=1
        print(count, dist)

    print(count)

def compare_centroids(test_db, test_seg):
    # Calculate the centroids
    centroid_test_db = ndimage.measurements.center_of_mass(test_db)
    centroid_test_seg = ndimage.measurements.center_of_mass(test_seg)
    print(centroid_test_db, centroid_test_seg)
    # Calculate the distance between the centroids
    distance = np.sqrt(np.sum((np.array(centroid_test_db) - np.array(centroid_test_seg))**2))
    return distance

# if __name__ == "__main__":
app = QApplication(sys.argv)
window = VisualiseVolume()
path = r'C:\GIT\Self-supervised-volumetric-detection\V2\inference\ID018'
path = r'C:\GIT\Self-supervised-volumetric-detection\V2\inference\padding'
path = r'C:\GIT\Self-supervised-volumetric-detection\V2\inference\lear\hilbert'
path = r'C:\GIT\Self-supervised-volumetric-detection\V2\inference\lear\rectified'
path = r'C:\GIT\Self-supervised-volumetric-detection\V2\inference\lear\belfast\moving_average'
segmentation='0.9999999_forward.npy'
segmentation='0.9999999_backward.npy'
segmentation='0.9999999_combined.npy'
segmentation='0.999_thresholded.npy'
window.load_data(path, segmentation=segmentation, segmentation_2=None)
# window.setup_new_plot()
print(window.data.shape)
print(window.segmentation.shape)
# window.show()
save_path = r'C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\PhD\Publications\Journal\SSL\media'
save_path = r'C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\PhD\Thesis\media\chapter 6'
depth = 30
part = segmentation.split('_')[-1].split('.npy')[0]
# plot_cscan(np.swapaxes(window.data, 0,2), depth=depth)#, path=os.path.join(save_path, 'test_data.png'))
# plot_cscan(np.swapaxes(window.data, 0,2), depth=depth, segmentation=np.swapaxes(window.segmentation, 0,2), path=os.path.join(save_path, f'{part}.png'))
# plot_cscan(window.data[:,:,:], depth=depth, segmentation=window.segmentation[:,:,:])#, path=os.path.join(save_path, 'rectified.png'))
plot_cscan(np.flip(window.data[10:,:,25:155], axis=(0,2)), depth=depth, segmentation=None, tof=False, path=os.path.join(save_path, 'tecnatom_clear.png'))
# plot_cscan(np.flip(window.data[10:,:,25:155], axis=(0,2)), depth=depth, segmentation=np.flip(window.segmentation[10:,:350,25:155], axis=(0,2)), tof=True)#, path=os.path.join(save_path, 'tecnatom.png'))
# plt.imshow(window.data[:, :, 64], aspect='auto')
# plot_bscan(window.data, segmentation=None, idx=2, path=None)
# plot_bscan(window.data, segmentation=window.segmentation, idx=2, path=None)
# depths, centers = get_depths(window.data, window.segmentation)
# plot_cscan_depths(window.segmentation, depths=[], centers=[])
plot_bscan(window.data, segmentation=None, idx=167, path=None) # 335, 402 
plot_bscan(window.data, segmentation=window.segmentation, idx=167, path=None) # 335, 402 
# compare_centers(window.data, window.segmentation)
sys.exit(app.exec_())
