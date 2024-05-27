#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import matplotlib.colors as mcolors
import matplotlib.patches as patches

import torch
import torch.distributions as dist

import scienceplots
from classifierv2 import CNN_shared

plt.style.use(['science','no-latex', 'bright'])

# load in numpy data function
def load_data(file_path):
    data = np.load(file_path)
    print(data.shape)
    series = data[10:74,100,5]
    gt = data[74,100,5]
    return series, gt

# load in pytorch model and get predictions
def load_model(model_path, data):
    # Load the model
    model = CNN_shared()
    model = model.double()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prepare the data
    data = torch.from_numpy(data).double()
    data = data.unsqueeze(0).unsqueeze(0)
    print(data.shape)

    # Get the predictions
    with torch.no_grad():
        concentration, scale = model(data)

    concentration = torch.abs(concentration)
    scale = torch.abs(scale)
    
    return concentration, scale

def plot_distribution(data):
    # Predict the mean and standard deviation of the next point
    mean = np.mean(data)
    std_dev = np.std(data)

    # Generate the y values for the distribution
    y = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)

    # Calculate the normal distribution
    pdf = stats.norm.pdf(y, mean, std_dev)

    # Plot the data
    plt.plot(data, label='Data')

    # Plot the distribution of the expected point as a 2D distribution
    plt.plot([len(data)]*len(y), y, color='red', alpha=0.3, linewidth=0)
    plt.fill_betweenx(y, len(data), len(data) + pdf*30, color='red', alpha=0.3)

    plt.legend()
    plt.show()

def plot_distribution_varying_opacity(data):
    # Predict the mean and standard deviation of the next point
    mean = np.mean(data)
    std_dev = np.std(data)

    # Generate the y values for the distribution
    y = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 10000)

    # Calculate the normal distribution
    pdf = stats.norm.pdf(y, mean, std_dev)

    # set figure size
    plt.figure(figsize=(10, 5))
    # Plot the data
    plt.plot(data, label='Data')

    # Create a colormap that varies with the distribution
    colors = plt.cm.Reds(pdf / np.max(pdf))
    rgba_colors = mcolors.to_rgba_array(colors)
    rgba_colors[:, 3] = pdf / np.max(pdf)  # Set the alpha values according to the distribution

    # Plot the distribution of the expected point as a 2D distribution
    for i in range(len(y)):
        plt.plot([len(data), len(data) + pdf[i]*0.5], [y[i], y[i]], color=rgba_colors[i])

    # Create a ScalarMappable object with the same colormap and normalization
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])

    # Create a colorbar with the ScalarMappable object
    plt.colorbar(sm, label='Probability density')

    plt.legend()
    plt.show()

def plot_prediction_varying_opacity(data, gt, distribution):
    print(distribution)
    # Define the Weibull distribution
    weibull_distribution = torch.distributions.Weibull(distribution[0], distribution[1])
    
    # Generate the y values for the distribution
    y = torch.linspace(weibull_distribution.icdf(torch.tensor(0.000001)), weibull_distribution.icdf(torch.tensor(0.999999)), 10000)
    
    # Calculate the Weibull distribution
    pdf = torch.exp(weibull_distribution.log_prob(y))

    # set figure size
    plt.figure(figsize=(10, 5))
    # Plot the data
    plt.plot(data, label='Input Series')#, color='black')

    # Create a colormap that varies with the distribution
    colors = plt.cm.Reds(pdf / torch.max(pdf))
    rgba_colors = mcolors.to_rgba_array(colors)
    rgba_colors[:, 3] = pdf / torch.max(pdf)  # Set the alpha values according to the distribution

    # Plot the distribution of the expected point as a 2D distribution
    for i in range(len(y)):
        plt.plot([len(data), len(data) + pdf[i]*0.5], [y[i], y[i]], color=rgba_colors[i])

    # Create a ScalarMappable object with the same colormap and normalization
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.scatter(len(data), gt, label='Ground Truth Output', marker='x', zorder=5)#color = 'black', 
    # Create a colorbar with the ScalarMappable object
    plt.colorbar(sm, label='Probability Density of Predicted Distribution')
    plt.xlabel('Series Index')    
    plt.ylabel('Amplitude')
    plt.legend()
    plt.ylim(0, 0.3)
    plt.xlim(0, 70)
    # plt.show()
    plt.savefig(r'C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\PhD\Publications\Journal\SSL\media\sequence prediction.png', dpi=400)

def plot_stride_demo(data):
    # set figure size
    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 14})
    # Plot the data
    plt.plot(data, label='Training Series')#, color='black')
    # set fig font size
    # Create a colorbar with the ScalarMappable object
    plt.xlabel('Series Index')    
    plt.ylabel('Amplitude')
    plt.ylim(0, 0.1)
    plt.xlim(0, len(data))

    # plt.gca().add_patch(patches.FancyBboxPatch((0, min(data)), 64, 0.01, boxstyle="Round, pad=0, rounding_size=0.005", fill=True, alpha = 0.3, color = 'green', label = 'Input Training Sample'))
    plt.gca().add_patch(patches.Rectangle((0, min(data)), 64, max(data)-min(data), fill=True, alpha = 0.3, color = 'green', label = 'Input Training Sample'))
    plt.gca().add_patch(patches.Rectangle((8, min(data)), 64, max(data)-min(data), fill=False, edgecolor = 'blue', label = 'Stride: 8', linestyle = '--', linewidth = 2))
    plt.gca().add_patch(patches.Rectangle((64, min(data)), 64, max(data)-min(data), fill=False, edgecolor = 'red', label = 'Stride: 64', linestyle = '-.', linewidth = 2))
    # plt.gca().add_patch(patches.Rectangle((0, min(data)), 64, max(data[:64])-min(data[:64]), fill=True, alpha = 0.3, color = 'green', label = 'Input Training Sample'))
    # plt.gca().add_patch(patches.Rectangle((8, min(data)), 64, max(data[8:64+8])-min(data[8:64+8]), fill=False, edgecolor = 'blue', label = 'Stride: 8', linestyle = '--', linewidth = 2))
    # plt.gca().add_patch(patches.Rectangle((64, min(data)), 64, max(data[64:64+64])-min(data[64:64+64]), fill=False, edgecolor = 'red', label = 'Stride: 64', linestyle = '-.', linewidth = 2))
    plt.arrow(0, 0.04, 4-0.1, 0, width = 0.0005, head_width=0.003, head_length=4, fc='blue', ec='blue', zorder=5)    
    plt.arrow(0, 0.04+0.005, 60-0.1, 0, width = 0.0005, head_width=0.003, head_length=4, fc='red', ec='red', zorder=5)    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=5)
    plt.savefig(r'C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\PhD\Publications\Journal\SSL\media\stride_demo.png', dpi=400)
    plt.show()

# Assuming 'data' is your time series data
data_path = r'C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\SEARCH NDE\Composites\Exp Data\Spirit Cell\Small CFRP Samples\ID010\ID010_hilbert.npy'
# test_series, gt = load_data(data_path)

# model_path = r'C:\GIT\Self-supervised-volumetric-detection\V2\stride_2_1_best_model.pth'
# scale, concentration = load_model(model_path, test_series)
# plot_prediction_varying_opacity(test_series, gt, [scale, concentration])

data = np.load(data_path)
print(data.shape)
plot_stride_demo(data[:260,200,10][:150])

exit()
# plot_distribution(test_series)
# plot_distribution_varying_opacity(test_series)

# %%
import matplotlib.pyplot as plt
import numpy as np

# Stride values
strides = [256, 128, 64, 32, 16, 8, 4, 2, 1]

# Mean values for each stride
means = [-2.7513651994037596, -2.7280875480070486, -2.923529449811516, -2.9133733147755265, -2.8955851961606336, -2.8959823017842914, -2.918114569183672, -2.9428965360081443, 2.9243448622970996]
#take absolute values of means
# means = [abs(i) for i in means]

# Standard deviation values for each stride
std_devs = [0.03538998615593571, 0.06516107752641004, 0.005809707570430849, 0.013651180636463499, 0.0400246864498967, 0.022942619389343423, 0.030555569049926734, 0.006543967797519257, 0.019878044566789082]

# Create a plot of the mean values with error bars for the standard deviations
plt.errorbar(range(len(strides)), means, yerr=std_devs, fmt='x', label='Mean with standard deviation')

# Set the labels for the x and y axes
plt.xlabel('Stride')
plt.ylabel('Mean')

# Set the x-ticks to be the stride values
plt.xticks(range(len(strides)), strides)

#add legend for mean and standard deviation
plt.legend()

# Show the plot
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Stride values
strides = [256, 128, 64, 32, 16, 8, 4, 2, 1]

# Values for each stride
values = [
    [-2.7017189810285345, -2.7816761525173206, -2.770700464665424],
    [-2.791834314388325, -2.638584212807473, -2.7538441168253485],
    [-2.9301339965313673, -2.9244596727658063, -2.9159946801373735],
    [-2.8941487583797425, -2.9245167437475175, -2.9214544421993196],
    [-2.9124258984811604, -2.840364670351846, -2.9339650196488947],
    [-2.8712473226478323, -2.890165147226071, -2.9265344354789704],
    [-2.944190783193335, -2.9349174993112683, -2.8752354250464123],
    [-2.937079980270937, -2.952038675546646, -2.93957095220685],
    [-2.89623451847001, -2.938647846924141, -2.9381522214971483]
]

# Convert values to absolute
# values = np.abs(values)

# Calculate the mean of each set of stride values
means = np.mean(values, axis=1)

# Create a plot for each set of values
for i, stride_values in enumerate(values):
    plt.plot([i]*len(stride_values), stride_values, 'x')

# Plot the mean values
plt.plot(range(len(means)), means, 'r-')

# Set the labels for the x and y axes
plt.xlabel('Stride')
plt.ylabel('Negative Logliklihood')

# Set the x-ticks to be the stride values
plt.xticks(range(len(strides)), strides)

# Show the plot
# plt.show()

#%%
import scienceplots
plt.style.use(['science','no-latex', 'ieee','bright'])

# Stride values
strides =[74127200, 37063600, 18531800, 9455000, 4916600, 2647400, 1512800, 756400, 378200 
        ]
#inverse strides order
strides = strides[::-1]
labels = ['256', '128', '64', '32', '16', '8', '4', '2', '1']

# Values for each stride
values = [
    [-2.7017189810285345, -2.7816761525173206, -2.770700464665424],
    [-2.791834314388325, -2.638584212807473, -2.7538441168253485],
    [-2.9301339965313673, -2.9244596727658063, -2.9159946801373735],
    [-2.8941487583797425, -2.9245167437475175, -2.9214544421993196],
    [-2.9124258984811604, -2.840364670351846, -2.9339650196488947],
    [-2.8712473226478323, -2.890165147226071, -2.9265344354789704],
    [-2.944190783193335, -2.9349174993112683, -2.8752354250464123],
    [-2.937079980270937, -2.952038675546646, -2.93957095220685],
    [-2.89623451847001, -2.938647846924141, -2.9381522214971483]
]

# # Convert values to absolute
# # values = np.abs(values)

# # Calculate the mean of each set of stride values
# means = np.mean(values, axis=1)

# # Create a plot for each set of values
# # for i, stride_values in enumerate(values):
# #     plt.plot([i]*len(stride_values), stride_values, 'x')

# plt.figure(figsize=(10, 5))
# plt.plot(strides, np.abs(values), 'x')
# plt.plot(strides, np.abs(means), 'r--')
# plt.xlabel('Training Set Size')
# plt.ylabel('Mean Test Log-Liklihood')
# # plt.xticks(strides, labels)
# plt.xscale('log')

# # Create a second set of axes
# ax2 = plt.twiny()

# # Set the x-ticks and x-labels for the second set of axes
# ax2.set_xticks(strides)
# ax2.set_xticklabels(labels)
# ax2.set_xlabel('Stride')

# # Show the plot
# # plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(['science', 'no-latex'])
with plt.style.context(['science', 'no-latex']):
    plt.rcParams["font.family"] = "Times New Roman"
    # Calculate means and standard deviations
    means = np.abs(np.mean(values, axis=1))
    std_devs = np.abs(np.std(values, axis=1))

    # Create a figure
    fig = plt.figure()

    # Add a subplot to the figure
    ax1 = fig.add_subplot(111)

    # Plot means with error bars for standard deviation on the first set of axes
    ax1.errorbar(strides, means, yerr=std_devs, fmt='x', capsize=3, color='black', label='Mean with \nStandard Deviation')
    ax1.set_xlabel('Training Set Size')
    ax1.set_ylabel('Mean Test Log-Liklihood')
    ax1.set_xscale('log')
    ax1.legend()
    # ax1.set_xlim(100000, 100000000)

    ax2 = ax1.twiny()  

    # Apply the logarithm to the strides and the x-limits of the first axes
    log_strides = np.log10(strides)
    log_xlim = np.log10(ax1.get_xlim())

    # Calculate the positions for the x-ticks on the second axis
    new_tick_locations = np.interp(log_strides, log_xlim, [0, 1])
    # Set the x-ticks and x-labels for the second set of axes
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(labels)
    #turn off minor ticks for top axis only
    ax2.xaxis.set_tick_params(which='minor', bottom=False, top=False)
    ax2.set_xlabel('Stride')
    # ax2.set_xscale('log')


    # Save the figure, including both sets of axes
    fig.savefig(r'C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\PhD\Publications\Journal\SSL\media\stride.png', dpi=400, bbox_inches='tight')

    # Show the plot
    plt.show()

 # %%
 #plot results for sizing
from matplotlib import pyplot as plt
import numpy as np 
import scienceplots

data = {
    "MAE": [3.29, 2.64, 2.15, 1.83, 1.55, 1.38],
    "STD": [0.66, 0.61, 0.69, 0.67, 0.66, 0.68],
    "Threshold": [0.01, 0.001, 1E-04, 1E-05, 1E-06, 1E-07]
} 
# Plotting

with plt.style.context(['science', 'no-latex']):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(5, 3))
    #plot thresholds against mea and std as error bars
    plt.errorbar(100*np.array(data["Threshold"]), data["MAE"], yerr=data["STD"], capsize=3, fmt='x', label='Mean with\nStandard Deviation', color='black')
    plt.xscale('log')
    plt.xlabel('Confidence Threshold (%)')
    plt.ylabel('Predicted Diameter Absolute Error (mm)')
    plt.ylim(0, 4)
    plt.legend(loc = 'lower right')
    plt.savefig(r'C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\PhD\Publications\Journal\SSL\media\sizing.png', dpi=400, bbox_inches='tight')
    plt.show()

# %%
