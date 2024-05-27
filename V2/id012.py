import numpy as np
import matplotlib.pyplot as plt

def visualise_data(data):
    #print c_scan
    plt.figure()
    plt.imshow(np.max(data[:,500:1200,:],1))
    plt.show()
    
def visualise_bscan(data):
    #print c_scan
    plt.figure()
    plt.imshow((data[:,:,32]), aspect='auto')
    plt.show()
    
data = np.load(r'c:\Users\Shaun McKnight\OneDrive - University of Strathclyde\SEARCH NDE\Composites\Exp Data\Spirit Cell\LBR\Hilbert_gated_shaun mulit pass thin.npy')
data=np.swapaxes(data, 0, 2)

num_passes=8
pass_length=135
for i in range(0,num_passes):
    if i==0:
        x=data[:,:,:pass_length]
        
    y=data[:,:,i*pass_length:i*pass_length+pass_length]
    if i%2==1 or i==2 or i==6:
        y=np.flip(y, axis=0)

    x=np.vstack((x,y))
x=x[64:,:,10:]
x = np.delete(x, np.s_[64:128], axis=0)
x = np.delete(x, np.s_[192:256], axis=0)
x = np.delete(x, np.s_[256:320], axis=0)
# flip = np.flip(x[122:122+64,:,:], axis =0)
# x[122:122+64,:,:] = flip
print(x.shape)
# x=x[0:310,:,10:135]
# drop values 64:128 on axis 1
# visualise_data(x)
# visualise_bscan(x)
(320, 2000, 125)
thin = x

# data = np.load(r'c:\Users\Shaun McKnight\OneDrive - University of Strathclyde\SEARCH NDE\Composites\Exp Data\Spirit Cell\LBR\Hilbert_gated_shaun_thick side.npy')
# # visualise_bscan(data)
# data=np.swapaxes(data, 0, 2)
# print(data.shape)
# # visualise_data(data)
# # visualise_bscan(data)
# num_passes=5
# pass_length=136
# for i in range(0,num_passes):
#     # print(i)
#     if i==0:
#         x=data[:,:,:pass_length]
#     y=data[:,:,i*pass_length:i*pass_length+pass_length]
#     if i%2==1 or i==2 or i==6:
#         # print('flip')
#         y=np.flip(y, axis=0)
#     # print(x.shape, y.shape)
#     # if i==0 or i==5:
#     #     print('pass')
#     # else:
#     x=np.vstack((x,y))
# # x=x[64:,:,10:]
# # x = np.delete(x, np.s_[64:128], axis=0)
# # x = np.delete(x, np.s_[192:256], axis=0)
# # x = np.delete(x, np.s_[256:320], axis=0)
# # flip = np.flip(x[122:122+64,:,:], axis =0)
# # x[122:122+64,:,:] = flip
# x=x[::-1,:,:]
# x=x[:-64,:,:125]
# print(x.shape)
# # x=x[0:310,:,10:135]
# # drop values 64:128 on axis 1
# # visualise_data(x)
# # visualise_bscan(x)
# # (320, 2000, 136)

# full=np.concatenate((thin,x), axis=0)
# print(full.shape)
full =x
# visualise_data(full)
# visualise_bscan(full)

# plt.figure()
# plt.plot(full[200,:,100])
# plt.plot(full[600,:,100])
# plt.show()


from scipy.signal import hilbert
data = np.load(r'c:\Users\Shaun McKnight\OneDrive - University of Strathclyde\SEARCH NDE\Composites\Exp Data\Spirit Cell\LBR\thick/Hilbert_gated_shaun_1.npy')
#apply hilbert transform along second axis
hilb = np.abs(np.apply_along_axis(hilbert, axis=1, arr=data[:,:2000,:]))
print(hilb.shape)
# print(data.shape)
# visualise_data(hilb)
# visualise_bscan(hilb)

# plt.figure()
# plt.plot(data[400,:,32])
# plt.plot(hilb[400,:,32])
# plt.show()
data=np.swapaxes(hilb, 0,2)
num_passes=6
pass_length=136
for i in range(0,num_passes):
    # print(i)
    if i==0:
        x=data[:,:,:pass_length]
    else:
        y=data[:,:,i*pass_length:i*pass_length+pass_length]
        if i%2==1 or i==4:
        #     # print('flip')
            y=np.flip(y, axis=0)
        print(x.shape, y.shape)
        # if i==0 or i==5:
        #     print('pass')
        # else:
        x=np.vstack((x,y))
# x=x[64:,:,10:]
# x = np.delete(x, np.s_[64:128], axis=0)
# x = np.delete(x, np.s_[192:256], axis=0)
# x = np.delete(x, np.s_[256:320], axis=0)
# flip = np.flip(x[122:122+64,:,:], axis =0)
# x[122:122+64,:,:] = flip
# x=x[::-1,:,:]
# x=x[:-64,:,:125]
print(x.shape)
x=x[:,:,10:135]
# visualise_data(x)
# visualise_bscan(x)

full=np.concatenate((full,x), axis=0)
full = np.delete(full, np.s_[320-64:320], axis=0)
full = np.delete(full, np.s_[128:128+64], axis=0)
visualise_data(full)
visualise_bscan(full)
np.save(r'c:\Users\Shaun McKnight\OneDrive - University of Strathclyde\SEARCH NDE\Composites\Exp Data\Spirit Cell\LBR\stepped_combined.npy', full)
# x=x[0:310,:,10:135]
# drop values 64:128 on axis 1
# visualise_data(x)
# visualise_bscan(x)
(320, 2000, 136)