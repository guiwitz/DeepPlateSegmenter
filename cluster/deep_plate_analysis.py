import sys
import os

from deeplate.MMdata import MMData 
import numpy as np
import matplotlib.pyplot as plt
import re
import scipy.ndimage as nd

from skimage.measure import label, regionprops
from skimage import morphology
from skimage.color import label2rgb
import pandas as pd

import deeplate.platesegmenter as ps

folder = sys.argv[1]
weights_folder = sys.argv[2]
folder_to_save = sys.argv[3]
position = int(sys.argv[4])
bf_ch = int(sys.argv[5])
fluo_ch = int(sys.argv[6])

#create MM object
MMobj = MMData(folder = folder)

#load deep learning model and weights
plate_model = ps.get_unet(1, MMobj.height,MMobj.width)
plate_model.load_weights(weights_folder+'weights.h5')


#get metadata
z_step = MMobj.get_zstep()
positions, well = MMobj.get_position_names()

#do the segmentation
for i in range(position,position+1):#range(len(positions)):
    
    #load bf stack
    stack = MMobj.get_stack(frame=0,channel=bf_ch,position=i, compress = 1)
    
    #calculate correlation image
    correlated_norm = ps.phase_corr_simple(stack,thickness=800,z_step=z_step)
    correlated_norm_gauss = nd.gaussian_filter(correlated_norm,3)
    
    #deep-segment image
    plate_im = correlated_norm_gauss.astype('float32')#-np.mean(imgs_train)
    plate_im = plate_im#/np.std(imgs_train)
    plate_im = plate_im[np.newaxis,...,np.newaxis]
    plate_im_mask = plate_model.predict(plate_im, verbose=1)
    plate_im_mask = np.reshape(plate_im_mask,correlated_norm.shape)
    
    #threshold deepe learning segmentation and clean resulting mask
    plate_im_mask2 = plate_im_mask.copy()
    plate_im_mask2[plate_im_mask2<0.5]=0
    plate_im_mask2[plate_im_mask2>0.5]=1

    masklab = morphology.label(plate_im_mask2)
    cellinfo = regionprops(masklab)
    newMask = np.zeros(masklab.shape)
    for c in cellinfo:
        if (c.label>0)&(c.area>10)&(c.area<10000):
            newMask[masklab==c.label]=1
    
    #load fluo image
    im_fluo = MMobj.get_stack(frame=0,channel=fluo_ch,position=i, compress = 1)[:,:,0]
    #calculate local properties
    cell_info = regionprops(label(newMask),im_fluo)
    
    mean_int = [x.mean_intensity for x in cell_info]
    posx = [x.centroid[0] for x in cell_info]
    posy = [x.centroid[1] for x in cell_info]
    #create a dataframe
    cell_struct = {'mean_fluo':mean_int,'posx': posx, 'posy': posy}
    cell_frame = pd.DataFrame(cell_struct)
    
    cell_frame.to_csv(folder_to_save+positions[i]+'.csv')
    
    #save image
    fig, ax = plt.subplots(figsize=(20,20))
    plt.imshow(correlated_norm_gauss,cmap='gray')
    plt.imshow(label2rgb(label(plate_im_mask2),bg_label=0),alpha = 0.4)
    plt.show()
    fig.savefig(folder_to_save+positions[i]+'seg.png')
