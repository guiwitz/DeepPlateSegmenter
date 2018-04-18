import sys
import os

from deeplate.MMdata import MMData 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import scipy.ndimage as nd

from skimage.measure import label, regionprops
from skimage import morphology
from skimage.color import label2rgb
import pandas as pd

import deeplate.platesegmenter as ps
import time
#print(sys.executable)

folder = sys.argv[1]
weights_folder = sys.argv[2]
folder_to_save = sys.argv[3]
position = int(sys.argv[4])
num_positions = int(sys.argv[5])
bf_ch = int(sys.argv[6])
fluo_ch = int(sys.argv[7])

#create MM object
MMobj = MMData(folder = folder)

#load deep learning model and weights
plate_model = ps.get_unet(10, 304,304)
plate_model.load_weights(weights_folder+'/weights.h5')


#get metadata
z_step = MMobj.get_zstep()
positions, well = MMobj.get_position_names()
well_str = [re.findall('(.*?)-.*',x)[0] for x in positions]


mid = int((MMobj.num_planes[bf_ch]-1)/2)

#setup folders
if not os.path.exists(folder_to_save+'/images'):
    os.makedirs(folder_to_save+'/images')
if not os.path.exists(folder_to_save+'/mask_prob'):
    os.makedirs(folder_to_save+'/mask_prob')
if not os.path.exists(folder_to_save+'/dataframes'):
    os.makedirs(folder_to_save+'/dataframes')

#do the segmentation
for i in range(position,position+num_positions):#range(len(positions)):
    
    start = time.time()
    #load bf stack
    #stack = MMobj.get_stack(frame=0,channel=bf_ch,position=i, compress = 1)
    stack = np.empty((MMobj.height,MMobj.width, 10))
    for j in range(10):
        plane = mid-5+j
        stack[:,:,j] = MMobj.get_image_fast(frame=0,channel=bf_ch,plane = plane,position=i, compress = 1)
        stack[:,:,j] = nd.gaussian_filter(stack[:,:,j],3)
    
    meanstack = np.mean(stack,axis =2)
    for m in range(stack.shape[2]):
        stack[:,:,m] = stack[:,:,m]-meanstack
    
    stack = stack/np.std(stack)
 
    #load fluo image
    im_fluo = MMobj.get_stack_fast(frame=0,channel=fluo_ch,position=i, compress = 1)[:,:,0]
 
    complete = np.empty((MMobj.height,MMobj.width))
    complete_proba = np.empty((MMobj.height,MMobj.width))
    
    topad = 24
    stack = np.pad(stack,((topad,topad),(topad,topad),(0,0)),mode = 'constant')
    for k in range(8):
        for m in range(8):
            
            stack_cur = stack[k*256:(k+1)*256+2*topad,m*256:(m+1)*256+2*topad,:]

            plate_im = stack_cur[np.newaxis,...]
            plate_im_mask = plate_model.predict(plate_im, verbose=1)
            plate_im_mask = np.reshape(plate_im_mask,[256+2*topad,256+2*topad])

            complete_proba[k*256:(k+1)*256,m*256:(m+1)*256] = plate_im_mask[topad:-topad,topad:-topad]
    
    stack = stack[topad:-topad,topad:-topad,:]
    complete = complete_proba.copy()

    complete[complete<0.3]=0
    complete[complete>0.3]=1
    masklab = morphology.label(complete)
    cellinfo = regionprops(masklab, im_fluo)  
    cellinfo2 = regionprops(masklab, complete_proba)  
    newMask = np.zeros(masklab.shape)
    
    for x in range(len(cellinfo)):
        c = cellinfo[x]
        c2 = cellinfo2[x]
        if (c.label>0)&(c.area>100)&(c.area<10000):
            if (c2.mean_intensity>0.5):
                if (c.solidity>0.8):#&(c.eccentricity>0.8):
                    newMask[c.coords[:,0],c.coords[:,1]]=1
    
    #calculate local properties
    complete_lab = label(newMask)
    cell_info = regionprops(complete_lab,im_fluo)
    
    mean_int = [x.mean_intensity for x in cell_info]
    posx = [x.centroid[0] for x in cell_info]
    posy = [x.centroid[1] for x in cell_info]
    sum_int = [np.sum(im_fluo[x.coords[:,0],x.coords[:,1]]) for x in cell_info]
    all_pix = [im_fluo[x.coords[:,0],x.coords[:,1]] for x in cell_info]
    area = [x.area for x in cell_info]
    eccentricity = [x.eccentricity for x in cell_info]
    box3_fluo = [np.mean(im_fluo[int(x.centroid[0])-1:int(x.centroid[0])+2,int(x.centroid[1])-1:int(x.centroid[1])+2]) for x in cell_info]
    #create a dataframe
    cell_struct = {'sum_fluo': sum_int,'mean_fluo':mean_int,'box3_fluo': box3_fluo, 'area': area, 'posx': posx, 'posy': 
                   posy,'all_pix':all_pix, 'eccentricity': eccentricity}
    cell_frame = pd.DataFrame(cell_struct)
    cell_frame['pos_name'] = positions[i]
    cell_frame['well_name'] = well_str[i]
    
    cell_frame.to_csv(folder_to_save+'/dataframes/'+positions[i]+'.csv')
    
    #save image
    fig, ax = plt.subplots(figsize=(20,20))
    plt.imshow(stack[:,:,0],cmap='gray')
    plt.imshow(label2rgb(label(newMask),bg_label=0),alpha = 0.3)
    for x in cell_info:
        plt.text(x=x.centroid[1],y=x.centroid[0],s = str(x.label))
    plt.show()
    fig.savefig(folder_to_save+'/images/'+positions[i]+'seg.png')
    
    np.save(folder_to_save+'/mask_prob/mask_'+str(i)+'.npy',newMask)
    #np.save(folder_to_save+'corr_'+str(i)+'.npy',correlated_norm_gauss)
    np.save(folder_to_save+'/mask_prob/prob_'+str(i)+'.npy',complete_proba)
    
    
    
