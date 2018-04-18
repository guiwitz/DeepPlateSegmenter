import numpy as np
import scipy.ndimage as nd
import scipy.signal
import re
from skimage.feature import match_template

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
smooth = 1.

def phase_corr_simple(stack, thickness, z_step):
    """Return phase correlation image based on a bright field stack
    
    Parameters
    ----------
    stack : 3D numpy array
        stack to be correlated
    thickness: float 
        Thickness of cell [nm] (typically 800)
    z_step: float
        Size of steps between planes [nm]
        
    Returns
    -------
    correlated_norm : 2D numpy array
        Correlation image
    """
    corr_stack = stack
    nbplanes = stack.shape[2]
    middle_plane = round(nbplanes/2)-1
    
    heightpos = z_step*(np.arange(stack.shape[2])-middle_plane)
    
    gaussderiv = -heightpos*np.exp(-heightpos**2/(2*thickness**2))
    gaussderiv = gaussderiv/np.linalg.norm(gaussderiv)
    #height_vals = height_vals[np.newaxis,np.newaxis,...]
    
    height_vals = np.empty(stack.shape)
    height_vals[:] = gaussderiv
    
    image_norm = corr_stack/np.resize(np.sqrt(np.sum(corr_stack*corr_stack,axis =2)),
                                      (nbplanes, corr_stack.shape[0],corr_stack.shape[1])).transpose(1,2,0)
    
    #correlated_norm = ndimage.convolve(image_norm, height_vals, mode='constant', cval=0.0)
    #correlated_norm = scipy.signal.convolve(image_norm, height_vals, mode='valid')
    correlated_norm = image_norm*height_vals
    correlated_norm=np.sum(correlated_norm,axis =2)
    
    return correlated_norm

def phase_corr_simple_midplane(stack, midplane, thickness, z_step):
    """Return phase correlation image based on a bright field stack
    
    Parameters
    ----------
    stack : 3D numpy array
        stack to be correlated
    thickness: float 
        Thickness of cell [nm] (typically 800)
    z_step: float
        Size of steps between planes [nm]
        
    Returns
    -------
    correlated_norm : 2D numpy array
        Correlation image
    """
    #corr_stack = stack
    num_max = np.max([midplane,stack.shape[2]-midplane])
    corr_stack = stack[:,:,midplane-num_max:midplane+num_max+1]
    nbplanes = corr_stack.shape[2]
    middle_plane = round((nbplanes-1)/2)
    
    heightpos = z_step*(np.arange(corr_stack.shape[2])-middle_plane)
    
    gaussderiv = -heightpos*np.exp(-heightpos**2/(2*thickness**2))
    gaussderiv = gaussderiv/np.linalg.norm(gaussderiv)
    #height_vals = height_vals[np.newaxis,np.newaxis,...]
    
    height_vals = np.empty(corr_stack.shape)
    height_vals[:] = gaussderiv
    
    image_norm = corr_stack/np.resize(np.sqrt(np.sum(corr_stack*corr_stack,axis =2)),
                                      (nbplanes, corr_stack.shape[0],corr_stack.shape[1])).transpose(1,2,0)
    
    #correlated_norm = ndimage.convolve(image_norm, height_vals, mode='constant', cval=0.0)
    #correlated_norm = scipy.signal.convolve(image_norm, height_vals, mode='valid')
    correlated_norm = image_norm*height_vals
    correlated_norm=np.sum(correlated_norm,axis =2)
    
    return correlated_norm


def phase_corr_varying(stack, midplane, thickness, z_step, numplanes):
   
    #corr_stack = stack
    num_max = np.max([midplane,stack.shape[2]-midplane])
    corr_stack = stack[:,:,midplane-numplanes:midplane+numplanes+1]
    nbplanes = corr_stack.shape[2]
    middle_plane = round((nbplanes-1)/2)
    
    heightpos = z_step*(np.arange(corr_stack.shape[2])-middle_plane)
    
    gaussderiv = -heightpos*np.exp(-heightpos**2/(2*thickness**2))
    gaussderiv = gaussderiv/np.linalg.norm(gaussderiv)
    #height_vals = height_vals[np.newaxis,np.newaxis,...]
    
    height_vals = np.empty(corr_stack.shape)
    height_vals[:] = gaussderiv
    
    image_norm = corr_stack/np.resize(np.sqrt(np.sum(corr_stack*corr_stack,axis =2)),
                                      (nbplanes, corr_stack.shape[0],corr_stack.shape[1])).transpose(1,2,0)
    
    #correlated_norm = ndimage.convolve(image_norm, height_vals, mode='constant', cval=0.0)
    #correlated_norm = scipy.signal.convolve(image_norm, height_vals, mode='valid')
    correlated_norm = image_norm*height_vals
    correlated_norm=np.sum(correlated_norm,axis =2)
    
    return correlated_norm

def phase_corr(stack, z0, thickness, z_step):
    """Return phase correlation image based on a bright field stack
    
    Parameters
    ----------
    stack : 3D numpy array
        stack to be correlated
    z0 : int
        index of "central" plane (cell signal extinction)
    thickness: float 
        Thickness of cell [nm] (typically 800)
    z_step: float
        Size of steps between planes [nm]
        
    Returns
    -------
    correlated_norm : 2D numpy array
        Correlation image
    """
    corr_stack = np.zeros(stack.shape)
    nbplanes = stack.shape[2]
    full_mask = np.zeros(stack.shape)
    middle_plane = round(nbplanes/2)
    #z0 = middle_plane*np.ones(stack.shape[0:2])
    #z0[0,2]=10
    delta_per_pix = np.min(np.stack((nbplanes-z0,z0-1),axis=2),axis = 2)

    for i in range(nbplanes):

        #masking of pixels outside allowed range
        mask = 1-(np.abs(i-z0)>delta_per_pix)
        full_mask[:,:,i] = mask

        #distance to center matrix
        height_mask=i-z0
        height_mask =z_step*height_mask

        #Calculate derivative of Gaussian for whole image
        gaussderiv = -height_mask*np.exp(-height_mask**2/(2*thickness**2))
        gaussderiv = gaussderiv*mask
        corr_stack[:,:,i]=gaussderiv
    
    image_masked = stack*full_mask
    image_norm = image_masked/np.resize(np.sqrt(np.sum(image_masked*image_masked,axis =2)),(nbplanes, image_masked.shape[0],image_masked.shape[1])).transpose(1,2,0)
    gauss_norm = corr_stack/np.resize(np.sqrt(np.sum(corr_stack*corr_stack,axis =2)),(nbplanes, corr_stack.shape[0],corr_stack.shape[1])).transpose(1,2,0)
    correlated_norm = gauss_norm* image_norm
    correlated_norm=np.sum(correlated_norm,axis =2)
    
    return correlated_norm

def steerable1(image, alpha=0, sigma=1):
    """Return firsr order gaussian steerable filtering of an image
    
    Parameters
    ----------
    image : numpy array
        Image to be filtered
    alpha : float
        Angle of steerable filter
    sigma : width of filter
    
    Returns
    -------
    filtered : 2D numpy array
        Filtered image
    """
    support = np.floor(4*sigma)
    if support<1:
        support = 1
    x = np.arange(-support, support+1)
    xx, yy = np.meshgrid(x,x)
    
    G1_0 = -(xx/sigma**2)*np.exp(-(xx**2+yy**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi));
    G1_90 = -(yy/sigma**2)*np.exp(-(xx**2+yy**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi));


    Ix = nd.convolve(image,G1_0)
    Iy = nd.convolve(image,G1_90)
    
    filtered = np.cos(theta)*Ix+np.sin(theta)*Iy
    return filtered


def steerable2(image, alpha=0, sigma=1):
    """Return second order gaussian steerable filtering of an image
    
    Parameters
    ----------
    image : numpy array
        Image to be filtered
    alpha : float
        Angle of steerable filter
    sigma : width of filter
    
    Returns
    -------
    filtered : 2D numpy array
        Filtered image
    """
    support = np.floor(4*sigma)
    if support<1:
        support = 1
    x = np.arange(-support, support+1)
    xx, yy = np.meshgrid(x,x)
    
    g0 = np.exp(-(xx**2+yy**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    G2a = -g0/sigma**2+g0*xx**2/sigma**4
    G2b = g0*xx*yy/sigma**4
    G2c = -g0/sigma**2+g0*yy**2/sigma**4
    
    I2a = nd.convolve(image,G2a)
    I2b = nd.convolve(image,G2b)
    I2c = nd.convolve(image,G2c)
    
    filtered = (np.cos(alpha))**2*I2a+np.sin(alpha)**2*I2c-2*np.cos(alpha)*np.sin(alpha)*I2b
    
    return filtered

def steerable2_crosscorr(image, alpha=0, sigma=1):
    """Return second order normalized cross-correletion gaussian steerable filtering of an image
    
    Parameters
    ----------
    image : numpy array
        Image to be filtered
    alpha : float
        Angle of steerable filter
    sigma : width of filter
    
    Returns
    -------
    filtered : 2D numpy array
        Filtered image
    """
    support = np.floor(4*sigma)
    if support<1:
        support = 1
    x = np.arange(-support, support+1)
    xx, yy = np.meshgrid(x,x)
    
    g0 = np.exp(-(xx**2+yy**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    G2a = -g0/sigma**2+g0*xx**2/sigma**4
    G2b = g0*xx*yy/sigma**4
    G2c = -g0/sigma**2+g0*yy**2/sigma**4
    
    filt = (np.cos(alpha))**2*G2a+np.sin(alpha)**2*G2c-2*np.cos(alpha)*np.sin(alpha)*G2b
    filtered = match_template(image,filt,pad_input=True)
    
    return filtered


output =True

def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def get_unet(dims,img_rows,img_cols):
    inputs = Input((img_rows, img_cols, dims))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    
    conv11 = Reshape((img_rows*img_cols,1),input_shape=(img_rows,img_cols,1))(conv10)

    model = Model(inputs=[inputs], outputs=[conv11])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef], sample_weight_mode='temporal')
        
    return model

def train_generator(folder):
    num = 0
    while num > -1:
        img_load = np.load(folder+'/imgs_train_'+str(num)+'.npy')
        mask_load = np.load(folder+'/imgs_mask_train_'+str(num)+'.npy')
        weight_load = np.load(folder+'/imgs_weight_train_'+str(num)+'.npy')
        
        img_load = img_load.astype('float32')
        mask_load = mask_load[..., np.newaxis]
        mask_load = mask_load.astype('float32')
        mask_load /= 255.  # scale masks to [0, 1]
    
        weight_load = weight_load.astype('float32')
    
        yield (img_load,mask_load,weight_load)
        num += 1
        if num==230:
            num=0

def valid_generator(folder):
    num = 0
    while num > -1:
        img_load = np.load(folder+'/imgs_valid_'+str(num)+'.npy')
        mask_load = np.load(folder+'/imgs_mask_valid_'+str(num)+'.npy')
        weight_load = np.load(folder+'/imgs_weight_valid_'+str(num)+'.npy')
        
        img_load = img_load.astype('float32')
        mask_load = mask_load[..., np.newaxis]
        mask_load = mask_load.astype('float32')
        mask_load /= 255.  # scale masks to [0, 1]
    
        weight_load = weight_load.astype('float32')
    
        yield (img_load,mask_load,weight_load)
        num += 1
        if num==58:
            num=0
        
        
def plate_deeptrain(folder, img_rows,img_cols, dims, batch_size = 32, epoch = 100, weights = None):

    imgs_train, imgs_mask_train, imgs_weight_train  = load_train_data(folder)
    imgs_train = imgs_train.astype('float32')

    imgs_mask_train = imgs_mask_train[..., np.newaxis]
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    
    imgs_weight_train = imgs_weight_train.astype('float32')

    plate_model = get_unet(dims,img_rows,img_cols)
    model_checkpoint = ModelCheckpoint(folder+'weights.h5', monitor='val_loss', save_best_only=True)
    
    if weights:
        plate_model.load_weights(weights)
        
    plate_model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epoch=epoch, verbose=1, shuffle=True,
              validation_split=0.2,sample_weight = imgs_weight_train,
              callbacks=[model_checkpoint])
    


    '''imgs_test, imgs_id_test = load_test_data(folder)

    imgs_test = imgs_test.astype('float32')
    
    plate_model.load_weights(folder+'weights.h5')

    imgs_mask_test = plate_model.predict(imgs_test, verbose=1)
    #np.save('imgs_mask_test.npy', imgs_mask_test)

    test_dir = folder+'test'
    if not os.path.exists(test_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = np.reshape(image,(img_rows,img_cols))
        image = (image * 255.).astype(np.uint8)
        imsave(os.path.join(test_dir, str(image_id) + '_pred.png'), image)'''
    
def plate_deeptrain_batches(folder, img_rows,img_cols, dims, train_batch_nb = 230, validation_batch_nb = 58, epochs = 100, weights = None):


    plate_model = get_unet(dims,img_rows,img_cols)
    model_checkpoint = ModelCheckpoint(folder+'weights.h5', monitor='val_loss', save_best_only=True)
    if weights:
        plate_model.load_weights(weights)
    
    plate_model.fit_generator(train_generator(folder), steps_per_epoch=train_batch_nb, epochs=epochs,
                              validation_data=valid_generator(folder),validation_steps = validation_batch_nb,verbose=1,
                              callbacks=[model_checkpoint])


        
def load_train_data(folder):
    imgs_train = np.load(folder+'imgs_train.npy')
    imgs_mask_train = np.load(folder+'imgs_mask_train.npy')
    imgs_weight_train = np.load(folder+'imgs_weight_train.npy')
    return imgs_train, imgs_mask_train, imgs_weight_train

def load_test_data(folder):
    imgs_test = np.load(folder+'imgs_test.npy')
    imgs_id = np.load(folder+'imgs_id_test.npy')
    return imgs_test, imgs_id

