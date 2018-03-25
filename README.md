# DeepPlateSegmenter
DeepPlateSegmenter is a Python module designed to segment images acquired using Micro-Manager using convolutional neural networks.
Certain functions and the Jupyter notebooks are specifically designed to analyze data acquired using the plate acquisition
module of Micro-Manager, and to segment cells of phase correlation images. However the code can easily be adapted to any input.

![](img/segmentation.png)

## Required packages
skimage  
scipy  
numpy 
pandas  
PIL  
matplotlib  
Keras  
Tensorflow

## Usage
The training set is created using generate_plate_training_set.ipynb. Segmentation is done on fluorescence and the used on the phase correlation images to create the trainin set. A weight-map is also created, allowing to enhance the importance of cell borders.  
The u-net convnet is trained  train_plate.ipynb (should be run on a GPU e.g. on AWS).  
The weights learned in the training can then be used to segement any image as shown in deep_plate_analysis.ipynb  
MMData.py is a class creating a "Micro-Manager object". Metadata of the dataset can be accessed easily through the object.  
platesegementer.py contains all functions related to segmentation.
