# DeepPlateSegmenter
DeepPlateSegmenter is a Python package designed to segment images acquired using Micro-Manager using convolutional neural networks.
Certain functions and the Jupyter notebooks are specifically designed to analyze data acquired using the plate acquisition
module of Micro-Manager, and to segment cells of phase correlation images. However the code can easily be recycled to accept other inputs.

##Installaton
Download or clone this repository. To use it, add it to you python path or alternatively move to the folder DeepPlateSegmenter/deeplate and install using:  

pip3 install .

This installs several large packages as well as the Jupyter notebook environment. If you want to avoid polluting your environment with those installs, I recommend installing this package in a virtual environment.

![](img/segmentation.png)

## Usage
The training set is created using ***generate\_plate\_training\_set.ipynb***. Segmentation is done on fluorescence and the used on the phase correlation images to create the trainin set. A weight-map is also created, allowing to enhance the importance of cell borders.  

The u-net convnet is trained in ***train\_plate.ipynb*** (should be run on a GPU e.g. on AWS).  

The weights learned in the training can then be used to segment any image as shown in ***deep\_plate\_analysis.ipynb***  

***MMData.py*** is a class creating a "Micro-Manager object". Metadata of the dataset can be accessed easily through the object.  

***platesegementer.py*** contains all functions related to segmentation.

## Required packages
skimage  
scipy  
numpy 
pandas  
PIL  
matplotlib  
Keras  
Tensorflow
