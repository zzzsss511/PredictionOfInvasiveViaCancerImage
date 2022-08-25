# Libraries

## Basics
import imp
import numpy as np
import pandas as pd
import datetime

## Visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sb



## Statistics
from scipy import sparse

## Machine learning
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.tree
import sklearn.ensemble

## Deep learning
import tensorflow as tf
import keras.preprocessing.image

## Image&Video related
import cv2 # Reads videos

## Miscellaneous
import gc
import os # Provides function for interacting with the operating system
import time
import warnings

# Self-defined functions


# Load data files
## Load images of shape (5547, 50, 50, 3)
## import data

# load images of shape (5547, 50, 50, 3)
x_images = np.load('Data/X.npy')  

# load labels of shape (5547,1); (0 = no cancer, 1 = cancer)
y_images = np.load('Data/Y.npy')   

# shuffle data
perm_array = np.arange(len(x_images))
np.random.shuffle(perm_array)
x_images = x_images[perm_array]
y_images = y_images[perm_array]

print('x_images.shape =', x_images.shape)
print('x_images.min/mean/std/max = %.2f/%.2f/%.2f/%.2f'%(x_images.min(),
                        x_images.mean(), x_images.std(), x_images.max()))
print('')
print('y_images.shape =', y_images.shape)
print('y_images.min/mean/std/max = %.2f/%.2f/%.2f/%.2f'%(y_images.min(),
                        y_images.mean(), y_images.std(), y_images.max()))


## plot some images  

imgs_0 = x_images[y_images == 0] # 0 = no cancer
imgs_1 = x_images[y_images == 1] # 1 = cancer

plt.figure(figsize=(10,10))
for i in range(30):
    plt.subplot(5,6,i+1)
    plt.title('IDC = %d'%y_images[i])
    plt.imshow(x_images[i])

plt.show()