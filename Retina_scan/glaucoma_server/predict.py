# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 19:11:56 2021

@author: Valerya
"""

# Prediction

import matplotlib.pyplot as plt
import numpy as np
import h5py
import random
import tensorflow as tf
import os

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

from unet_segmentation import split_train_val

with h5py.File("./glaucoma.h5",'r') as f:
    train_id = split_train_val(0.8, 0, True)
    train_data = f["image"][train_id]
    train_mask = f["mask"][train_id]
    val_id = split_train_val(0.8, 0, False)
    val_data = f["image"][val_id]
    val_mask = f["mask"][val_id]


model = load_model('D:/Files/4/BIA4/ICA1/glaucoma_server/model-glaucoma-1.h5', custom_objects={'accuracy'})
predict = model.predict(val_data)
print(predict)