# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 02:37:59 2021

@author: Valerya

adapted from https://stackoverflow.com/questions/48843599/u-net-image-segmentation-with-multiple-masks
"""

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

f = h5py.File("D:/Files/4/BIA4/ICA1/glaucoma/glaucoma.h5",'r')
img_num = 650

def split_train_val(percent_train,seed,train):
    l = np.array(range(img_num))
    np.random.seed(seed)
    np.random.shuffle(l)
    train_num = int(img_num * percent_train)
    if train:
        data_id = np.sort(l[:train_num])
    else:
        data_id = np.sort(l[train_num:])
    return data_id

train_id = split_train_val(0.8, 0, True)
train_data = f["image"][train_id]
train_mask = f["mask"][train_id]

'''
# check figure
ix = random.randint(0, len(train_id))
plt.imshow(train_data[ix])
plt.show()
plt.imshow(np.squeeze(train_mask[ix,:,:,0]), cmap="gray")
plt.show()
plt.imshow(np.squeeze(train_mask[ix,:,:,1]), cmap="gray")
plt.show()
'''

# Build a U-net model

img_length = 800
img_channel = 3

inputs = tf.keras.layers.Input((img_length, img_length, img_channel))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs) #  normalization

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(2, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# training
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-glaucoma-1.h5', verbose=1, save_best_only=True)
results = model.fit(train_data, train_mask, validation_split=0.1, batch_size=16, epochs=50, 
                    callbacks=[earlystopper, checkpointer])

'''
# prediction
idx = random.randint(0, len(train_data))
x=np.array(train_data[idx])
x=np.expand_dims(x, axis=0)
predict = model.predict(x, verbose=1)

predict = (predict > 0.5).astype(np.uint8)

plt.imshow(np.squeeze(predict[0]))
plt.show()

plt.imshow(train_data[idx])

plt.show()
'''
f.close()