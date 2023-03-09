# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1limL-EjpEQBvVQl4rw85LIc3inufJSDg
"""

#! /opt/bin/nvidia-smi 
from google.colab import drive 
drive.mount('/content/drive/')

import keras.applications.xception
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Sequential, Model 
#from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import asarray
from PIL import Image
import pandas as pd
import imageio
import numpy as np
from skimage.transform import resize

TRAIN_DIR = '/content/drive/MyDrive/ORIGA/ORIGA/cropped/Train'
VAL_DIR = '/content/drive/MyDrive/ORIGA/ORIGA/cropped/Validation'

#TRAIN_DIR = '/content/drive/MyDrive/Fundus_Train_Val_Data/Fundus_Scanes_Sorted/Train'
#VAL_DIR = '/content/drive/MyDrive/Fundus_Train_Val_Data/Fundus_Scanes_Sorted/Validation'

HEIGHT = 300
WIDTH = 300
class_list = ["class_1", "class_2"]
FC_LAYERS = [1024, 512, 256]
dropout = 0.4
NUM_EPOCHS = 50
BATCH_SIZE = 8

def build_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    preditions = Dense(num_classes, activation='softmax')(x)
    x = Dropout(dropout)(x)
    finetune_model = Model(inputs = base_model.input, outputs = preditions)
    return finetune_model
base_model_1 = tf.keras.applications.Xception(weights = 'imagenet',pooling="max",
                       include_top = False,
                       input_shape = (HEIGHT, WIDTH, 3))

train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                   rotation_range = 90,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.1,)

val_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                  rotation_range = 90,
                                  horizontal_flip = True,
                                  vertical_flip = True)

train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    class_mode='categorical',
                                                    target_size = (HEIGHT, WIDTH),
                                                    batch_size = BATCH_SIZE)

val_generator = val_datagen.flow_from_directory(VAL_DIR,
                                                  class_mode='categorical',
                                                  target_size = (HEIGHT, WIDTH),
                                                  batch_size = BATCH_SIZE)




model = build_model(base_model_1,
                                      dropout = dropout,
                                      fc_layers = FC_LAYERS,
                                      num_classes = len(class_list))
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])

filepath = "./model_weights.h5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor = ["acc"], verbose= 1, mode = "max")
cb=TensorBoard(log_dir=("/content/drive/MyDrive/ORIGA"))
callbacks_list = [checkpoint, cb]
print(train_generator.class_indices)
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)

history = model.fit(train_generator, epochs = NUM_EPOCHS, steps_per_epoch = 50, 
                                       shuffle = True,validation_data=val_generator,
                    callbacks=[early_stop])




###plot loss and accuracy###
_, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 3))
history=history.history
ax[0].plot(history['loss'], c='gray', label = "Training loss")
ax[0].plot(history['val_loss'], c='red', label = "Validation loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].set_ylim(0, 5)
ax[0].legend()

ax[1].plot(history['accuracy'], c='gray', label="Training accuracy")
ax[1].plot(history['val_accuracy'], c='red', label = "Validation accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend()

model.save("/content/drive/MyDrive/model6")
np.save("/content/drive/MyDrive/model6_history.npy",history)
model.save("/content/drive/MyDrive/model6.h5")



###test###
from os import listdir
dir1='/content/drive/MyDrive/ORIGA/ORIGA/cropped/test/Glaucoma_Negative/'
list1=listdir(dir1)
print(list1)
dir2='/content/drive/MyDrive/ORIGA/ORIGA/cropped/test/Glaucoma_Positive/'
list2=listdir(dir2)
print(list2)
predict1=np.ndarray((0,0))
for i in list1:
  tmp_img=imageio.imread(dir1+i)
  temp_img = resize(tmp_img, (300,300, 3))
  temp_img = temp_img.reshape(1,300,300,3)
  predict=model.predict(temp_img)
  predict1=np.append(predict1,predict)
print(predict1)
predict2=np.ndarray((0,0))
for i in list2:
  tmp_img=imageio.imread(dir2+i)
  temp_img = resize(tmp_img, (300,300, 3))
  temp_img = temp_img.reshape(1,300,300,3)
  predict=model.predict(temp_img)
  predict2=np.append(predict2,predict)


###plot ROC###
r1=predict1[range(1,97,2)]
print(r1)
r2=predict2[range(1,35,2)]
print(r2)
r=np.append(r1,r2)
y_true=np.zeros((1,65))
y_true[0,48:65]=1
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = metrics.roc_curve(y_true[0], r)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
roc_auc_score(y_true[0],r)

