# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 17:23:12 2021

@author: Valerya
"""

import scipy.io as scio
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import numpy as np
from skimage.transform import resize
from skimage.exposure import rescale_intensity

label_path = "D:/Files/4/BIA4/ICA1/glaucoma/archive/glaucoma.csv"
img_length = 256
glaucoma_label = pd.read_csv(label_path)
print(glaucoma_label.head())
dataset_annotation = np.zeros((650,img_length,img_length))
dataset_image = np.zeros((650,img_length,img_length,3))
dataset_expCDR = np.zeros(650)
dataset_eye = np.zeros(650, dtype='int8')
dataset_set = np.zeros(650, dtype='int8')
dataset_glaucoma = np.zeros(650, dtype='int8')

directory_annotation = "D:/Files/4/BIA4/ICA1/glaucoma/archive/ORIGA/ORIGA/Semi-automatic-annotations/"
directory_image = "D:/Files/4/BIA4/ICA1/glaucoma/archive/ORIGA/ORIGA/Images/"
for i in range(glaucoma_label.shape[0]):
    n = str(i+1).rjust(3,'0')
    #print(i)
    # annotation
    path=directory_annotation+n+".mat"
    l = scio.loadmat(path)
    img_annotation = l['mask']
    mid = int(img_annotation.shape[1]/2)
    img_annotation = rescale_intensity(img_annotation, in_range=(0, 2), out_range=(0, 255))
    dataset_annotation[i] = resize(img_annotation[:,mid-1024:mid+1024], (img_length,img_length))
    # image
    path=directory_image+n+".jpg"
    img = plt.imread(path)
    dataset_image[i] = resize(img[:,mid-1024:mid+1024,:], (img_length,img_length,3))
    # CDR
    dataset_expCDR[i] = glaucoma_label.loc[i,"ExpCDR"]
    # Eye
    dataset_eye[i] = 0 if glaucoma_label.loc[i,"Eye"] == "OD" else 1
    # Set
    dataset_set[i] = 0 if glaucoma_label.loc[i,"Set"] == "A" else 1
    # Glaucoma
    dataset_glaucoma[i] = glaucoma_label.loc[i,"Glaucoma"]

### augmentation/processing to images here

f = h5py.File("D:/Files/4/BIA4/ICA1/glaucoma/glaucoma.h5","w")
f.create_dataset("image",data = dataset_image)
f.create_dataset("annotation",data = dataset_annotation)
f.create_dataset("expCDR",data = dataset_expCDR)
f.create_dataset("eye",data = dataset_eye)
f.create_dataset("set",data = dataset_set)
f.create_dataset("glaucoma",data = dataset_glaucoma)

f.close()
