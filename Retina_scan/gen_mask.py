# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 05:07:05 2021

@author: Valerya
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("D:/Files/4/BIA4/ICA1/glaucoma/glaucoma.h5",'r+') as f:
    
    img_annotations = f["annotation"]
    plt.hist(img_annotations[0].ravel(), bins=255, color="black")
    #plt.imshow(img_annotations[0], cmap = "gray")
    l = img_annotations.shape[0]
    img_length = 256
    mask = np.zeros((l,img_length,img_length,2), dtype=np.bool)
    for i in range(l):
        mask[i,:,:,0] = img_annotations[i] > 100
        mask[i,:,:,1] = img_annotations[i] > 200
        
    plt.imshow(mask[0,:,:,0], cmap = "gray")
    #plt.imshow(mask[0,:,:,1], cmap = "gray")
    
    f.create_dataset("mask",data = mask)