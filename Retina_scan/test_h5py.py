# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 20:51:26 2021

@author: Valerya
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("D:/Files/4/BIA4/ICA1/glaucoma/glaucoma.h5",'r') as f:
    print(f)
    imgs = f["image"]
    print(imgs.shape)
    plt.imshow(imgs[0])
    img_annotations = f["annotation"]
    plt.imshow(img_annotations[0], cmap = "gray")
    print(f["expCDR"][:6])
    print(f["eye"][:6])
    print(f["set"][:6])
    print(f["glaucoma"][:6])
    
    