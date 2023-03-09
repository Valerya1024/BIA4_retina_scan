# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 19:44:25 2021

@author: Valerya
"""

#import numpy as np
#import skimage.io as io
import matplotlib.pyplot as plt
from skimage.exposure import  equalize_adapthist
from skimage.filters import threshold_otsu
import cv2 
from skimage.exposure import rescale_intensity

directory = "D:/Files/4/BIA4/ICA1/glaucoma/archive/ORIGA/ORIGA/Images/"
img_rgb = plt.imread(directory+"002.jpg")
img_r = img_rgb[:,:,0]
img_g = img_rgb[:,:,1]
img_b = img_rgb[:,:,2]
'''
img_GaussianBlur=cv2.GaussianBlur(img_g,(9,9),1.5)
plt.imshow(img_GaussianBlur, cmap = "gray")
'''
d = 20
sigmaColor = 10
sigmaSpace = 10
img_bilateralFilter = cv2.bilateralFilter(img_g, d, sigmaColor, sigmaSpace)
plt.imshow(img_bilateralFilter, cmap = "gray")

r = 3
i = 5
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
img_blackhat = cv2.morphologyEx(img_bilateralFilter, cv2.MORPH_BLACKHAT, kernel, iterations=i)
plt.imshow(img_blackhat, cmap = "gray")

img_equalized_CLAHE = equalize_adapthist(img_blackhat, clip_limit=0.05)
plt.imshow(img_equalized_CLAHE, cmap = "gray")

img_rescale_intensity = rescale_intensity(img_equalized_CLAHE, in_range=(0.2, 0.6), out_range=(0, 255))
plt.hist(img_rescale_intensity.ravel(), bins=100, color="black")
plt.imshow(img_rescale_intensity, cmap = "gray")


'''
t = threshold_otsu(img_rescale_intensity)
plt.imshow(img_rescale_intensity > t, cmap = "gray")
'''
