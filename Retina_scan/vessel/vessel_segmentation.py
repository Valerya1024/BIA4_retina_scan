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
from skimage.filters import sobel

n = "001"

directory = "D:/Files/4/BIA4/ICA1/glaucoma/archive/ORIGA/ORIGA/Images/"
img_rgb = plt.imread(directory+n+".jpg")
img_r = img_rgb[:,:,0]
img_g = img_rgb[:,:,1]
img_b = img_rgb[:,:,2]

img_GaussianBlur=cv2.GaussianBlur(img_g,(9,9),1.5)
plt.imshow(img_GaussianBlur, cmap = "gray")

r = 4
i = 5
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * r + 1, 2 * r + 1))
img_blackhat = cv2.morphologyEx(img_GaussianBlur, cv2.MORPH_BLACKHAT, kernel, iterations=i)
plt.imshow(img_blackhat, cmap = "gray")

img_equalized_CLAHE = equalize_adapthist(img_blackhat, clip_limit=0.04)
plt.imshow(img_equalized_CLAHE, cmap = "gray")

ret, mask = cv2.threshold(img_g, 30, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
plt.imshow(mask, cmap = "gray")

mask_erode = cv2.erode(mask, kernel, iterations=10)/255
plt.imshow(mask_erode, cmap = "gray")

img_masked = mask_erode * img_equalized_CLAHE
plt.imshow(img_masked, cmap = "gray")

img_rescale_intensity = rescale_intensity(img_masked, in_range=(0.2, 0.6), out_range=(0, 255))
plt.hist(img_rescale_intensity.ravel(), bins=100, color="black")
plt.imshow(img_rescale_intensity, cmap = "gray")

plt.imsave(directory+n+".vessel.jpg", img_rescale_intensity, cmap="gray")
'''
t = threshold_otsu(img_rescale_intensity)
plt.imshow(img_rescale_intensity > t, cmap = "gray")
'''
