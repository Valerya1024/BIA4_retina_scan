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
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

n = "001"

directory = "D:/Files/4/BIA4/ICA1/glaucoma/archive/ORIGA/ORIGA/Images/"
img_rgb = plt.imread(directory+n+".jpg")
img_r = img_rgb[:,:,0]
img_g = img_rgb[:,:,1]
img_b = img_rgb[:,:,2]

def detect_ridges(gray, sigma=1.0):
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges

def plot_images(*images):
    images = list(images)
    n = len(images)
    fig, ax = plt.subplots(ncols=n, sharey=True)
    for i, img in enumerate(images):
        ax[i].imshow(img, cmap='gray')
        ax[i].axis('off')
    plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
    plt.show()

#Gaussian Blur    
img_GaussianBlur=cv2.GaussianBlur(img_g,(15,15),5)
plt.imshow(img_GaussianBlur, cmap = "gray")

#Bilateral
d = 15
sigmaColor = 200
sigmaSpace = 7
img_bilateralFilter = cv2.bilateralFilter(img_g, d, sigmaColor, sigmaSpace)
plt.imshow(img_bilateralFilter, cmap = "gray")

a, b = detect_ridges(img_bilateralFilter, sigma=3.0)

plot_images(img_g, a, b)
plt.hist(a.ravel(), bins=100, color="black")
img_rescale_intensity = rescale_intensity(a, in_range=(0.0004, 0.001), out_range=(0, 255))
plt.imshow(img_rescale_intensity, cmap = "gray")

r = 4
i = 5
ret, mask = cv2.threshold(img_g, 30, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
plt.imshow(mask, cmap = "gray")
mask_erode = cv2.erode(mask, kernel, iterations=10)/255
plt.imshow(mask_erode, cmap = "gray")

img_masked = mask_erode * img_rescale_intensity
plt.imshow(img_masked, cmap = "gray")


plt.imsave(directory+n+".vessel.ridge.jpg", img_masked, cmap="gray")
'''
t = threshold_otsu(img_masked)
img_otsu = img_masked > t
plt.imshow(img_otsu, cmap = "gray")
'''
