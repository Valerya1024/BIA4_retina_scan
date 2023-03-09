# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 07:39:11 2021

@author: Valerya
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
from skimage.transform import rescale

n = "001"
directory = "D:/Files/4/BIA4/ICA1/glaucoma/test_set/result/"#"D:/Files/4/BIA4/ICA1/glaucoma/archive/ORIGA/ORIGA/Images/"

im_path = directory+n+"_bloodvessel.png"#".vessel.ridge.maxima.no_edge.jpg"
im = cv2.imread(im_path, 0)
 
if im is None:
	print(im_path, " not exist")
	sys.exit()

ret, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
 
skel = np.zeros(im.shape, np.uint8)
erode = np.zeros(im.shape, np.uint8)
temp = np.zeros(im.shape, np.uint8)
 
i = 0
while True:
    print(i)
    #cv2.imshow('im %d'%(i), im)
    erode = cv2.erode(im,element)
    temp  = cv2.dilate(erode, element)
 
	#消失的像素是skeleton的一部分
    temp = cv2.subtract(im, temp)
    skel = cv2.bitwise_or(skel, temp)
    im = erode.copy()
	
    if cv2.countNonZero(im)== 0:
        break;
    i += 1
    
    
plt.imshow(skel, cmap = "gray")

plt.imsave(directory+n+".vessel.skeleton.png", skel, cmap="gray")


def count_vessel_length(img):
    print("Approximate length:",cv2.countNonZero(img), "pixels")

#if "__name__" == 
count_vessel_length(skel)


