# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 21:21:49 2021

@author: Valerya
"""


import cv2
import numpy as np
import os
import csv

def CLAHE(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def extract_green(img):
    b,g,r = cv2.split(img)
    return g

def extract_bv(img_g):		
    
    clahe_g = CLAHE(img_g)
    
	# applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(clahe_g, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)	
    f4 = cv2.subtract(R3,clahe_g)
    f5 = CLAHE(f4)	

	# removing very small contours through area parameter noise removal
    ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)	
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255	
    contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)			
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)			
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)	

	# removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
	#vessels and also in an interval of area
    fundus_eroded = cv2.bitwise_not(newfin)	
    xmask = np.ones(fundus.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)	
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)   				
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = "circle"	
        else:
            shape = "veins"
        if(shape=="circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)	
	
    blood_vessels = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)	
	#blood_vessels = cv2.bitwise_not(finimage)
    return blood_vessels	

def remove_edge(ori_img, vessel_img, e = 1, r = 9, i = 5):
    
    ret, fundus_edge_mask = cv2.threshold(ori_img, 30, 255, cv2.THRESH_BINARY)
    figure_edge_mask = np.zeros(ori_img.shape)
    figure_edge_mask[e:-e,e:-e] = 1
    mask = fundus_edge_mask * figure_edge_mask
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    #plt.imshow(mask, cmap = "gray")
    mask_erode = cv2.erode(mask, kernel, iterations=10)/255
    #plt.imshow(mask_erode, cmap = "gray")

    vessel_img_masked = mask_erode * vessel_img
    #plt.imshow(img_masked, cmap = "gray")
    
    return vessel_img_masked

def gaussian_blur(img, d, i):
    img_GaussianBlur=cv2.GaussianBlur(img,(d,d),i)
    
    return img_GaussianBlur

def median_blur(img, d, i):
    img_MedianBlur=cv2.MedianBlur(img,d,i)
    
    return img_MedianBlur

if __name__ == "__main__":	
    pathFolder = "D:/Files/4/BIA4/ICA1/glaucoma/test_set"
    filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]
    destinationFolder = "D:/Files/4/BIA4/ICA1/glaucoma/test_set/result/"
    if not os.path.exists(destinationFolder):
        os.mkdir(destinationFolder)
    for file_name in filesArray:
        file_name_no_extension = os.path.splitext(file_name)[0]
        fundus = cv2.imread(pathFolder+'/'+file_name)
        fundus_g = extract_green(fundus)
        bloodvessel = extract_bv(fundus_g)
        bloodvessel_no_edge = remove_edge(fundus_g, bloodvessel)
        bloodvessel_blur = gaussian_blur(bloodvessel_no_edge, 25, 5)
        cv2.imwrite(destinationFolder+file_name_no_extension+"_bloodvessel.png",bloodvessel_blur)
        