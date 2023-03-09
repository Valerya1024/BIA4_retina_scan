# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 17:08:49 2021

@author: Valerya
"""

## remove noise

import matplotlib.pyplot as plt
import numpy as np
import cv2

from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot


n = "001"
directory = "D:/Files/4/BIA4/ICA1/glaucoma/test_set/result/"#"D:/Files/4/BIA4/ICA1/glaucoma/archive/ORIGA/ORIGA/Images/"

img = cv2.imread(directory+n+".vessel.skeleton.png",0)

vessel_points = []

def find_neighbours(img, x, y, r):
    return img[i-r:i+r+1,j-r:j+r+1]

def is_neighbour(point1, point2, d=1):
    neighbour = False
    if abs(point1[0]-point2[0]) <= d:
        if abs(point1[0]-point2[0]) <= d:
            neighbour = True
    return neighbour

## denoising
for i in range(1,img.shape[0]):
    for j in range(1,img.shape[1]):
        if img[i,j] != 0:
            num = cv2.countNonZero(find_neighbours(img, i, j, r=1))
            if num == 1:
                img[i,j] = 0
            else:
                vessel_points.append((i,j))

plt.imsave(directory+n+".vessel.skeleton.denoise.png", img, cmap="gray")

## plot circle
r = 8
mask_circle = np.zeros((2*r+1,2*r+1))
cv2.circle(mask_circle, (r,r), r, 1, 1)

mask_round = np.zeros((2*r+1,2*r+1))
cv2.circle(mask_round, (r,r), r, 1, -1)

#plt.imshow(mask_round, cmap="gray")

## detect_bifurcation

#for point in vessel_points:
point = vessel_points[0]
neighbours = find_neighbours(img, point[0], point[1], r)
plt.imshow(neighbours)
crossed = cv2.bitwise_and(neighbours, mask_circle)
if cv2.countNonZero(crossed) >= 3:
    img[point[0], point[1]] = 0
'''   
model = AffinityPropagation(damping=0.9)
# 匹配模型
model.fit(X)
# 为每个示例分配一个集群
yhat = model.predict(X)
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
for cluster in clusters:
# 获取此群集的示例的行索引
row_ix = where(yhat == cluster)
# 创建这些样本的散布
pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# 绘制散点图
pyplot.show()
'''


