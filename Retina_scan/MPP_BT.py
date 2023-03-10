# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 04:35:11 2021

@author: Valerya
"""

import cv2
import imageio
#import nibabel as nib
import numpy as np
from p_cost import Cost
from nibabel.viewers import OrthoSlicer3D
import SimpleITK as sitk
import matplotlib.pyplot as plt

# 获取图片的mask像素值矩阵
def read_img(img_path):
    img = nib.load(img_path)
    return img


# 读取nii图片
def read_nii(img_path):
    img = nib.load(img_path)
    return img


# 裁剪图像
def crop_img(image):
    pass


# 获取领域的坐标点
def get_neighbors(p):
    x, y, z = p
    x_left = 0 if x == 0 else x - 1
    x_right = W if x == W - 1 else x + 2
    y_front = 0 if y == 0 else y - 1
    y_back = H if y == H - 1 else y + 2
    z_top = Q if z == Q - 1 else z + 2
    z_bottom = 0 if z == 0 else z - 1

    return [(x, y, z) for x in range(x_left, x_right) for y in range(y_front, y_back)
            for z in range(z_bottom, z_top)]


# dijkstra算法
def dijkstra():
    pass


def path_to_p(end_, steps):
    for i in range(steps):
        top_p = paths[end_]
        end_ = top_p
        if end_ == start_:
            break

    return end_, i + 1


def filter_I_bk():
    for q in range(Q):
        for i in range(W):
            for j in range(H):
                if I_bk[i][j][q] != 0 and I_bk[i][j][q] < l_a:
                    I_bk[i][j][q] = 0
                elif I_bk[i][j][q] != 0 and I_bk[i][j][q] >= l_a:
                    I_bk[i][j][q] = 1


def save_img():
    out = I_bk.astype(np.int16)
    out = nib.Nifti1Image(out, np.eye(4))
    out.to_filename('data_result.nii.gz')


def MPP_BT(min_R, max_R, R_step, intensity_range):
    # 已经处理的集合
    processed = set()
    # 初始化成本函数
    cost_ = Cost(image, min_R, max_R, R_step, intensity_range, W, H, Q)
    start_cost = cost_.p_cost_grayscale(start_)
    # 当前路径的累积成本
    cost_all = {start_: start_cost}
    cost = {start_: start_cost}
    # 计算S(p)
    S_p_all = []

    StopPropagation = False
    l_SM = 0
    i = 0
    while cost and StopPropagation == False:
        # while StopPropagation == False:
        # 每次取出当前成本代价最小值
        p = min(cost, key=cost.get)
        # 获取当前成本最小值的领域节点
        neighbors = get_neighbors(p)
        # 保存已经处理过的点
        processed.add(p)
        # 新扩展点的记录累积成本和回溯跟踪点
        for next_p in [x for x in neighbors if x not in processed]:
            next_p_cost = cost_.p_cost_grayscale(next_p)
            now_cost = next_p_cost + cost[p]
            # 如果该领域点之前计算过了，则需要判断此时所用的代价小还是之前的代价小
            if next_p in cost:
                if now_cost < cost[next_p]:
                    cost_all.pop(next_p)
                    cost.pop(next_p)
            else:
                # 如果该领域点之前没有计算过，或者需要更新
                cost[next_p] = now_cost
                processed.add(next_p)
                paths[next_p] = p

            # 回溯了Ibk步
            pbk, step_ = path_to_p(next_p, lbk)
            P_pbk = cost_all[pbk]
            I_bk[pbk] = I_bk[pbk] + 1 / (n_ + P_pbk)

            if cost_all.get(pbk) != None:
                S_p = step_ / (cost[next_p] - cost_all[pbk])
            else:
                S_p = step_ / (cost[next_p] - cost[pbk])

            if len(S_p_all) < l_ave:
                S_p_all.append(S_p)
            else:
                S_p_all.pop(0)
                S_p_all.append(S_p)

            # 计算NS
            N_S = (sum(S_p_all) / len(S_p_all)) / (max(S_p_all))

            # 判断是否需要停止
            if N_S < NS_min:
                l_SM = l_SM + 1
            else:
                l_SM = 0

            if l_SM == l_E:
                StopPropagation = True

            if i == 50000:
                print(cost)

            # if len(cost_all) >= 15000:
            #     cost_all.pop(0)

        i = i + 1

        cost_all[p] = cost[p]
        cost.pop(p)

    # filter_I_bk()
    save_img()


if __name__ == "__main__":
    global image, W, H, Q, intensity_range
    global start_, paths, lbk, I_bk, NS_min, l_a, l_ave, l_E
    img_path = "D:/Files/4/BIA4/ICA1/glaucoma/test_set/result/001_bloodvessel.png"
    image_ = plt.imread(img_path)
    W, H, Q = image_.dataobj.shape
    # OrthoSlicer3D(image_.dataobj).show()
    # affine = image_.affine
    image = image_.get_fdata().transpose(1, 0, 2)
    # 初始化血管强度
    intensity_range = [1, 600]
    # 初始化参数最小半径，最大半径，回溯步数
    min_R = 2
    max_R = 15
    R_step = 3
    lbk = 10
    l_a = 0.7
    l_ave = 1000
    l_E = 5000
    # 初始化平衡系数k_和y_，血管中轴提取阈值yc
    k_ = 100
    y_ = 1
    n_ = 0.1

    # 初始化停止常数
    NS_min = 0.05
    # NS_o_min = 0

    # 初始化全局变量起始点start_，回溯路径paths，特征图I_bk
    start_ = (256, 256, 137)
    paths = {}
    I_bk = np.zeros([W, H, Q])
    MPP_BT(min_R, max_R, R_step, intensity_range)
    print('finish!')
    # print(image)
