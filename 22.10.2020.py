#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os

def get_jpg_files():
    os.getcwd()
    os.listdir()
    path = os.getcwd()
    jpg_files = [f for f in os.listdir(path) if f.endswith('.jpg')]
    return jpg_files
get_jpg_files()

im_1 = plt.imread('wallpaper.jpg')


def get_value_from_triple(temp_1):
    #temp_1 = im_1[0,0,:]
    return int(temp_1[0]/3 + temp_1[1]/3 + temp_1[2]/3)
get_value_from_triple(im_1[0,0,:])

def convert_rgb_to_gray(im_1):
    m,n,k = im_1.shape
    new_image = np.zeros((m,n),dtype='uint8')
    for i in range(m):
        for j in range(n):
            s = get_value_from_triple(im_1[i,j,:])
            new_image[i,j] = s
    return new_image

im_2 = convert_rgb_to_gray(im_1)


plt.subplot(1,2,1)
plt.imshow(im_1)
plt.subplot(1,2,2)
plt.imshow(im_2,cmap='gray')
plt.show

plt.imsave('wallpaper_gray.jpg', im_2)


def get_0_1_from_triple(temp_1):
    #temp_1 = im_1[0,0,:]
    temp = int(temp_1[0]/3 + temp_1[1]/3 + temp_1[2]/3)
    if temp<110:
        return 0
    else:
        return 1
get_0_1_from_triple(im_1[0,0,:])

def convert_rgb_to_bw(im_1):
    m,n,k = im_1.shape
    new_image = np.zeros((m,n),dtype='uint8')
    for i in range(m):
        for j in range(n):
            s = get_0_1_from_triple(im_1[i,j,:])
            new_image[i,j] = s
    return new_image

im_3 = convert_rgb_to_bw(im_1)


plt.subplot(1,3,1)
plt.imshow(im_1)
plt.subplot(1,3,2)
plt.imshow(im_2,cmap='gray')
plt.subplot(1,3,3)
plt.imshow(im_3,cmap='gray')
plt.show

plt.imsave('wallpaper_bw.jpg', im_3)
