#!/usr/bin/env python
# coding: utf-8


import os
os.getcwd(),os.listdir()


path=r"C:\Users\Batuhan"
file_name_with_path=path+"\cameraman.jpg"
file_name_with_path


import matplotlib.pyplot as plt
import numpy as np


img_0=plt.imread(file_name_with_path)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(img_0)
plt.show


np.min(img_2),np.max(img_2)


img_0.ndim,img_0.shape


img_1.ndim,img_1.shape


def convert_rgb_to_gray_level(im_1):
    m=im_1.shape[0]
    n=im_1.shape[1]
    im_2=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            im_2[i,j]=get_distance(im_1[i,j,:])
    return im_2
def get_distance(v,w=[1/3,1/3,1/3]):
    a,b,c=v[0],v[1],v[2]
    w1,w2,w3=w[0],w[1],w[2]
    d=((a**2)*w1 + (b**2)*w2 + (c**2)*w3)**.5
    return d


def my_f_1(a,b):
    assert a>=0;"intensity positive","error intensity not positive"
    if(a<=255-b):
        return a+b
    else:
        return 255
my_f_1(244,30)


def my_f_2(a):
    #assert a>=0;"intensity positive","error intensity not positive"
    return int(255-a)
my_f_2(243)


img_1=convert_rgb_to_gray_level(img_0)
plt.imshow(img_1,cmap='gray')
plt.show()


m,n=img_1.shape
img_2=np.zeros((m,n),dtype="uint8")


for i in range(m):
    for j in range(n):
        ### intensity=img_1[i,j]
        ### increment=50
        ### img_2[i,j]=my_f_1(intensity,increment)
        intensity=img_1[i,j]
        img_2[i,j]=my_f_2(intensity)


plt.subplot(2,2,1),plt.imshow(img_0,cmap='gray')
plt.subplot(2,2,2),plt.imshow(img_1,cmap='gray')
plt.subplot(2,2,3),plt.imshow(img_2,cmap='gray')
plt.show


x=np.array(list(range(100)))
#y=np.array(list(range(100)))
#y=np.sin(np.array(list(range(100))))
#y=1/(1+np.exp(x))
y1=np.power(x/float(np.max(x)),1)
y2=np.power(x/float(np.max(x)),10)
y3=np.power(x/float(np.max(x)),1/10)

plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)


def my_f_3(image_001,gamma):
    return np.power(image_001/float(np.max(image_001)),gamma)
    
x=img_0
img_100=np.power(x/float(np.max(x)),1/10)
plt.imshow(img_100)
plt.show
