# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 08:17:33 2021

@author: Serkan
"""

import os
import numpy as np
import cv2
from scipy.ndimage.filters import convolve
import flow_vis
from matplotlib import pyplot as plt
import glob

def makedir(path):
    if os.path.exists(os.path.join(path)) == False:
        os.mkdir(path)
        print('make :',path)
    else :
        print('{} already exists.'.format(path))

makedir('./frames')
makedir('./estimated')
makedir('./color_flows')
makedir('./plots')
cap = cv2.VideoCapture('signed.avi')
i=0
frames = []

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    gry = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./frames/frame'+str(i)+'.jpg',gry)
    frames.append(gry)
    i+=1

h,w = frames[0].shape
flow = np.zeros((h,w,2), dtype='float32')


def HornSchunck(img1,img2, alpha, iteration):

    HSKernel=np.array([[0, 1/4, 0],
                       [1/4, 0, 1/4],
                       [0, 1/4, 0]], dtype="float32")
    imm1 = img1
    img1 = np.array(img1,dtype="float32")
    img2 = np.array(img2, dtype="float32")

    kernel_x = np.array([[-1,1]],dtype="float32")
    kernel_y = np.array([[-1],[+1]],dtype="float32")

    Ix = convolve(img1,kernel_x)
    Iy = convolve(img1,kernel_y)
    It = np.subtract(img2,img1)

    U = np.zeros_like(img1)
    V = np.zeros_like(img1)

    errs = []
    
    for i in range(iteration):
        u_mean=convolve(U,HSKernel)
        v_mean=convolve(V,HSKernel)
        P=np.multiply(Ix,u_mean)+np.multiply(Iy,v_mean)+It
        D=alpha**2+np.multiply(Ix,Ix)+np.multiply(Iy,Iy)
        U=u_mean-np.divide(np.multiply(Ix,P),D)
        V=v_mean-np.divide(np.multiply(Iy,P),D)
        u_ = (U + np.arange(w)).astype('float32')
        v_ = (V + np.arange(h)[:,np.newaxis]).astype('float32')
        next_frame = cv2.remap(imm1,u_,v_,cv2.INTER_LINEAR)
        err = np.mean(((next_frame-img2)**2).mean(axis=1))
        errs.append(err)
    

    return U,V,errs,next_frame


im1 = frames[0]
im2 = frames[1]
flows = []
ests = []
its = np.arange(1,51,1)
for k in range(len(frames)-1):
    u,v,errs,est = HornSchunck(frames[k], frames[k+1], alpha=70, iteration=50)
    flow_color = flow_vis.flow_uv_to_colors(u, v)
    cv2.imwrite('./color_flows/flow'+str(k)+'.jpg',flow_color)
    flows.append(flow_color)
    cv2.imwrite('./estimated/est'+str(k)+'.jpg',est)
    ests.append(est)    
    
img_array = []
for filename in glob.glob('./estimated/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()



u1,v1,errs1,est1 = HornSchunck(frames[0], frames[1], alpha=70, iteration=50)
plt.figure()
plt.plot(np.arange(1,51,1),errs1)
plt.savefig('./plots/plot1.jpg')
