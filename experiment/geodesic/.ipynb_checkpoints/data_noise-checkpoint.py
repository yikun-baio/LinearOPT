#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:29:07 2022

@author: baly
"""


import numpy as np
import torch
import os
import sys
#import numba as nb
from typing import Tuple #,List
from numba.typed import List
import ot
work_path=os.path.dirname(__file__)
loc1=work_path.find('/LinearOPT')
parent_path=work_path[0:loc1+10]
sys.path.append(parent_path)
os.chdir(parent_path)

from sklearn.datasets import make_swiss_roll, make_moons, make_circles
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from celluloid import Camera
from IPython.display import HTML
import ot

save_path='experiment/geodesic/data/'
data=torch.load(save_path+'data.pt')
X=data['X']
X0=data['X0']
# Number of data samples
M=len(X)
np.random.seed(1)
noise=np.random.uniform([[-6,6],[-6,6]])

per=0.2
n=int(per*X[0].shape[0]) # size of noise 
X_noise=list()

for i in range(M):
    Xi=X[i]
    noise_i=np.random.uniform(low=[-6*2,-6*2],high=[6*2,6*2],size=(n,2))
    X_hat_i=np.concatenate((Xi,noise_i))
    X_noise.append(X_hat_i)


# M=100 # Number of data samples
# X=list()
# for i in range(M):
#     data,y=make_circles(n_samples=int(50*np.random.rand()+50),noise=.1,factor=.95)
#     translate=np.zeros((1,2))
#     if np.random.rand()>.5:
#       translate[:,1]=10*(np.random.rand()-.5)
#     else:
#       translate[:,0]=10*(np.random.rand()-.5)
#     data=(np.random.rand()+.5)*data+translate
#     X.append(data)

k=5
fig,ax=plt.subplots(1,k,figsize=(k*3,3))
for i in range(k):
    ax[i].scatter(X_noise[i][:,0],X_noise[i][:,1],s=5)
    ax[i].set_title(r'$N_%d$=%d'%(i+1,X_noise[i].shape[0]),fontsize=14)
#    ax[i].set_xlim(-6,6)
#    ax[i].set_ylim(-6,6)
fig.text(0.5, -0.05, 'Samples from the Dataset', ha='center',fontsize=14)
plt.show()

data['X_noise']=X_noise
#data['mu0']=mu0
save_path='experiment/geodesic/data'
torch.save(data,save_path+'/data.pt')