#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:23:03 2022

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


np.random.seed(1)
M=100 # Number of data samples
X=list()
for i in range(M):
#    data,y=make_circles(n_samples=int(50*np.random.rand()+50),noise=.1,factor=.95)
    data,y=make_circles(n_samples=70,noise=.1,factor=.95)
    translate=np.zeros((1,2))
    if np.random.rand()>.5:
      translate[:,1]=10*(np.random.rand()-.5)
    else:
      translate[:,0]=10*(np.random.rand()-.5)
    data=(np.random.rand()+.5)*data+translate
    X.append(data)

k=5
fig,ax=plt.subplots(1,k,figsize=(k*3,3))
for i in range(k):
    ax[i].scatter(X[i][:,0],X[i][:,1],s=5)
    ax[i].set_title(r'$N_%d$=%d'%(i+1,X[i].shape[0]),fontsize=14)
    ax[i].set_xlim(-6,6)
    ax[i].set_ylim(-6,6)
fig.text(0.5, -0.05, 'Samples from the Dataset', ha='center',fontsize=14)
plt.show()


N=int(np.asarray([x.shape[0] for x in X]).mean()) #/5.)
mu0=.25*np.random.randn(N,2)

fig=plt.figure(figsize=(3,3))
plt.scatter(mu0[:,0],mu0[:,1],s=5)
plt.xlim((-6,6))
plt.ylim((-6,6))
plt.title('Template',fontsize=14)

data={}
data['X']=X
data['X0']=mu0
save_path='experiment/geodesic/data'
torch.save(data,save_path+'/data.pt')