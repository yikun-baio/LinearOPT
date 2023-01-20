#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:04:58 2022

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
import numba as nb 

import scipy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from lib.lib_lot import *



save_path='experiment/geodesic/data/'
data=torch.load(save_path+'data.pt')
X_noise=data['X_noise']
X0=data['X0']

#N=mu_noise[0].shape[0]
ind=2

#Ni=mu_noise[ind].shape[0]

#C=ot.dist(X[ind],X0)

Xi=X_noise[ind]
Ni=Xi.shape[0]

N0=X0.shape[0]
Pi=np.ones(Ni)/Ni
P0=np.ones(N0)/Ni
Vi,P0_Ti =linear_embedding(X0,Xi,P0,Pi,40)

#p=ot.emd(a,b,C) # exact linear program
#Vi=np.matmul((N*p).T,X[ind])-X0

k=6
fig,ax=plt.subplots(1,k,figsize=(k*3,3))
domain=P0_Ti>0
# ax[k-1].scatter(X[ind][:,0],X[ind][:,1],c='r',s=20)


for i,alpha in enumerate(np.linspace(0,1,k-1)):
    sample=X0.copy()
    sample[domain]=sample[domain]+alpha*Vi[domain]
    ax[i].scatter(sample[:,0],sample[:,1],s=5,alpha=1)
    ax[i].set_title(r'$\alpha=$%.02f'%alpha,fontsize=14)
    ax[i].set_xlim(-10,10)
    ax[i].set_ylim(-10,10)

i=5
sample2=Xi
    
ax[i].scatter(sample2[:,0],sample2[:,1],s=5,alpha=1)
ax[i].set_title('Xi',fontsize=14)
ax[i].set_xlim(-10,10)
ax[i].set_ylim(-10,10)

fig.text(0.5, -0.05, 'Transport Geodesic, OPT', ha='center',fontsize=14)
plt.show()