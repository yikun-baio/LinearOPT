#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:19:52 2022

@author: baly
"""


import numpy as np
import torch
import os
import ot

import numba as nb
from typing import Tuple #,List
from numba.typed import List
import matplotlib.pyplot as plt

@nb.njit()
def cost_function(x,y): 
    ''' 
    case 1:
        input:
            x: float number
            y: float number 
        output:
            (x-y)**2: float number 
    case 2: 
        input: 
            x: n*1 float np array
            y: n*1 float np array
        output:
            (x-y)**2 n*1 float np array, whose i-th entry is (x_i-y_i)**2
    '''
#    V=np.square(x-y) #**p
    V=np.power(x-y,2)
    return V




@nb.njit(['float32[:,:](float32[:])','float64[:,:](float64[:])'],fastmath=True)
def transpose(X):
    Dtype=X.dtype
    n=X.shape[0]
    XT=np.zeros((n,1),Dtype)
    for i in range(n):
        XT[i]=X[i]
    return XT

@nb.njit(['float32[:,:](float32[:],float32[:])','float64[:,:](float64[:],float64[:])'],fastmath=True)
def cost_matrix(X,Y):
    '''
    input: 
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
    '''
    n=X.shape[0]
    XT=transpose(X)
    M=cost_function(XT,Y)
    return M


@nb.njit(['float32[:,:](float32[:,:],float32[:,:])','float64[:,:](float64[:,:],float64[:,:])'],fastmath=True)
def cost_matrix_d(X,Y):
    '''
    input: 
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
    '''
    n=X.shape[0]
    m=Y.shape[0]
    Dtype=X.dtype
    M=np.zeros((n,m),Dtype)
    for i in range(n):
        for j in range(m):
            M[i,j]=np.sum(cost_function(X[i,:],Y[j,:]))
    return M

@nb.njit(['int64[:](int64,int64)'],fastmath=True)
def arange(start,end):
    n=end-start
    L=np.zeros(n,np.int64)
    for i in range(n):
        L[i]=i+start
    return L


#@nb.njit(['float32[:](float32[:],float32[:],float32[:])','float64[:](float64[:],float64[:],float64[:])'],fastmath=True)
def quantile_function(qs, mu_com, mu_values):
    n0=mu_com.shape[0]
    n=qs.shape[0]
    index_list=arange(0,n0)
    Dtype=mu_com.dtype
    quantile_value=np.zeros(n,dtype=Dtype)
    quantile_index=np.zeros(n,dtype=np.int64)
    for i in range(n):
        sup_index=index_list[mu_com>=qs[i]][0]
        quantile_value[i]=mu_values[sup_index]
        quantile_index[i]=sup_index
    return quantile_value, quantile_index


def Gromov_total(pi,mu_quantile_value,nu_quantile_value):
    N=pi.shape[0]
    Sum=0
    for i in range(N):
        for j in range(i+1,N):
            D1=cost_function(mu_quantile_value[i],mu_quantile_value[j])
            D2=cost_function(nu_quantile_value[i],nu_quantile_value[j])
            cost=cost_function(D1,D2)
            Sum+=cost
    Sum=Sum*2
    return Sum
    

def Gromov_OT_1d_a(X,Y,mu,nu):
    mu_cum=np.cumsum(mu)
    nu_cum=np.cumsum(nu)
    qs=np.concatenate((mu_cum,nu_cum[:-1]))
    qs.sort()
    qs=np.unique(qs)
    mu_quantile_value, mu_quantile_index=quantile_function(qs, mu_cum, X)
    nu_quantile_value, nu_quantile_index=quantile_function(qs, nu_cum, Y)
    
    pi=np.concatenate((qs[0:1],np.diff(qs)))
    total_cost=Gromov_total(pi,mu_quantile_value,nu_quantile_value)
    plan=np.vstack((pi,mu_quantile_index,nu_quantile_index))
    return total_cost,plan

def plan_to_matrix(plan,n,m):
    T=np.zeros((n,m))
    l=plan.shape[1]
    for i in range(l):
        p=plan[0,i]
        x_index=int(plan[1,i])
        y_index=int(plan[2,i])
        T[x_index,y_index]=p
    return T
        

def Gromov_OT_1d(X,Y,mu,nu):
    mu1=mu
    X1=X
    cost1,mu_quantile_index1,nu_quantile_index1=Gromov_OT_1d_a(X1,Y,mu1,nu)
    mu2=mu[::-1]
    X2=-X[::-1]
    cost2,mu_quantile_index2,nu_quantile_index2=Gromov_OT_1d_a(X2,Y,mu2,nu)

    return min(cost1,cost2)

    
a=-1/6
X=np.array([1.0,2.0,3.0])
mu=np.array([1/3-a,1/3,1/3+a])
Y=np.array([3.0,4.0,5.0])
nu=np.array([1/3,1/3,1/3])
C1=cost_matrix(X,X)
C2=cost_matrix(Y,Y)
T=ot.gromov.gromov_wasserstein(C1, C2, mu, nu,'square_loss')
print('result in Python OT')
print(T)
total_cost, plan=Gromov_OT_1d_a(X,Y,mu,nu)
T1=plan_to_matrix(plan,3,3)
print('Our method')
print(T1)