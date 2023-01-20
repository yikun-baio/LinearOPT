#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:52:28 2022

@author: baly
"""

# # -*- coding: utf-8 -*-
# """
# Created on Tue Apr 19 11:32:17 2022

# @author: laoba
# """

import numpy as np
import torch
import os
import ot

import numba as nb
#from typing import Tuple #,List
from numba.typed import List
import matplotlib.pyplot as plt

epsilon=1e-10
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

# @nb.njit(['float32[:,:](float32[:])','float64[:,:](float64[:])'],fastmath=True)
# def transpose(X):
#     Dtype=X.dtype
#     n=X.shape[0]
#     XT=np.zeros((n,1),Dtype)
#     for i in range(n):
#         XT[i]=X[i]
#     return XT

@nb.njit(['float32[:,:](float32[:],float32[:])','float64[:,:](float64[:],float64[:])'],fastmath=True)
def cost_matrix(X,Y):
    '''
    input: 
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
    '''
    X1=np.expand_dims(X,1)
    Y1=np.expand_dims(Y,0)
    M=cost_function(X1,Y1)
    return M


#@nb.njit(fastmath=True)
@nb.njit(['float32[:,:](float32[:,:],float32[:,:])','float64[:,:](float64[:,:],float64[:,:])'],fastmath=True)
def cost_matrix_d(X,Y):
    '''
    input: 
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
    '''
#    n,d=X.shape
#    m=Y.shape[0]
#    M=np.zeros((n,m)) 
    # for i in range(d):
    #     C=cost_function(X[:,i:i+1],Y[:,i])
    #     M+=C
    X1=np.expand_dims(X,1)
    Y1=np.expand_dims(Y,0)
    M=np.sum(cost_function(X1,Y1),2)
    return M



def opt_lp(X,Y,mu,nu,Lambda,numItermax=100000):
    n,d=X.shape
    m=Y.shape[0]
    mass_mu=np.sum(mu)
    mass_nu=np.sum(nu)
    exp_point=np.inf
    mu1=np.zeros(n+1)
    nu1=np.zeros(m+1)
    mu1[0:n]=mu
    nu1[0:m]=nu
    mu1[-1]=mass_nu
    nu1[-1]=mass_mu       
    cost_M=cost_matrix_d(X,Y)
    cost_M1=np.zeros((n+1,m+1))
    cost_M1[0:n,0:m]=cost_M-2*Lambda
    gamma1=ot.lp.emd(mu1,nu1,cost_M1,numItermax=numItermax,numThreads=10)
    gamma=gamma1[0:n,0:m]
    cost=np.sum(cost_M*gamma)
    destroyed_mass=np.sum(mu)+np.sum(nu)-2*np.sum(gamma)
    penualty=destroyed_mass*Lambda
    return cost,gamma,penualty


def opt_pr(X,Y,mu,nu,mass,numItermax=100000):
    n,d=X.shape
    m=Y.shape[0]
    cost_M=cost_matrix_d(X,Y)
    gamma=ot.partial.partial_wasserstein(mu,nu,cost_M,m=mass,nb_dummies=n+m)
    cost=np.sum(cost_M*gamma)
    return cost,gamma

def lopt_embedding(X0,X1,p0,p1,Lambda,numItermax=100000):
    n,d=X0.shape
    cost,gamma,penualty=opt_lp(X0,X1,p0,p1,Lambda)
#   cost,plan=opt_pr()
    N0=X0.shape[0]
    domain=np.sum(gamma,1)>1e-10
    p1_hat=np.sum(gamma,1) # martial of plan 
    # compute barycentric projetion 
    X1_hat=np.zeros((N0,d)) 
    X1_hat[domain]=gamma.dot(X1)[domain]/np.expand_dims(p1_hat,1)[domain]
    
    # separate barycentric into U_1 and p1_hat,M1
    U1=X1_hat-X0
    M1=np.sum(p1)-np.sum(p1_hat)
    return U1,p1_hat,M1

def lopt_embedding_pr(Xi,X0,p1,p0,Lambda):
    n,d=X0.shape
    cost,gamma=opt_pr(X0,Xi,p0,p1,Lambda)
    n=X0.shape[0]
    domain=np.sum(gamma,1)>1e-10
    p1_hat=np.sum(gamma,1)
    Xi_hat=np.full((n,d),np.inf)
    Xi_hat[domain]=gamma.dot(Xi)[domain]/np.expand_dims(p1_hat,1)[domain]
    U1=Xi_hat-X0
    return U1,p1_hat


# def vector_norm(U1,p1_hat): 
#     norm2=np.sum((U1.T)**2*p1_hat[domain])
#     return norm2

# def vector_penalty(p0_hat,p1_hat,M1,Lambda): 
#     penalty=Lambda*(np.abs(p0_hat-p1_hat)+M1)
#     return penalty

# def vector_minus(U1,U2,p1_hat,P2_hat):
#     p1j_hat=np.minimum(p1_hat,P2_hat)
#     diff=np.full(n,0)
#     diff=U1-U2
#     return diff,P_ij


def lopt(U1,U2,p1_hat,p2_hat,Lambda,M1=.0,M2=.0):
    p12_hat=np.minimum(p1_hat,p2_hat)
    norm2=np.sum(np.minimum(np.sum((U1-U2)**2,1),2*Lambda)*p12_hat)
    penualty=Lambda*(np.sum(np.abs(p1_hat-p2_hat))+M1+M2)
    return norm2, penualty

    
    
    


# def vector_norm(Vi,p0_Ti,total_mass,Lambda):
#     domain=p0_Ti>0
#     Vi_take=Vi[domain]
#     if len(Vi.shape)==1:
#         norm=np.sum((Vi_take)**2*p0_Ti[domain])
#     else:
#         norm=np.sum(np.sum((Vi_take)**2,1)*p0_Ti[domain])
#     penualty=Lambda*(total_mass-np.sum(p0_Ti[domain]))
#     return norm, penualty







    


