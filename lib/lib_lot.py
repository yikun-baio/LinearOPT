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
    XT=transpose(X)
    M=cost_function(XT,Y)
    return M



#@nb.jit(nopython=True)
@nb.njit(['float32[:,:](float32[:,:],float32[:,:])','float64[:,:](float64[:,:],float64[:,:])'],fastmath=True)
def cost_matrix_d(X,Y):
    '''
    input: 
        X: (n,d) float np array
        Y: (m,d) float np array
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




def opt_lp(X,Y,mu,nu,Lambda,numItermax=100000):
    if len(X.shape)==1:
        d=1
    else:
        d=X.shape[1]
    n=X.shape[0]
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
    if d==1:
        cost_M=cost_matrix(X,Y)
    else:        
        cost_M=cost_matrix_d(X,Y)
    cost_M1=np.zeros((n+1,m+1))
    cost_M1[0:n,0:m]=cost_M-Lambda
    plan1=ot.lp.emd(mu1,nu1,cost_M1,numItermax=numItermax)
    plan=plan1[0:n,0:m]
    cost=np.sum(cost_M*plan)
    destroyed_mass=np.sum(mu)+np.sum(nu)-2*np.sum(plan)
    penualty=destroyed_mass*Lambda/2
    return cost,plan,penualty


def opt_pr(X,Y,mu,nu,mass,numItermax=100000):
    n=X.shape[0]
    m=Y.shape[0]
    if len(X.shape)==1:
        cost_M=cost_matrix(X,Y)
    else:
        cost_M=cost_matrx_d(X,Y)

    plan=ot.partial.partial_wasserstein(mu,nu,cost_M,m=mass,nb_dummies=n+m)
    
    cost=np.sum(cost_M*plan)

    return cost,plan

def linear_embedding(X0,Xi,P0,Pi,Lambda):
    if len(X0.shape)==1:
        d=1
    else:
        d=X0.shape[1]

    cost,plan,penualty=opt_lp(X0,Xi,P0,Pi,Lambda)
#   cost,plan=opt_pr()
    n=X0.shape[0]
    domain=np.sum(plan,1)>0
    P0_T=np.sum(plan,1)
    if d==1:
        Fi=np.full(n,np.inf)
        Fi[domain]=Xi.dot(plan)[domain]/P0_T[domain]
    else:
        Fi=np.full((n,d),np.inf)
        Fi[domain]=plan.dot(Xi)[domain]/transpose(P0_T)[domain]
    Vi=Fi-X0
    return Vi,P0_T #,index_i



def linear_embedding_pr(Xi,X0,Pi,P0,Lambda):
    cost,plan=opt_pr(X0,Xi,P0,Pi,Lambda)
    n=X0.shape[0]
    domain=np.sum(plan,1)>0
    P0_T=np.sum(plan,1)
    Fi=np.full(n,np.inf)
    Fi[domain]=plan.dot(Xi)[domain]/P0_T[domain]
    Vi=Fi-X0
    return Vi,P0_T #,index_i

def vector_norm_pr(Vi,P0_T): #,total_mass,Lambda):
    domain=P0_T>0
    Vi_take=Vi[domain]
    norm=np.sum((Vi_take)**2*P0_T[domain])
    return norm


def vector_norm(Vi,P0_Ti,total_mass,Lambda):
    domain=P0_Ti>0
    Vi_take=Vi[domain]
    if len(Vi.shape)==1:
        norm=np.sum((Vi_take)**2*P0_Ti[domain])
    else:
        norm=np.sum(np.sum((Vi_take)**2,1)*P0_Ti[domain])
    penualty=Lambda*(total_mass-np.sum(P0_Ti[domain]))
    return norm, penualty

def vector_plus(Vi,Vj,P0_Ti,P0_Tj):
    P0_Tij=np.minimum(P0_Ti,P0_Tj)
    n=Vi.shape[0]
    domain_ij=P0_Tij>0
    Vi_take=Vi[domain_ij]
    Vj_take=Vj[domain_ij]
    Sum=np.full(n,np.inf)
    Sum[domain_ij]=Vi_take+Vj_take
    return Sum,P0_Tij




n=5
m=5
Xi=np.array([2.0,3.0,4.0])
X0=np.array([0.0,1.0,2.0])
Pi=np.array([1/6,1/3,1/2])
P0=np.array([1/6,1/3,1/2])
Lambda=10.0
#cost,plan,penualty=opt_lp(X0,Xi,P0,Pi,Lambda,numItermax=100000)

#Vi,P0_T,index_i=linear_embedding(Xi,X0,Pi,P0,Lambda)
#norm,penualty1=vector_norm(Vi,P0_T,P0,Lambda)

Lambda=20
step_size=50
k=5
n_list=np.array(range(10,500,step_size))
opt_cost1=np.zeros((n_list.shape[0],k))
opt_cost2=np.zeros((n_list.shape[0],k))

for i in range(n_list.shape[0]):
    n=n_list[i]
    for k_p in range(k):
        X0=np.random.uniform(-20,20,(n,2))+15
        Xi=np.random.uniform(-20,20,(n,2))+0
        Xj=np.random.uniform(-20,20,(n,2))+30
        X0.sort()
        Xi.sort()
        Xj.sort()
        P0=np.ones(n)/n
        Pi=np.ones(n)/n
        Pj=np.ones(n)/n
        cost1,plan1,penualty1=opt_lp(X0,Xi,P0,Pi,Lambda,numItermax=200000)
        opt_cost1[i,k_p]=cost1+penualty1
        Vi,P0_Ti=linear_embedding(X0,Xi,P0,Pi,Lambda)
        cost2,penualty2=vector_norm(Vi,P0_Ti,1,Lambda)
        opt_cost2[i,k_p]=cost2+penualty2

#        Vi,P0_Ti=linear_embedding(Xi,X0,Pi,P0,Lambda)
#        Vj,P0_Tj=linear_embedding(Xj,X0,Pj,P0,Lambda)
#        Vij,P0_Tij=vector_plus(Vi,-Vj,P0_Ti,P0_Tj)
#        cost2,penualty2=vector_norm(Vij,P0_Tij,1,Lambda)
#        opt_cost2[i,k_p]=cost2+penualty2
        # if cost2+penualty2<cost1+penualty1-0.00001:
            #     print('error')
            #     print('cost2',cost2)
    #     print('cost1',cost1)
    #     print('penualty1',penualty1)
    #     print('penualty2',penualty2)
    #     break



fig = plt.figure()
ax = plt.subplot(111)
error=opt_cost2-opt_cost1
error_mean=error.mean(1)
error_std=error.std(1)

plt.plot(n_list,error_mean,'-',c='blue',label='error')
plt.fill_between(n_list,error_mean-1*error_std,error_mean+1*error_std,alpha=0.3)

plt.xlabel("n: size of X")
plt.ylabel("error")
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.13),
          fancybox=True, shadow=True, ncol=3)
#plt.savefig('experiment/test/results/accuracy_error'+str(Lambda)+'.png',format="png",dpi=800,bbox_inches='tight')
plt.show()



for i in range(n_list.shape[0]):
    n=n_list[i]
    mass=(n-5)/n
    for k_p in range(k):
        X0=np.random.uniform(-20,20,n)+15
        Xi=np.random.uniform(-20,20,n)+0
        Xj=np.random.uniform(-20,20,n)+30
        X0.sort()
        Xi.sort()
        Xj.sort()
        P0=np.ones(n)/n
        Pi=np.ones(n)/n
        Pj=np.ones(n)/n
#        print('start')
        cost1,plan1=opt_pr(X0,Xi,P0,Pi,mass,numItermax=200000)
#        print('end')
        opt_cost1[i,k_p]=cost1
        Vi,P0_Ti=linear_embedding_pr(X0,Xi,P0,Pi,mass)
        cost2=vector_norm_pr(Vi,P0_Ti)
        opt_cost2[i,k_p]=cost2

#        Vi,P0_Ti=linear_embedding(Xi,X0,Pi,P0,Lambda)
#        Vj,P0_Tj=linear_embedding(Xj,X0,Pj,P0,Lambda)
#        Vij,P0_Tij=vector_plus(Vi,-Vj,P0_Ti,P0_Tj)
#        cost2,penualty2=vector_norm(Vij,P0_Tij,1,Lambda)
#        opt_cost2[i,k_p]=cost2+penualty2
        # if cost2+penualty2<cost1+penualty1-0.00001:
            #     print('error')
            #     print('cost2',cost2)
    #     print('cost1',cost1)
    #     print('penualty1',penualty1)
    #     print('penualty2',penualty2)
    #     break



fig = plt.figure()
ax = plt.subplot(111)
error=opt_cost2-opt_cost1
error_mean=error.mean(1)
error_std=error.std(1)

plt.plot(n_list,error_mean,'-',c='blue',label='error')
plt.fill_between(n_list,error_mean-1*error_std,error_mean+1*error_std,alpha=0.3)

plt.xlabel("n: size of X")
plt.ylabel("error")
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.13),
          fancybox=True, shadow=True, ncol=3)
#plt.savefig('experiment/test/results/accuracy_error'+str(Lambda)+'.png',format="png",dpi=800,bbox_inches='tight')
plt.show()

# cost_list={}
# cost_list['cost_v2_list']=cost_v2_list
# cost_list['cost_v2a_list']=cost_v2a_list 
# cost_list['cost_pr_list']=cost_pr_list
# cost_list['cost_lp_list']=cost_lp_list

# torch.save(cost_list,'experiment/test/results/accuracy_list'+str(Lambda)+'.pt')

    
    


