#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 14:12:34 2018

@author: Sammi Chekroud
"""

import numpy as np
import scipy as sp

#%%
def besseliln(nu, z):
    w = np.add(np.log(sp.special.ive(nu, z)), np.abs(np.real(z)))
    return w

#%%
def vonmisespdf(x, mu, K):
    p = np.exp( np.subtract(np.subtract(np.multiply(K, np.cos(np.subtract(x,mu))), np.log(2*sp.pi)), besseliln(0, K)))
    return p

#%%
def A1inv(R):
    if np.logical_and(0 <= R, R < 0.53):
        K =2*R + R**3 + (5 * R**5)/6
    elif R < 0.85:
        K = -0.4 + 1.39*R + 0.43/(1-R)
    else:
        K = 1/(R**3 - 4*R**2 + 3*R)
    return K

#%%
def mixmodel_function(X, T, NT, B_start):
    ''' will return B, LL, W'''

    max_iter = 10**4
    max_dLL  = 10**-4
    
    n = X.shape[0]
    
    if NT == []:
        NT = np.zeros([n,0])
        nn = 0
    elif len(NT.shape) == 1:
        nn = 1
    else:
        nn = NT.shape[1]
    
    if B_start == []: #if initialised without starting parameters
        K = 5
        Pt = 0.5
        if nn > 0:
            Pn = 0.3
        else:
            Pn = 0
        Pu = 1 - Pt - Pn
    else:
        K  = B_start[0]
        Pt = B_start[1]
        Pn = B_start[2]
        Pu = B_start[3]
    
    
    E = np.subtract(X,T)
    E = np.subtract(np.mod(np.add(E,sp.pi), 2*sp.pi), sp.pi)
    NE = np.subtract(np.matlib.repmat(X,1,nn), NT)
    
    NE = np.subtract(np.mod(np.add(NE,sp.pi), 2*sp.pi), sp.pi).flatten()
    
    
    LL  = np.nan
    dLL = np.nan
    iteri = 0
    while True:
        iteri += 1
        
        Wt = Pt * vonmisespdf(E,0,K)
        Wg = np.ones(n);    Wg = np.divide(Wg, 2*sp.pi);    Wg = np.multiply(Wg, Pu)
        
        
        if nn == 0:
            Wn = np.zeros(NE.shape)
        else:
            Wn = np.multiply(np.squeeze(vonmisespdf(NE, 0, K)), Pn/nn)
            
        W = np.sum(np.array([Wt, Wn, Wg]),0)
        
        dLL = LL - np.sum(np.log(W))
        LL = np.sum(np.log(W))
        if np.logical_or(abs(dLL) < max_dLL, iteri > max_iter):
            break
        
        Pt = np.divide(np.sum(np.divide(Wt, W)), n)
        
        
        if len(Wn.shape) == 1:
            Pn = np.divide(np.sum(np.divide(Wn, W)),n)
        else:
            Pn = np.divide(np.sum(np.divide(np.sum(Wn,1),W)),n)
        
        Pu = np.divide(np.sum(np.divide(Wg, W)), n)
        
        rw = np.array([np.divide(Wt,W),np.divide(Wn,np.matlib.repmat(W,1,nn)).flatten()]).T
        
        
        S  = np.array([np.sin(E), np.sin(NE)]).T
        C  = np.array([np.cos(E), np.cos(NE)]).T
        r  = np.array([np.sum(np.sum(np.multiply(S,rw))), np.sum(np.sum(np.multiply(C, rw)))])
        
        if np.sum(np.sum(rw)) == 0:
            K = 0
        else:
            R = np.divide(np.sqrt(np.sum(np.power(r, 2))), np.sum(np.sum(rw)))
            K = A1inv(R)
        
        if n <= 15:
            if K < 2:
                K = np.max(K-2/(n * K))
                if K < 0:
                    K = 0
            else:
                K = K * ((n-1)**3)/(n**3 + n)
        
    if iteri > max_iter:
        print('Warning: maximum iteration limit exceeded')
        B  = np.array([np.nan, np.nan, np.nan, np.nan])
        LL = np.nan
        W  = np.nan
    else:
        B = np.array([K, Pt, Pn, Pu])
        if len(Wn.shape) == 1:
            W = np.divide(np.array([Wt, Wn, Wg]), W)
        elif len(Wn.shape) > 1:
            W = np.divide(np.array([Wt, np.sum(Wn,0), Wg]),W)
    
    return B, LL, W
#%%

def mixmodel_fit(X, T, NT):    
    n = X.shape[0]
    
    if np.logical_and(T.size == 0,NT.size == 0):
        T  = np.zeros([n,1])
        NT = np.zeros([n,1])
        nn = 0
    elif np.logical_and(T.size != 0, NT.size == 0):
        NT = np.zeros([n,0])
        nn = 0
    else:
        if len(NT.shape) == 1:
            nn = 1
        else:
            nn = NT.shape[1]
    
    #starting parameters
    K = np.array([   1,  10, 100])
    N = np.array([0.01, 0.1, 0.4])
    U = np.array([0.01, 0.1, 0.4])
    
    if nn == 0:
        N = 0
    
    LL = -np.inf
    B  = np.array([np.nan, np.nan, np.nan, np.nan])
    W  = np.nan 
    for i in range(len(K)):
        for j in range(len(N)):
            for k in range(len(U)):
                b, ll, w = mixmodel_function(X, T, NT, np.array([K[i], 1-N[j]-U[k], N[j], U[k]]))
                if ll > LL:
                    LL = ll
                    B = b
                    W = w
    
    return B, LL, W
#%%
