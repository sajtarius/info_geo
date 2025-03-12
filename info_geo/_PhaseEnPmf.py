#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:54:10 2024

@author: hengjie

Estimate the probability mass function (PMF) of the second-order difference plot 
for the Phase Entropy. 

return array of PMF. NOTE the range of PMF is just the number of bins.

parameters:
    Sig: array of 1D signal; numpy.array is expected. 
    K: number of bin size or number of section in second-order difference plot. Even number is suggested to be used; default is 80. 
    tau: number of time delay for embedding into second-order difference plot. 
    
return: 
    numpy.ndarray: PMF
    
*NOTE: the code is from EntropyHub Phase Entropy (PhasEn())

"""

import numpy as np

def phase_en_pmf(Sig, K=80, tau=1):
    Sig = np.squeeze(Sig)
   
    assert Sig.shape[0]>10 and Sig.ndim == 1,  "Sig:   must be a numpy vector"
    assert isinstance(K,int) and (K > 1), "K:     must be an integer > 1"
    assert isinstance(tau,int) and (tau > 0), "tau:   must be an integer > 0"                         
        
    Yn = Sig[2*tau:] - Sig[tau:-tau]
    Xn = Sig[tau:-tau] - Sig[:-2*tau]    
    with np.errstate(divide='ignore', invalid='ignore'):
        Theta_r = np.arctan(Yn/Xn)   
        Theta_r[np.logical_and((Yn<0),(Xn<0))] += np.pi
        Theta_r[np.logical_and((Yn<0),(Xn>0))] += 2*np.pi
        Theta_r[np.logical_and((Yn>0),(Xn<0))] += np.pi
    
    #Limx = np.ceil(np.max(np.abs([Yn, Xn])))
    Angs = np.linspace(0,2*np.pi,K+1)
    Tx = np.zeros((K,len(Theta_r)))
    Si = np.zeros(K)
    
    for n in range(K):
        Temp = np.logical_and((Theta_r > Angs[n]), (Theta_r < Angs[n+1]))
        Tx[n,Temp] = 1
        Si[n] = np.sum(Theta_r[Temp])
            
    return Si/np.sum(Si)
