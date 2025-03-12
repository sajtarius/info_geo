#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:21:59 2024

@author: hengjie

Calculation for the [distance PDF], this function is to calculate the distances between 
the vectos within the interested sampling windows. The calculated sitances matrix (upper trianlge) 
is used to estiamte the PDF via histogram. 

parameters: 
    data: given data should have 2 dimensions numpy array of [T x N], T=time index, N=eigenvector dimension.
    int_bins: histogram bin size. It can be integer or rice or sturges; default is rice.
    int_dist: interested distance should be in string. Any option from sklearn; default is [chebyshev]. 
    int_range: range of the interested axis; default is [0.001 to 2.001]. Tuple or list is needed.
    
return: 
    numpy.ndarray: 1D array of amplitude of histogram. 
    numpy.ndarray: 1D array of range of histogram. 
    float: minimum of the distance. 
    flaot: maximum of the distance. 

"""

import numpy as np

from sklearn.metrics import pairwise_distances


def any_dist_his(data, int_bins='rice', int_dist='chebyshev', int_range=(0.001, 2.001)):
    data = np.array(data)
    assert len(data.shape) == 2, 'data: given data should have 2 dimensions numpy array of [N x W], N=eigenvector dimension, W=window of data.'
    assert (type(int_bins)==int) or (int_bins=='rice') or (int_bins=='sturges'), 'bins: histogram bin size. It can be integer or rice or sturges; default is rice.'
    assert isinstance(int_dist, (str)), 'int_dist: interested distance should be in string. Any option from sklearn; default is [chebyshev]. '
    assert isinstance(int_range, (tuple, list)), 'int_range: range of the interested axis; default is [0.001 to 2.001]. Tuple or list is needed.'
    assert len(int_range) == 2, 'int_range: ONLY 2 values(min and max of axis) are needed.'

    temp_vec = data

    #calculate the [any distances] of the [vectors]
    temp_matrix = pairwise_distances(temp_vec, metric=int_dist)
    temp_loc = np.triu_indices(temp_matrix.shape[0], k=1)
    temp_ans_matrix = temp_matrix[temp_loc]

    if int_bins=='rice':
        temp_bin = int(np.ceil(2*(temp_ans_matrix.shape[0])**(1/3))) #rice rule bin size
    elif int_bins=='sturges': 
        temp_bin = int(1 + np.log2(temp_ans_matrix.shape[0])) #sturges rule bin size
    elif type(int_bins)==int:
        temp_bin = int_bins
    ###############################################################################################
    temp_his_amp, temp_his_axs = np.histogram(temp_ans_matrix, bins=temp_bin, density=True, range=[np.min(int_range), np.max(int_range)])
    temp_his_axs = 0.5 * (temp_his_axs[1:] + temp_his_axs[:-1])
    ###############################################################################################
    
    
    #check the PDF --> sum to ONE. 
    check_pdf = np.sum(temp_his_amp) * np.diff(temp_his_axs)[0]
    if (check_pdf < 0.9) or (check_pdf > 1.1): 
        print(f'PDF not normalized within 0.9 to 1.1: {check_pdf}')
    
    #return [histogram amplitude], [histogram range], [minimum distance], [max distance]
    return temp_his_amp, temp_his_axs, np.min(temp_ans_matrix), np.max(temp_ans_matrix)
