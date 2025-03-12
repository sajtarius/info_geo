#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 19:02:20 2024

@author: hengjie

Calculation for the [distance PDF], this function is to calculate the distances between 
the vector within the interested sampling windows. The calculated distances matrix (upper trianlge) 
is used to estiamte the PDF via kernel density estimation. 

parameters: 
    data: given data should have 2 dimensions numpy array of [N x W], N=eigenvector dimension, W=window of data.
    int_bins: histogram bin size. It can be integer or rice or sturges; default is rice.
    int_dist: interested distance should be in string. Any option from sklearn; default is [chebyshev]. 
    int_range: range of the interested axis; default is [0.001 to 2.001]. Tuple or list is needed.
    
return: 
    numpy.ndarray: 1D array of amplitude of histogram. 
    numpy.ndarray: 1D array of range of histogram. 
    float: minimum of the histogram range. 
    flaot: maximum of the histogram range. 

"""

import numpy as np

from sklearn.metrics import pairwise_distances
from KDEpy import FFTKDE

def any_dist_kde(data, int_dist='chebyshev', int_range=(0.001, 2.005), int_bins=1000, int_kernel='biweight', int_bw='scott', add_noise=False):
    data = np.array(data)
    assert len(data.shape) == 2, 'data: given data should have 2 dimensions numpy array of [N x W], N=eigenvector dimension, W=window of data.'
    assert isinstance(int_dist, (str)), 'int_dist: interested distance should be in string. Any option from sklearn; default is [chebyshev].'
    assert isinstance(int_range, (list, tuple)), 'int_range: range of the interested axis; default is [0.001 to 2.005]. Tuple or list is needed.'
    assert len(int_range)==2, 'int_range: ONLY 2 values(min and max of axis) are needed.'
    assert isinstance(int_bins, (int)), 'int_bins: the bin size for the KDE estimation for PDF; default is 1000.'
    assert isinstance(int_kernel, (str)), 'int_kernel: kernel used for the KDE estimation; default is [biweight].'
    assert isinstance(int_bw, (str, int, float)), 'int_bw: bandwidth used for the KDE estimation; default is [scott].'
    assert isinstance(add_noise, (bool)), 'add_noise: adding the noise to ensure the KDE works. Boolean is expected; default is False.'
    
    temp_vec = data

    #calculate the [chebyshev distances] of the [vectors]
    temp_matrix = pairwise_distances(temp_vec, metric=int_dist)
    temp_loc = np.triu_indices(temp_matrix.shape[0], k=1)
    temp_ans_matrix = temp_matrix[temp_loc]
    if add_noise:
        temp_ans_matrix += np.random.normal(0, 1e-6, temp_ans_matrix.shape[0])

    #estimate the PDF through KDE 
    temp_axs_lw = np.min(int_range) #lower limit of the PDF axis
    temp_axs_up = np.max(int_range) #upper limit of the PDF axis
    temp_cheby_axs = np.linspace(temp_axs_lw, temp_axs_up, int_bins) #range of the axis for PDF
    temp_cheby_amp = FFTKDE(kernel=int_kernel, bw=int_bw).fit(temp_ans_matrix).evaluate(temp_cheby_axs) #KDE estimation.

    # return [pdf amplitude], [pdf range], [minimum matrix value], [maximum matrix value]
    return temp_cheby_amp, temp_cheby_axs, np.min(temp_ans_matrix), np.max(temp_ans_matrix)
