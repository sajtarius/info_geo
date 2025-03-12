#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:50:10 2024

@author: hengjie

calculation of Dispersion Entropy.

parameters: 
    time_series: given time series should be 1D.
    d_val: the delay of embedding should be integer; default is 1.
    m_val: the dimension of embedding should be integer; default is 2.
    c_val: the linear algorithm mapping C value should be integer or float; default is 3.
    norm: normalization of the Shannon entropy. Only accept boolean; True for normalization (default).
    
return: 
    temp_entropy: float; Dispersion Entropy. 
"""

import numpy as np
import scipy.stats as ss

from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import entropy


def map_func(x, c):
    temp_x = np.round(c*x + 0.5)
    return temp_x

def disper_entropy(time_series, d_val=1, m_val=2, c_val=3, norm=True):
    time_series = np.array(time_series)
    time_series = np.squeeze(time_series) 
    assert len(time_series.shape) == 1, 'time_series: given time series should be 1D.'
    assert isinstance(d_val, (int)), 'd_val: the delay of embedding should be integer; default is 1.'
    assert isinstance(m_val, (int)), 'm_val: the dimension of embedding should be integer; default is 2.'
    ### MAKE SURE TO MODIFY THIS IF THE MAPPING FUNCTION CHANGED. ###
    assert isinstance(c_val, (int, float)), 'c_val: the linear algorithm mapping C value should be integer or float; default is 3.'
    assert isinstance(norm, (bool)), 'norm: normalization of the Shannon entropy. Only accept boolean; True for normalization (default).'

    #change of the variables to different variable names. 
    x_list = time_series
    int_d = d_val 
    int_m = m_val 
    int_c = c_val 

    cdf_y = ss.norm.cdf(x_list, loc=np.mean(x_list), scale=np.std(x_list)) #normal cumulative distribution function
    z_list = map_func(cdf_y, int_c) #linear algorithm. CHECK the map_func(); one can modify it if needed. 

    #embedding with the sliding window.
    pi_pattern = sliding_window_view(z_list, window_shape=int_m)
    pi_pattern = pi_pattern[::int_d, :]

    #count the number of the repeated combination.
    unique_rows, counts = np.unique(pi_pattern, axis=0, return_counts=True)

    #forming the PMF for the Shannon entropy calculation. 
    pmf_deno = pi_pattern.shape[0] #denominator of the PDF counting 
    remain_pi_pattern_dim = pi_pattern.shape[0] - counts.shape[0] #count how many does not have the patterns from the combination.
    remain_count = np.zeros(remain_pi_pattern_dim) #form zeros array for those that are not counted. 
    pmf_nume = np.concatenate((counts, remain_count)) #numerator of the PDF counting

    pmf = pmf_nume/pmf_deno #probability mass function (PMF). 

    if norm:
        temp_entropy = entropy(pmf, base=(int_c**(int_m)))
    else: 
        temp_entropy = entropy(pmf)

    return temp_entropy
