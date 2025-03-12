#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:30:28 2024

@author: hengjie

calculation for 2D information rate Shannon entropy. The range of distribution for information rate are sampled with respect of two adjecent windows. 

parameters: 
    sig1: given signal 1 should be in 1D numpy.array with time index.
    sig2: given signal 2 should be in 1D numpy.array with time index.
    time: given time data should be in 1D numpy.array with time index.
    win: window size of the sliding window. Integer is expected; default is 10.
    sld: sliding of the sliding window. Integer is expected; default is 2.
    bins: bins size of the distribution for information rate calculation. Integer is expected; default is 50.
    entro_bins: bins size of the distribution for shannon entropy calculation. Integer is expected; default is 10.
    base: base for the shannon entropy calculation. Integer or float is expected; default is 2.
    norm: normalization of the shannon entropy. Boolean is expected; default is True.
    
return: 
    float: inforate_entropy; 2D information rate Shannon entropy.
    numpy.ndarray: inforate_all_data; information rate data. 
    numpy.ndarray: inforate_all_pmf; PMF of the information rate over time.
"""

import numpy as np 
import info_geo as ig

from numpy.lib.stride_tricks import sliding_window_view
from joblib import Parallel, delayed
from scipy.stats import entropy

def adj2d_inforate_shannon_entro(sig1, sig2, time, win=10, sld=2, bins=50, entro_bins=10, base=2, norm=True):
    sig1 = np.array(sig1)
    sig2 = np.array(sig2)
    time = np.array(time)
    
    sig1 = np.squeeze(sig1)
    sig2 = np.squeeze(sig2)
    time = np.squeeze(time)
    
    assert len(sig1.shape)==1, 'sig1: given signal 1 should be in 1D numpy.array with time index.'
    assert len(sig2.shape)==1, 'sig2: given signal 2 should be in 1D numpy.array with time index.'
    assert len(time.shape)==1, 'time: given time data should be in 1D numpy.array with time index.'
    assert sig1.shape[0] == time.shape[0], 'sig1 and time should have the same dimension.'
    assert sig2.shape[0] == time.shape[0], 'sig2 and time should have the same dimension.'
    assert isinstance(win, int), 'win: window size of the sliding window. Integer is expected; default is 10.'
    assert isinstance(sld, int), 'sld: sliding of the sliding window. Integer is expected; default is 2.'
    assert isinstance(bins, int), 'bins: bins size of the distribution for the information rate calculation. Integer is expected; default is 50.'
    assert isinstance(entro_bins, int), 'entro_bins: bins size of the distribution for shannon entropy calculation. Integer is expected; default is 10.'
    assert isinstance(base, (int, float)), 'base: base for the shannon entropy calculation. Integer or float is expected; default is 2.'
    assert isinstance(norm, bool), 'norm: normalization of the shannon entropy. Boolean is expected; default is True.'

    int_win = win
    int_sld = sld

    time_sliding = sliding_window_view(time, window_shape=int_win)
    time_sliding = time_sliding[::int_sld, :]

    x_sliding = sliding_window_view(sig1, window_shape=int_win)
    x_sliding = x_sliding[::int_sld, :]

    y_sliding = sliding_window_view(sig2, window_shape=int_win)
    y_sliding = y_sliding[::int_sld, :]

    inforate_data, inforate_time = zip(*Parallel(n_jobs=-1)(delayed(ig.adj2d_collect_inforate_square)(
            np.expand_dims(y_sliding, axis=0), 
            np.expand_dims(x_sliding, axis=0), 
            time_sliding, 
            i=t, 
            bins_size=bins
        ) for t in range(time_sliding.shape[0] - 1)
    ))
    inforate_data = np.array(inforate_data)
    inforate_time = np.array(inforate_time)


    temp_inforate_pdf, temp_inforate_range = np.histogram(inforate_data**0.5, bins=entro_bins, density=True, )
    temp_inforate_range = (temp_inforate_range[1:] + temp_inforate_range[:-1])/2
    temp_inforate_pmf = temp_inforate_pdf * np.diff(temp_inforate_range)[0]

    if norm==True:
        inforate_entropy = entropy(temp_inforate_pmf, base=base)/entropy(np.ones(temp_inforate_pmf.shape[0])/temp_inforate_pmf.shape[0], base=base)
    elif norm==False:
        inforate_entropy = entropy(temp_inforate_pmf, base=base)
    #print(f'inforate_entropy: {inforate_entropy}')

    inforate_all_data = np.vstack((inforate_data, inforate_time)).T
    inforate_all_pmf = np.vstack((temp_inforate_pmf, temp_inforate_range)).T

    #[information rate shannon entropy], [information rate data], [PMF for the information rate shannon entropy]
    return inforate_entropy, inforate_all_data, inforate_all_pmf