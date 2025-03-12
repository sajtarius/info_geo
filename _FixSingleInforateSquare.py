#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:57:56 2024

@author: hengjie

The [information rate] is computed on a single signal for a fixed range. 

Return the information rate square for 2D distribution. 

parameters:
    data: array of data with 3 dimensions; NxTxW, N=signals #, T=time index, W=window of data. 
    time: array of time data with 2 dimensions; TxW, T=time index, W=window of data. 
    int_bins: bins size of the distribution estimation; it should be integer, 2-tuple, 2-list. Default is 30.
    int_range: range of the distribution estimationl; it should be integer, 2-tuple, 2-list. Default is 1.05.
    
return: 
    numpy.ndarray: information rate square
    numpy.ndarray: time
    
"""

import numpy as np

# PDF estimation through histogram
def histogram_func(data, int_range=1.05, int_bins=30):
    assert isinstance(data, (np.ndarray)), 'data: given data should be in numpy.array.'
    assert len(data.shape) == 1, 'data: given data should only have one dimension of sampled data.'
    assert isinstance(int_range, (int, float)), 'int_range: range of the distribution should be in integer or float number; default is 1.05.'
    assert isinstance(int_bins, (int)), 'int_bins: bins size of the distribution should be in integer; default is 30.'

    temp_pdf, temp_range = np.histogram(data, range=(-1*int_range, int_range), bins=int_bins, density=True)
    temp_range = (temp_range[1:] + temp_range[:-1])/2
    
    return temp_pdf, temp_range

# Compute the [information rate]
def fix_single_inforate_square(data, time_data, int_bins=30, int_range=1.05):
    data = np.array(data)
    time_data = np.array(time_data)
    
    assert isinstance(data, (np.ndarray)), 'data: given data should be in numpy.array.'
    assert len(data.shape) == 2, 'data: given data should have 2 dimensions; TxW.'
    assert isinstance(time_data, np.ndarray), 'time_data: given time data should be in numpy.array.'
    assert len(time_data.shape) == 2, 'time_data: given time data should have 2 dimensions; TxW.'
    assert data.shape[0]==time_data.shape[0], 'BOTH data and time_data should have the same sliding window.'
    #assert data.shape[1]==time_data.shape[1], 'BOTH data and time_data should have the same sliding window.'
    assert isinstance(int_bins, (int)), 'int_bins: bin size of the distribution; integer is expected.'
    assert isinstance(int_range, (float, int)), 'int_range: range of the distribution; float or integer is expected. Default is 1.05.'

    #Probability distribution estimation
    temp_pdf_chnl, temp_range_chnl = zip(*np.apply_along_axis(histogram_func, axis=-1, arr=data, int_range=int_range, int_bins=int_bins))
    temp_pdf_chnl = np.array(temp_pdf_chnl)
    temp_range_chnl = np.array(temp_range_chnl)
    
    #information rate square calculation 
    temp_pdf_chnl_square = np.sqrt(temp_pdf_chnl)
    diff_pdf_chnl_square = np.diff(temp_pdf_chnl_square, axis=0)
    diff_range = np.diff(temp_range_chnl, axis=1)[0][0]
    diff_time = np.diff(time_data, axis=0)[0][0]
    inforate_data = 4 * np.sum(diff_pdf_chnl_square**(2), axis=-1) * (diff_range / (diff_time**2))
    
    return inforate_data, time_data[:-1, 0]
