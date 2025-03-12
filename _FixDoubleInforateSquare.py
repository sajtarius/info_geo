#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:07:25 2024

@author: hengjie

This code compute the information rate square for 2D distribution at the fix range. 
This code can accommodate two data with different shapes. 
The bins size and the range of the distribution allow two different value instead of just one single bins size or range of distribution. 

Return the information rate square for 2D distribution. 

parameters:
    data1: array of data with 3 dimensions; NxTxW, N=signals #, T=time index, W=window of data. 
    data2: array of data with 3 dimensions; NxTxW, N=signals #, T=time index, W=window of data. 
    time: array of time data with 2 dimensions; TxW, T=time index, W=window of data. 
    int_bins: bins size of the distribution estimation; it should be integer, 2-tuple, 2-list. Default is 30.
    int_range: range of the distribution estimationl; it should be integer, 2-tuple, 2-list. Default is 1.05.
    
return: 
    numpy.ndarray: information rate square
    numpy.ndarray: time
    
"""

import numpy as np

from joblib import Parallel, delayed

# PDF estimation through histogram
def histogram2d_func(data1, data2, int_range=1.05, int_bins=30):
    assert isinstance(data1, (np.ndarray)), 'data1: given data should be in numpy.array.'
    assert len(data1.shape) == 2, 'data1: given data should have two dimensions of sampled data; CxW, C=channel, W=window of data.'
    assert isinstance(data2, (np.ndarray)), 'data2: given data should be in numpy.array.'
    assert len(data2.shape) == 2, 'data2: given data should have two dimensions of sampled data; CxW, C=channel, W=window of data.'
    assert (type(int_range)==int) or (type(int_range)==float) or (type(int_range)==tuple) or (type(int_range)==list), 'int_range: range of the distribution should be in integer, float, tuple, or list; default is 1.05.'
    assert (type(int_bins)==int) or (type(int_bins)==tuple) or (type(int_bins)==list), 'int_bins: bins size of the distribution can be integer, tuple, or list; default is 30.'

    # make the data to be same size by duplicate the data till it has same size. 
    if data1.shape != data2.shape: 
        temp_data1 = np.tile(data1, (data2.shape[0], 1))
        temp_data2 = np.tile(data2, (data1.shape[0], 1))
    else: 
        temp_data1 = data1 
        temp_data2 = data2

    if (type(int_range)==tuple) or (type(int_range)==list): 
        int_range1 = int_range[0]
        int_range2 = int_range[1]
    else: 
        int_range1 = int_range 
        int_range2 = int_range

    temp_pdf, temp_rangex, temp_rangey = np.histogram2d(temp_data1.flatten(), temp_data2.flatten(), range=((-1*int_range1, int_range1), (-1*int_range2, int_range2)), bins=int_bins, density=True)
    temp_rangex = (temp_rangex[1:] + temp_rangex[:-1])/2
    temp_rangey = (temp_rangey[1:] + temp_rangey[:-1])/2
    
    return temp_pdf, temp_rangex, temp_rangey

# Compute the [information rate]
def fix_double_inforate_square(data1, data2, time_data, int_bins=30, int_range=1.05):
    data1 = np.array(data1)
    data2 = np.array(data2)
    time_data = np.array(time_data)
    
    assert isinstance(data1, (np.ndarray)), 'data1: given data should be in numpy.array.'
    assert len(data1.shape) == 3, 'data1: given data should have 3 dimensions; CxTxW, C=channel, T=time index, W=window of data.'
    assert isinstance(data2, (np.ndarray)), 'data2: given data should be in numpy.array.'
    assert len(data2.shape) == 3, 'data2: given data should have 3 dimensions; CxTxW, C=channel, T=time index, W=window of data.'
    assert isinstance(time_data, (np.ndarray)), 'time_data: given time data should be in numpy.array.'
    assert len(time_data.shape) == 2, 'time_data: given time data should have 2 dimensions; TxW, T=time index, W=window of data.'
    assert data1.shape[1]==time_data.shape[0], 'BOTH data1 and time_data should have the same sliding window.'
    assert data2.shape[1]==time_data.shape[0], 'BOTH data2 and time_data should have the same sliding window.'
    assert data1.shape[2]==time_data.shape[1], 'BOTH data1 and time_data should have the same sliding window.'
    assert data2.shape[2]==time_data.shape[1], 'BOTH data2 and time_data should have the same sliding window.'
    assert (type(int_bins)==int) or (type(int_bins)==tuple) or (type(int_bins)==list), 'int_bins: bins size of the distribution; integer, tuple, or list is expected.'
    assert (type(int_range)==float) or (type(int_range)==int) or (type(int_range)==tuple) or (type(int_range)==list), 'int_range: range of the distribution; float, integer, tuple, or list is expected. Default is 1.05.'

    # estimate the series of distribution 
    '''
    given data1 and data2 need to have the same sliding window.
    '''
    if data1.shape[1] == data2.shape[1]: 
        temp_pdf2d_array, temp_rangex_array, temp_rangey_array = zip(*Parallel(n_jobs=-1)(delayed(histogram2d_func)(data1[:, i, :], data2[:, i, :], int_range=int_range, int_bins=int_bins) for i in range(data1.shape[1])))
    else: 
        print('Sliding window of two signals must be the same.')

    temp_pdf2d_array = np.array(temp_pdf2d_array)
    temp_rangex_array = np.array(temp_rangex_array)
    temp_rangey_array = np.array(temp_rangey_array)

    # information rate square calculation
    diff_pdf2d_array = np.diff((temp_pdf2d_array)**(0.5), axis=0)
    delta_rangex_array = np.diff(temp_rangex_array, axis=-1)[0][0]
    delta_rangey_array = np.diff(temp_rangey_array, axis=-1)[0][0]
    delta_time_array = np.diff(time_data, axis=0)[0][0]

    temp_inforate_square2d = 4 * np.sum(diff_pdf2d_array**(2), axis=(1, 2)) * ((delta_rangex_array*delta_rangey_array)/(delta_time_array**2))

    return temp_inforate_square2d, time_data[:-1, 0]