#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:51:28 2024

@author: hengjie
The [information rate] is computed on a collective signals for a fixed range.

Return the information rate square for 1D distribution.

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
    assert len(data.shape) == 1, 'data: given data should only have one dimension of sampled data.'
    assert (type(int_range)==int) or (type(int_range)==float) or (type(int_range)==tuple) or (type(int_range)==list), 'int_range: range of the distribution should be in integer, float, tuple, or list; default is 1.05.'
    assert (type(int_bins)==int) or (int_bins=='rice') or (int_bins=='sturges'), 'int_bins: bins size of the distribution can be integer, tuple, or list; default is 30.'

    if (type(int_range)==list) or (type(int_range)==tuple):
        int_range1 = int_range[0]
        int_range2 = int_range[1]
    else:
        int_range1 = int_range
        int_range2 = -1*int_range

    if int_bins=='rice':
        temp_bins = int(np.ceil(2*data.shape[0]**(1/3)))
    elif int_bins=='sturges':
        temp_bins = int(np.ceil(np.log2(data.shape[0])) + 1)
    elif type(int_bins)==int:
        temp_bins = int_bins

    temp_pdf, temp_range = np.histogram(data, range=(int_range1, int_range2), bins=temp_bins, density=True)
    temp_range = (temp_range[1:] + temp_range[:-1])/2

    return temp_pdf, temp_range

# Compute the [information rate]
def fix_collect_inforate_square(data, time_data, int_range=1.05, int_bins=30):
    data = np.array(data)
    time_data = np.array(time_data)

    assert len(data.shape) == 3, 'data: given data should have 3 dimensions; CxTxW, C=channel, T=time index, W=window size.'
    assert len(time_data.shape) == 2, 'time_data: given time data should have 2 dimensions; TxW.'
    assert data.shape[1]==time_data.shape[0], 'BOTH data and time_data should have the same sliding window.'
    #assert data.shape[2]==time_data.shape[1], 'BOTH data and time_data should have the same sliding window.'
    assert (type(int_range)==int) or (type(int_range)==float) or (type(int_range)==tuple) or (type(int_range)==list), 'int_range: range of the distribution should be in integer, float, tuple, or list; default is 1.05.'
    assert (type(int_bins)==int) or (int_bins=='rice') or (int_bins=='sturges'), 'int_bins: bins size of the distribution can be integer, tuple, or list; default is 30.'

    #Probability distribution estimation
    data = np.transpose(data, (1, 2, 0))
    data = np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2])) #reshape is used to make the CxT into ONE vector for histogram_func().
    temp_pdf_chnl, temp_range_chnl = zip(*np.apply_along_axis(histogram_func, axis=-1, arr=data, int_range=int_range, int_bins=int_bins))
    temp_pdf_chnl = np.array(temp_pdf_chnl)
    temp_range_chnl = np.array(temp_range_chnl)

    print(f'dimension of pdf: {temp_pdf_chnl.shape}')
    print(f'dimension of range: {temp_range_chnl.shape}')

    #information rate square calculation
    temp_pdf_chnl_square = np.sqrt(temp_pdf_chnl)
    diff_pdf_chnl_square = np.diff(temp_pdf_chnl_square, axis=0)
    diff_range = np.diff(temp_range_chnl, axis=1)[0][0]
    diff_time = np.diff(time_data, axis=0)[0][0]
    inforate_data = 4 * np.sum(diff_pdf_chnl_square**(2), axis=-1) * (diff_range / (diff_time**2))

    return inforate_data, time_data[:-1, 0]


