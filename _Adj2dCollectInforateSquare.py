#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 18:23:18 2024

@author: hengjie

The [information rate] here is evaluated by having the sampling the probability distribution (PDF)
at the ensemble of TWO regions instead of just one single signal or one single region. 
For instance, NxW of data would be sampled to form a PDF to evaluate the [information rate]. 

Return the information rate square for 2D distribution. 

parameters:
    data1: array of data with 3 dimensions; NxTxW, N=signals #, T=time index, W=window of data. 
    data2: array of data with 3 dimensions; NxTxW, N=signals #, T=time index, W=window of data. 
    time: array of time data with 2 dimensions; TxW, T=time index, W=window of data. 
    i : time index for data1, data2, and time; it should be integer. 
    bins_size: bins size of the distribution estimation; it should be integer, 2-tuple, 2-list. Default is 50.
    
return: 
    float: information rate square
    float: time

"""

import numpy as np

def adj2d_collect_inforate_square(data1, data2, time, i, bins_size=50):
    data1 = np.array(data1)
    data2 = np.array(data2)
    time = np.array(time)
    
    assert isinstance(data1, (np.ndarray)), 'data1: given data should be in numpy.array.'
    assert len(data1.shape) == 3, 'data1: given data should be in 3 dimensional array of NxTxW; N=signals #, T=time index, W=window of data.'
    assert isinstance(data2, (np.ndarray)), 'data2: given data should be in numpy.array.'
    assert len(data2.shape) == 3, 'data2: given data should be in 3 dimensional array of NxTxW; N=signals #, T=time index, W=window of data.'
    assert isinstance(time, (np.ndarray)), 'time: given time data should be in numpy.array.'
    assert len(time.shape) == 2, 'time: given data should be in 2 dimension array of TxW; T=time index, W=window of data.'
    assert data1.shape[1] == time.shape[0], 'BOTH data and time should have same length of time.'
    assert data2.shape[1] == time.shape[0], 'BOTH data and time should have same length of time.'
    assert (bins_size=='rice') or (bins_size=='sturges') or (type(bins_size)==int) or (type(bins_size)==tuple) or (type(bins_size)==list), 'bins_size: bin size for the histogram. it can estimated by rice, sturges, or specify to certain number.'
    assert isinstance(i, (int)), 'i: index for the data and time; it should be integer.'

    #copy the array to have the same dimension for the both array data.     
    if data1.shape != data2.shape:
        temp_data1 = np.tile(data1, (data2.shape[0], 1, 1)) 
        temp_data2 = np.tile(data2, (data1.shape[0], 1, 1))
    else: 
        temp_data1 = data1 
        temp_data2 = data2

    temp_time_interval = time

    if (type(bins_size)==int) or (type(bins_size)==tuple) or (type(bins_size)==list):
        temp_bins = bins_size
    elif bins_size=='rice':
        temp_bins = int(np.ceil(2 * ((data1.shape[0] * data1.shape[-1]) + (data2.shape[0] * data2.shape[-1]))**(1/3))**(0.5))
    elif bins_size=='sturges':
        temp_bins = int(np.ceil(1 + np.log2((data1.shape[0] * data1.shape[-1]) + (data2.shape[0] * data2.shape[-1])))**(0.5))
    

    # getting the range between two consecutive distribution
    temp_range1 = (min(np.min(temp_data1[:, i, :]), np.min(temp_data1[:, i+1, :])), max(np.max(temp_data1[:, i, :]), np.max(temp_data1[:, i+1, :])))
    temp_range2 = (min(np.min(temp_data2[:, i, :]), np.min(temp_data2[:, i+1, :])), max(np.max(temp_data2[:, i, :]), np.max(temp_data2[:, i+1, :])))

    # estimating the distribution BEFORE
    temp_data_interval1_before = temp_data1[:, i, :].flatten()
    temp_data_interval2_before = temp_data2[:, i, :].flatten()
    temp_pdf1, temp_edges11, temp_edges12 = np.histogram2d(temp_data_interval1_before, temp_data_interval2_before, range=((np.min(temp_range1), np.max(temp_range1)), (np.min(temp_range2), np.max(temp_range2))), bins=temp_bins, density=True)
    temp_dist_range11 = 0.5*(temp_edges11[1:] + temp_edges11[:-1]) 
    temp_dist_range12 = 0.5*(temp_edges12[1:] + temp_edges12[:-1])
    #temp_pmf1 = temp_pdf1 * np.diff(temp_dist_range1)[0] # mass function

    # estimating the distribution AFTER
    temp_data_interval1_after = temp_data1[:, i+1, :].flatten()
    temp_data_interval2_after = temp_data2[:, i+1, :].flatten()
    temp_pdf2, temp_edges21, temp_edges22 = np.histogram2d(temp_data_interval1_after, temp_data_interval2_after, range=((np.min(temp_range1), np.max(temp_range1)), (np.min(temp_range2), np.max(temp_range2))), bins=temp_bins, density=True)
    temp_dist_range21 = 0.5*(temp_edges21[1:] + temp_edges21[:-1])
    temp_dist_range22 = 0.5*(temp_edges22[1:] + temp_edges22[:-1])
    #temp_pmf2 = temp_pdf2 * np.diff(temp_dist_range2)[0] # mass function

    # checking the range between two consecutive distribution; both should be the same. 
    if not np.array_equal(temp_dist_range11, temp_dist_range21): 
        print('consecutive distribution range is incorrect for axis1!')
        return None

    if not np.array_equal(temp_dist_range12, temp_dist_range22): 
        print('consecutive distribution range is incorrect for axis2!')
        return None
    
    # information rate square calculation
    temp_dt = temp_time_interval[i+1, 0] - temp_time_interval[i, 0]
    temp_dx1 = np.diff(temp_dist_range11)[0]
    temp_dx2 = np.diff(temp_dist_range12)[0]
    temp_diff_pdf = (temp_pdf2**0.5) - (temp_pdf1**0.5)

    temp_inforate_square = ((temp_diff_pdf)/(temp_dt))**(2) * (temp_dx1*temp_dx2)
    temp_inforate_square = 4*np.sum(temp_inforate_square)

    return temp_inforate_square, temp_time_interval[i, 0]
