#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:32:21 2024

@author: hengjie

The [information rate] here is evaluated by having the sampling the probability distribution (PDF)
at the ensemble of regions instead of just one single signal. For instance, NxW of data would be 
sampled to form a PDF to evaluate the [information rate]. 

Return information rate square for 1D distribution 

parameters: 
    data: array of data in 3 dimensions; NxTxW, N=signals #, T=time index, W=window of data. 
    time: array of time data with 2 dimensions; TxW, T=time index, W=window of data. 
    i : time index for data1, data2, and time; it should be integer. 
    bins_size: bins size of the distribution estimation; it should be integer. Default is 50.
    
return: 
    float: information rate square
    float: time

"""
import numpy as np

def adj_collect_inforate_square(data, time, i, bins_size=50):
    data = np.array(data)
    time = np.array(time)
    
    assert isinstance(data, (np.ndarray)), 'data: given data should be in numpy.array.'
    assert len(data.shape) == 3, 'data: given data should be in 3 dimensional array of NxTxW; N=signals #, T=time index, W=window of data.'
    assert isinstance(time, (np.ndarray)), 'time: given time data should be in numpy.array.'
    assert len(time.shape) == 2, 'time: given data should be in 2 dimension array of TxW; T=time index, W=window of data.'
    assert data.shape[1] == time.shape[0], 'BOTH data and time should have same length of time.'
    assert (bins_size=='rice') or (bins_size=='sturges') or (type(bins_size)==int), 'bins_size: bin size for the histogram. it can estimated by rice, sturges, or specify to certain number.'
    assert isinstance(i, (int)), 'i: index for the data and time; it should be integer.'

    temp_data_interval = data
    temp_time_interval = time

    if type(bins_size)==int:
        temp_bins = bins_size
    elif bins_size=='rice':
        temp_bins = int(np.ceil(2 * (data.shape[0] * data.shape[-1])**(1/3)))
    elif bins_size=='sturges':
        temp_bins = int(np.ceil(1 + np.log2(data.shape[0] * data.shape[-1])))
    

    # getting the range between two consecutive distribution
    temp_range = (min(np.min(temp_data_interval[:, i, :]), np.min(temp_data_interval[:, i+1, :])), max(np.max(temp_data_interval[:, i, :]), np.max(temp_data_interval[:, i+1, :])))

    # estimating the distribution BEFORE
    temp_pdf1, temp_edges1 = np.histogram(temp_data_interval[:, i, :], range=(np.min(temp_range), np.max(temp_range)), bins=temp_bins, density=True)
    temp_dist_range1 = 0.5*(temp_edges1[1:] + temp_edges1[:-1]) 
    #temp_pmf1 = temp_pdf1 * np.diff(temp_dist_range1)[0] # mass function

    # estimating the distribution AFTER
    temp_pdf2, temp_edges2 = np.histogram(temp_data_interval[:, i+1, :], range=(np.min(temp_range), np.max(temp_range)), bins=temp_bins, density=True)
    temp_dist_range2 = 0.5*(temp_edges2[1:] + temp_edges2[:-1])
    #temp_pmf2 = temp_pdf2 * np.diff(temp_dist_range2)[0] # mass function

    # checking the range between two consecutive distribution; both should be the same. 
    if not np.array_equal(temp_dist_range1, temp_dist_range2): 
        print('consecutive distribution range is incorrect!')
        return None
    
    # information rate square calculation
    temp_dt = temp_time_interval[i+1, 0] - temp_time_interval[i, 0]
    temp_dx = np.diff(temp_dist_range1)[0]
    temp_diff_pdf = (temp_pdf2**0.5) - (temp_pdf1**0.5)

    temp_inforate_square = ((temp_diff_pdf)/(temp_dt))**(2) * (temp_dx)
    temp_inforate_square = 4*np.sum(temp_inforate_square)

    return temp_inforate_square, temp_time_interval[i, 0]
