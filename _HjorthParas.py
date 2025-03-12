#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:10:50 2024

@author: hengjie

Hjorth parameters calculation for window slided signals. 

Return Hjorth parameters for the window slided signals. 

parameters:
    data: data of the signal 
    time: data of the time 
    axis_data: axis at which the data is evaluated. Mainly for the numpy.diff(); default is -1.
    axis_time: axis at which the time is evaluated. Mainly for the numpy.diff(); default is 0.

return: 
    numpy.ndarray: complexity 
    numpy.ndarray: mobility 
    numpy.ndarray: activity 
    numpy.ndarray: time

"""

import numpy as np

def hjorth_act(data, time, axis_data=-1, axis_time=0):
    assert isinstance(data, np.ndarray), 'data: given data should be in numpy.array'
    assert len(data.shape) == 3, 'data: given data should be in 3 dimensions; CxTxW, C=# channels, T=time index, W=window of data.'
    assert isinstance(time, np.ndarray), 'time: given time data should be in numpy.array.'
    assert len(time.shape) == 2, 'time: given time data should be in 2 dimensions; TxW, T=time index, W=window of data.'
    #assert data.shape[1] == time.shape[0], 'BOTH data and time should have the same sliding window.'
    #assert data.shape[2] == time.shape[1], 'BOTH data and time should have the same sliding window.'
    assert isinstance(axis_data, int), 'axis_data: axis that evaluate the parameter for DATA. Integer is needed; default is -1.'
    assert isinstance(axis_time, int), 'axis_time: axis that evaluate the parameter for TIME. Integer is needed; default is 0.'

    temp_activity = np.var(data, axis=axis_data) 

    return temp_activity, time[:, 0]

def hjorth_mob(data, time, axis_data=-1, axis_time=0): 
    assert isinstance(data, np.ndarray), 'data: given data should be in numpy.array.'
    assert len(data.shape) == 3, 'data: given data should be in 3 dimensions; CxTxW, C=# channels, T=time index, W=window of data.'
    assert isinstance(time, np.ndarray), 'time: given time data should be in numpy.array.'
    assert len(time.shape) == 2, 'time: given time data should be in 2 dimensions; TxW, T=time index, W=window of data.'
    #assert data.shape[1] == time.shape[0], 'BOTH data and time should have the same sliding window.'
    #assert data.shape[2] == time.shape[1], 'BOTH data and time should have the same sliding window.'
    assert isinstance(axis_data, int), 'axis_data: axis that evaluate the parameter for DATA. Integer is needed; default is -1.'
    assert isinstance(axis_time, int), 'axis_time: axis that evaluate the parameter for TIME. Integer is needed; default is 0.'

    temp_nume, _ = hjorth_act(data=(np.diff(data, axis=axis_data))/(np.diff(time, axis=axis_time)[0][0]), time=time, axis_data=axis_data, axis_time=axis_time)
    temp_deno, _ = hjorth_act(data, time, axis_data=axis_data, axis_time=axis_time)

    temp_mobility = np.sqrt((temp_nume)/(temp_deno))

    return temp_mobility, time[:, 0]

def hjorth_com(data, time, axis_data=-1, axis_time=0): 
    assert isinstance(data, np.ndarray), 'data: given data should be in numpy.array.'
    assert len(data.shape) == 3, 'data: given data should be in 3 dimensions; CxTxW, C=# channels, T=time index, W=window of data.'
    assert isinstance(time, np.ndarray), 'time: given time data should be in numpy.array.'
    assert len(time.shape) == 2, 'time: given time data should be in 2 dimensions; TxW, T=time index, W=window of data.'
    assert data.shape[1] == time.shape[0], 'BOTH data and time should have the same sliding window.'
    assert data.shape[2] == time.shape[1], 'BOTH data and time should have the same sliding window.'
    assert isinstance(axis_data, int), 'axis_data: axis that evaluate the parameter for DATA. Integer is needed; default is -1.'
    assert isinstance(axis_time, int), 'axis_time: axis that evaluate the parameter for TIME. Integer is needed; default is 0.'

    temp_nume, _ = hjorth_mob(data=(np.diff(data, axis=axis_data))/(np.diff(time, axis=axis_time)[0][0]), time=time, axis_data=axis_data, axis_time=axis_time)
    temp_deno, _ = hjorth_mob(data, time, axis_data=axis_data, axis_time=axis_time)

    temp_complexity = (temp_nume)/(temp_deno)

    return temp_complexity, time[:, 0]

def hjorth_paras(data, time, axis_data=-1, axis_time=0): 
    assert isinstance(data, np.ndarray), 'data: given data should be in numpy.array.'
    assert len(data.shape) == 3, 'data: given data should be in 3 dimensions; CxTxW, C=# channels, T=time index, W=window of data.'
    assert isinstance(time, np.ndarray), 'time: given time data should be in numpy.array.'
    assert len(time.shape) == 2, 'time: given time data should be in 2 dimensions; TxW, T=time index, W=window of data.'
    assert data.shape[1] == time.shape[0], 'BOTH data and time should have the same sliding window.'
    assert data.shape[2] == time.shape[1], 'BOTH data and time should have the same sliding window.'
    assert isinstance(axis_data, int), 'axis_data: axis that evaluate the parameter for DATA. Integer is needed; default is -1.'
    assert isinstance(axis_time, int), 'axis_time: axis that evaluate the parameter for TIME. Integer is needed; default is 0.'
    
    temp_complexity, _ = hjorth_com(data, time, axis_data=axis_data, axis_time=axis_time)
    temp_mobility, _ = hjorth_mob(data, time, axis_data=axis_data, axis_time=axis_time)
    temp_activity, _ = hjorth_act(data, time, axis_data=axis_data, axis_time=axis_time)
    
    return temp_complexity, temp_mobility, temp_activity, time[:, 0]
    