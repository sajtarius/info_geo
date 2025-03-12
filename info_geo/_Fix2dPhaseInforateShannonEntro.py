#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:06:40 2024

@author: hengjie

parameters: 
    sig1: given signal 1 should be in 1D numpy.array with time index.
    sig2: given signal 2 should be in 1D numpy.array with time index.
    time: given time data should be in 1D numpy.array with time index.
    win: window size of the sliding window. Integer is expected; default is 10.
    sld: sliding of the sliding window. Integer is expected; default is 2.
    bins: bins size of the distribution for information rate calculation. Integer is expected; default is 50.
    int_range: range of the distribution for information rate calculation. Integer, float, tuple, or list is expected; default is 1.05. 
    entro_bins: bins size of the distribution for shannon entropy calculation. Integer is expected; default is 10.
    base: base for the shannon entropy calculation. Integer or float is expected; default is 2.
    norm: normalization of the shannon entropy. Boolean is expected; default is True.
    
return: 
    float: temp_infoentropy; 2D information rate Shannon entropy.
    numpy.ndarray: inforate_all_data; information rate data. 
    numpy.ndarray: inforate_all_pmf; PMF of the information rate over time.
"""

# @title 2D information rate Shannon entropy (hilbert transform; phase space)
import numpy as np
import info_geo as ig

from scipy.signal import hilbert 
from scipy.stats import entropy
from numpy.lib.stride_tricks import sliding_window_view

def fix2d_phase_inforate_shannon_entro(sig1, sig2, time, win=10, sld=2, bins=50, int_range=1.05, entro_bins=10, base=2, norm=True):
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
    assert (type(int_range)==int) or (type(int_range)==float) or (type(int_range)==tuple) or (type(int_range)==list), 'int_range: range of the distribution should be in integer, float, tuple, or list; default is 1.05.'
    assert isinstance(entro_bins, int), 'entro_bins: bins size of the distribution for shannon entropy calculation. Integer is expected; default is 10.'
    assert isinstance(base, (int, float)), 'base: base for the shannon entropy calculation. Integer or float is expected; default is 2.'
    assert isinstance(norm, bool), 'norm: normalization of the shannon entropy. Boolean is expected; default is True.'

    analy_sig1 = hilbert(sig1)
    analy_sig2 = hilbert(sig2)

    instance_phase1 = np.unwrap(np.angle(analy_sig1, deg=False))
    instance_phase2 = np.unwrap(np.angle(analy_sig2, deg=False))

    instance_phase_angle1 = np.cos(instance_phase1)
    instance_phase_angle2 = np.cos(instance_phase2)

    int_win = win
    int_sld = sld

    instance_phase_angle_slide1 = sliding_window_view(instance_phase_angle1, window_shape=int_win)
    instance_phase_angle_slide2 = sliding_window_view(instance_phase_angle2, window_shape=int_win)

    instance_phase_angle_slide1 = instance_phase_angle_slide1[::int_sld, :]
    instance_phase_angle_slide2 = instance_phase_angle_slide2[::int_sld, :]

    time_slide = sliding_window_view(time, window_shape=int_win)
    time_slide = time_slide[::int_sld, :]

    temp_infosquare, temp_infotime = ig.fix_double_inforate_square(np.expand_dims(instance_phase_angle_slide1, axis=0), np.expand_dims(instance_phase_angle_slide2, axis=0), time_slide, int_bins=bins, int_range=int_range)

    '''rice_bins_info = int(np.ceil(2 * temp_infosquare.shape[0]**(1/3)))
    print(f'rice rule: {rice_bins_info}')'''

    pdf_info, range_info = np.histogram(temp_infosquare**0.5, bins=entro_bins, density=True)
    range_info = (range_info[1:] + range_info[:-1])/2
    pmf_info = pdf_info * np.diff(range_info)[0]

    if norm==True:
        temp_infoentropy = entropy(pmf_info, base=base)/(entropy(np.ones(pmf_info.shape[0])/pmf_info.shape[0], base=base))
    elif norm==False:
        temp_infoentropy = entropy(pmf_info, base=base)
    print(f'inforate_entropy: {temp_infoentropy}')

    inforate_all_data = np.vstack((temp_infosquare, temp_infotime)).T
    inforate_all_pmf = np.vstack((pmf_info, range_info)).T

    #[information rate shannon entropy], [information rate data], [PMF for the information rate shannon entropy]
    return temp_infoentropy, inforate_all_data, inforate_all_pmf
