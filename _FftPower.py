#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:54:33 2024

@author: hengjie

calculation of the power of the interested frequency range. 

parameters: 
    fft_amp: magnitude of the frequency spectrum. 2D is expected; Nxf, N=number of channels, f=frequency index.
    fft_freq: frequency of the frequency spectrum, 1D is expected; f=frequency index.
    freq_range: upper bound and lower bound of the frequency range; 2D list, tuple, or numpy.array is expected. 
    
return: 
    temp_power: numpy.array; power for each N channels. 
"""
import numpy as np


def fft_power(fft_amp, fft_freq, freq_range):
    fft_amp = np.array(fft_amp) #make the fft_amp to numpy.array
    fft_freq = np.array(fft_freq) #make the fft_freq to numpy.array
    
    assert len(fft_amp.shape) == 2, 'fft_amp: magnitude of the fft spectrum needs to be 2D array; Nxf, N=number of channels, f=frequency index. '
    assert len(fft_freq.shape) == 1, 'fft_freq: frequency fo the fft spectrum ONLY has 1D array.'
    assert fft_amp.shape[1] == fft_freq.shape[0], 'BOTH fft_amp and fft_freq need to be in the same length.'
    assert isinstance(freq_range, (tuple, list, np.ndarray)), 'freq_range: interested range for the power calculation. It needs to be tuple, list, or numpy.array.'
    assert len(freq_range) == 2, 'freq_range: ONLY two values for the range; lower boundary and upper boundary of the frequency range.'
    assert freq_range[1] > freq_range[0], 'freq_range: the first value must smaller than the second value.'
    
    temp_amp = fft_amp
    temp_freq = fft_freq
    
    int_freq_low, int_freq_high = freq_range[0], freq_range[1] #interested frequencies
    temp_loc = np.where((temp_freq >= int_freq_low) & (temp_freq <= int_freq_high)) #index of the interested frequencies
    temp_int_range_freq = temp_freq[temp_loc[0][0] : temp_loc[0][-1]]  
    temp_int_range_amp = temp_amp[:, temp_loc[0][0] : temp_loc[0][-1]]
    temp_power = temp_int_range_amp**2 #SQUARE of the MAGNITUDE
    temp_power = np.sum(temp_power, axis=1) * np.diff(temp_int_range_freq)[0] #calculate the power by integrate the SQUARE of MAGNITUDE within interested range
    
    return temp_power