#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 19:59:27 2024

@author: hengjie

The [information rate] here is evaluated by sampling the probability dsitribution (PDF) 
through the frequency spectrum of short time Fourier transform (stft). The range in this 
case is fixed based on the interested frequency range. 

return information rate square for 1D distribution based on the frequency spectrum. 

parameters: 
    int_signal: array of data in 1 dimension; T, T=time index. 
    int_time: array of time in 1 dimension; T, T=time index.
    win_size: window size for the stft. It should be integer, default is 150.
    overlap: number of points of overlapping. It should be integer, default is None=(win_size/2).
    int_freq_range: range of interested frequency. It should be in list or tuple, default is [0, 300].
    
return: 
    numpy.ndarray: information rate square 
    numpy.ndarray: time
"""

import numpy as np 

from scipy.signal import stft


def inforate_square_stft(int_signal, int_time, win_size=150, overlap=None, int_freq_range=[0, 300]):
    assert len(int_signal.shape)==1, 'int_signal: given data should be in numpy.array format.'
    assert len(int_time.shape)==1, 'int_time: given time data should be in numpy.array format.'
    assert isinstance(win_size, int), 'win_size: window size should be in integer; default is 150.'
    assert (overlap==None) or type(overlap)==int, 'overlap: number of overlapping; default is [win_size/2].'
    assert type(int_freq_range)==list or type(int_freq_range)==tuple, 'int_freq_range: range of the interested frequency. List or tuple is expected.'
    assert len(int_freq_range)==2, 'int_freq_range: range of interested frequency should only have TWO elements. ONLY minimum and maximum of the range.'

    stft_freq, stft_time, stft_z = stft(int_signal, fs=1/np.diff(int_time)[0], nperseg=win_size, noverlap=overlap)
    int_minfreq = np.where(stft_freq <= np.min(int_freq_range))[0][-1]
    int_maxfreq = np.where(stft_freq <= np.max(int_freq_range))[0][-1]

    pmf_stft = np.abs(stft_z)[int_minfreq: int_maxfreq, :]/np.sum(np.abs(stft_z)[int_minfreq: int_maxfreq, :], axis=0) #probability mass function; index is [frequency, time]
    pdf_stft = pmf_stft/np.diff(stft_freq)[0] #probability density function; index is [frequency, time]
    range_stft = stft_freq #range of the distribution. It is the same as the frequency range.

    # information rate calculation
    diff_pdf = np.diff(np.sqrt(pdf_stft), axis=-1)
    diff_time = np.diff(stft_time)[0]
    diff_range = np.diff(range_stft)[0]

    inforate_stft = 4 * np.sum(diff_pdf, axis=0)**(2) * (diff_range/(diff_time**2))

    return inforate_stft, stft_time[:-1]