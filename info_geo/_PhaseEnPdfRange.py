#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:23:46 2024

@author: hengjie

construct the evolution of probability density function (PDF) based on second-order difference plot. 

parameters:
    signal: signal data should be in numpy.array. 
    time: time data should be in numpy.array. 
    int_win: sampling PDF window size. Integer is expected; default is 500. 
    int_sld: slide of the sliding window. Integer is expected; default is 250. 
    int_delay: time delay of the embedding for the second-order difference plot. Integer or boolean is expected; default is 10. True will be estimated by Takens Theorem. 
    int_sigma: standard deviation of the gaussian filter to smoothen the PDF. Integer is expected; default is 5.
    
return: 
    all_phase_pdf: numpy.ndarray 
    all_phase_range: numpy.ndarray

"""

import numpy as np
import info_geo as ig # NOTE: this is own package

from numpy.lib.stride_tricks import sliding_window_view
#from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
#from gtda.time_series import SingleTakensEmbedding


def phase_en_pdf(data, index, int_k=80, int_tau=10, int_sigma=5): 
    assert data.ndim==2, 'data: given data should be in two dimension. TxW, T=time index, W=window of sample data.'
    assert isinstance(index, (int)), 'index: time index of the data. Integer is expected.'
    
    temp_phase_pmf = ig.phase_en_pmf(data[index, :], K=int_k, tau=int_tau)
    temp_phase_pmf = gaussian_filter1d(temp_phase_pmf, sigma=int_sigma)
    temp_phase_pdf = temp_phase_pmf 
    temp_phase_range = np.linspace(1, temp_phase_pmf.shape[0], temp_phase_pmf.shape[0])

    return temp_phase_pdf, temp_phase_range

def phase_en_pdf_range(signal, time, int_win=500, int_sld=250, int_bins=80, int_delay=10, int_sigma=5):
    signal = np.squeeze(signal)
    time = np.squeeze(time)
    
    assert signal.shape[0]>10 and signal.ndim == 1, 'signal: given signal must be a numpy.array with length of 10.'
    assert time.shape[0]>10 and time.ndim == 1, 'time: given time must be a numpy.array with length of 10.'
    assert isinstance(int_win, (int)), 'int_win: slidng window, window of estimating the distribution.'
    assert isinstance(int_sld, (int)), 'int_sld: slidinng window, sliding of estimating the evolution of the distribution.'
    assert isinstance(int_bins, (int)), 'int_bins: bin size of the distribution.'
    #assert type(int_delay)==int or type(int_delay)==bool, 'int_delay: embedding of the time delay for second-order difference plot.Integer or bool is expected; default is 10. True is estimated by Takens theorem.'
    assert isinstance(int_delay, (int)), 'int_delay: embedding of the time delay for second-order difference plot. Integer is expected; default is 10.'
    assert isinstance(int_sigma, (int)), 'int_sigma, gaussian filter smoothing standard deviation. Integer is expected; default is 5.'
    
    y=signal 
    t=time
    slide_y = sliding_window_view(y, window_shape=int_win, axis=-1)
    slide_t = sliding_window_view(t, window_shape=int_win, axis=-1)
    
    slide_y = slide_y[::int_sld, :]
    slide_t = slide_t[::int_sld, :]
     
    '''if int_delay==True:
        embed = SingleTakensEmbedding(parameters_type='search', time_delay=int(slide_y[0].shape[0]/4), dimension=2)
        embed_slide_y1 = embed.fit_transform(slide_y[0])
        print(f'embed time delay based on mutual information: \t {embed.time_delay}')
        print(f'embed dimension based on false nearest neighbour: \t {embed.dimension}')
        print(f'embed signal shape: \t {embed_slide_y1.shape}')
        int_delay = embed.time_delay
    else:
        int_delay = int_delay'''
    
    
    all_phase_pdf, all_phase_range = zip(*Parallel(n_jobs=-1)(delayed(phase_en_pdf)(slide_y, i, int_k=int_bins, int_tau=int_delay, int_sigma=int_sigma) for i in (range(slide_y.shape[0]))))
    all_phase_pdf = np.array(all_phase_pdf)
    all_phase_range = np.array(all_phase_range)

    return all_phase_pdf, all_phase_range, slide_t[:-1, 0]
