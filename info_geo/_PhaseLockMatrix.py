#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:59:33 2024

@author: hengjie
the [phase lock matrix] between the signals. The input signals are evaluated 
by cosine the different of the hilbert transformed phase angle between 
the signals.

Return the [phase lock matrix] of the signals in 3 dimensions. 

parameters:
    data: given the data of the signals. It should be in numpy.ndarray. 
    
return:
    numpy.ndarray: [phase lock matrix] of the given group of signals in 3 dimensions [N x N x T]. 

"""

import numpy as np

from scipy.signal import hilbert
from tqdm import tqdm

def phase_lock_matrix(data):
    data = np.array(data)
    assert len(data.shape) == 2, 'data: data should be in numpy.ndarray with 2 dimensions [N x T]; N=# channels, T=time index.'
    assert data.shape[0] >= 2, 'data: data should have 2 or more channels [N >= 2]; N=# channels.'
    assert data.shape[1] >= 1, 'data: data should have 1 or more time data [T >= 1]; T=time index.'

    hilbert_angle = data
    hilbert_angle = hilbert(hilbert_angle)
    hilbert_angle = np.unwrap(np.angle(hilbert_angle, deg=False)) #getting the angle in radian; unwarp is used to change the absolute jumps that is greater than a specified period to their 2pi complement.

    matrix_hilbert_angle = np.zeros((hilbert_angle.shape[0], hilbert_angle.shape[0], hilbert_angle.shape[1]))
    for i in tqdm(range(hilbert_angle.shape[0]), position=0, leave=False):
        for j in range(i, hilbert_angle.shape[0]):
            if i == j: 
                matrix_hilbert_angle[i,j,:] = 1
            else:
                temp_dif_angle = hilbert_angle[i] - hilbert_angle[j]
                temp_dif_angle = np.cos(temp_dif_angle)
                matrix_hilbert_angle[i,j,:] = temp_dif_angle
                matrix_hilbert_angle[j,i,:] = temp_dif_angle

    #return the values of the [phase lock matrix] 
    return matrix_hilbert_angle
