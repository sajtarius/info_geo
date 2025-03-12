#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:17:07 2024

@author: hengjie

Calculation for the [leading eigenvector series] for the square matrix time series. 
Note, the leading eigenvectors here has the highest (in term of magnitude) eigenvalues. 

parameters: 
    matrix_data: given matrix needs to be 3 dimensions array in [N x N x T]; N=dimension, T=time index.
    
return: 
    numpy.ndarray: [leading eigenvector] series with 3 dimensions [T x N]; T=time index, N=dimension. 

"""

#compute the [leading eigenvector] using scipy
#the [leading eigenvector] is computed at each time step
import numpy as np 

from scipy.linalg import eigh
from tqdm import tqdm
from joblib import Parallel, delayed

def job_lead_eigvec_cal(matrix, i):
    matrix = np.array(matrix) 
    assert len(matrix.shape) == 3, 'matrix: given matrix needs to be 3 dimensions array in [N x N x T]; N=dimension, T=time index.'
    assert isinstance(i, (int)), 'i: time index of the matrix. It should be a integer.'
    
    temp_matrix = matrix[:, :, i]
    temp_eigval, temp_eigvec = eigh(temp_matrix)

    # index of the highest eigenvalue (in term of magnitude)
    loc_high_temp_eigval = np.argmax(np.abs(temp_eigval))

    #the [reshape] is done to get the [N x 1] vector of the [leading eigenvector] else it will be [N] vector.
    #the [real] is needed to make sure to "get rid" of the [imaginary part], JUST in case. Can be removed since the eigenvector is always positive in this case.
    high_temp_eigvec = np.real(temp_eigvec[:, loc_high_temp_eigval])#.reshape(-1, 1)

    #The largest component is set to negative.
    #This step is needed because the same eigenvector can have a negative symmetric result.
    #To make sure the symmetric is always the same. Negative is chosen as the reference.
    #refer to [https://github.com/juanitacabral/LEiDA/blob/master/LEiDA/LEiDA.m]
    if len(high_temp_eigvec[high_temp_eigvec>0]) == 0:
        high_temp_eigvec = high_temp_eigvec
    elif np.mean(high_temp_eigvec[high_temp_eigvec>0]) > 0.5:
        high_temp_eigvec = -1*high_temp_eigvec
    elif np.mean(high_temp_eigvec[high_temp_eigvec > 0]==0.5) and np.sum(high_temp_eigvec[high_temp_eigvec>0]) > -1*np.sum(high_temp_eigvec[high_temp_eigvec>0]):
        high_temp_eigvec = -1*high_temp_eigvec


    return high_temp_eigvec 



def lead_eigvec_cal(matrix_data):
    matrix_data = np.array(matrix_data)
    assert len(matrix_data.shape) == 3, 'matrix_data: the matrix needs to be [3 dimensions] numpy.array such that it has dimension for [N x N x T].'
    assert matrix_data.shape[0] == matrix_data.shape[1], 'matrix_data: the matrix should be a square matrix time series.'

    matrix_hilbert_angle = matrix_data

    high_eigvec = Parallel(n_jobs=-1)(delayed(job_lead_eigvec_cal)(matrix_hilbert_angle, i) for i in tqdm(range(0, matrix_hilbert_angle.shape[2], 1), position=0, leave=False))

    high_eigvec = np.array(high_eigvec) #[leading eigenvector]

    #return the [leading eigenvector] series [T x N]. 
    return high_eigvec