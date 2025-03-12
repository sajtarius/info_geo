#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:16:41 2024

@author: hengjie

Calculation for the [leading eigenvector series] for the square matrix time series. 
Note, the leading eigenvectors here has the highest (in term of magnitude) eigenvalues. 

parameters: 
    matrix_data: given matrix needs to be 3 dimensions array in [N x N x T]; N=dimension, T=time index.
    
return: 
    numpy.ndarray: [leading eigenvector] series with 3 dimensions [T x N]; T=time index, N=dimension. 
    
"""

import numpy as np

from tqdm import tqdm
from scipy.linalg import eigh
from joblib import Parallel, delayed

def job_lead_eigvec_cal_optimized(matrix_slice):
    """
    Optimized calculation of the leading eigenvector for a single time slice.
    """
    # Compute eigenvalues and eigenvectors
    temp_eigval, temp_eigvec = eigh(matrix_slice)

    # Index of the highest eigenvalue by magnitude
    loc_high_temp_eigval = np.argmax(np.abs(temp_eigval))

    # Extract the leading eigenvector and ensure it's real-valued
    high_temp_eigvec = np.real(temp_eigvec[:, loc_high_temp_eigval])

    # Symmetrize the eigenvector sign to ensure consistency
    if np.sum(high_temp_eigvec[high_temp_eigvec > 0]) < np.abs(np.sum(high_temp_eigvec[high_temp_eigvec < 0])):
        high_temp_eigvec = -high_temp_eigvec

    return high_temp_eigvec

def lead_eigvec_cal_optimized(matrix_data):
    """
    Optimized function to calculate the leading eigenvector time series.
    """
    matrix_data = np.asarray(matrix_data)
    assert len(matrix_data.shape) == 3, 'matrix_data must be a 3D array [N x N x T].'
    assert matrix_data.shape[0] == matrix_data.shape[1], 'matrix_data must contain square matrices.'

    # Use Parallel and delayed for efficient computation
    high_eigvec = Parallel(n_jobs=-1)(
        delayed(job_lead_eigvec_cal_optimized)(matrix_data[:, :, i]) 
        for i in tqdm(range(matrix_data.shape[2]), position=0)
    )

    # Convert the result list into a NumPy array
    return np.array(high_eigvec)

# Example usage:
# matrix_data = np.random.rand(100, 100, 200)  # Example input
# result = lead_eigvec_cal_optimized(matrix_data)
