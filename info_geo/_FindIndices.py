#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:20:23 2024

@author: hengjie
"""

import numpy as np 

def find_indices(original, subset):
    original_array = np.array(original)
    subset_array = np.array(subset)
    
    # Create a sorted version of the original array and get the sorting indices
    sorted_indices = np.argsort(original_array)
    sorted_original = original_array[sorted_indices]
    
    # Use searchsorted to find indices in the sorted array
    subset_indices_sorted = np.searchsorted(sorted_original, subset_array)
    
    # Map these indices back to the original array's indices
    subset_indices = sorted_indices[subset_indices_sorted]
    
    return list(map(str, subset_indices))
