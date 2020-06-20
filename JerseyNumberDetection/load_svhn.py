# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:47:08 2020

@author: user
"""

from scipy.io import loadmat

path = r"D:\Projects\RetrieveByJerseyNumber\train\train\digitStruct.mat"
import h5py
import numpy as np
arrays={}
with h5py.File(path, 'r') as f:
    for k in f.items():
        
#        arrays[k] = np.array(v)
        for v in k:
            print(v)
