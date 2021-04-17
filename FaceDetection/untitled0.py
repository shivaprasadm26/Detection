# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:34:21 2020

@author: user
"""

from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
with CustomObjectScope({'tf': tf}):
  model = load_model('./model/nn4.small2.v1.h5')