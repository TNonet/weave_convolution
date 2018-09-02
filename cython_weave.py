from cython_weave_pyx import *
import numpy as np

def array_weave_fast_forward(X, weave_param):
    cache = (X, weave_param)
    if len(X.shape) <= 2:
    	out = cython_2d_array_weave_forward(X, weave_param['num_zeros'], weave_param['filter_size'])
    elif len(X.shape) == 3:
        out = cython_3d_array_weave_forward(X, weave_param['num_zeros'], weave_param['filter_size'])
    else:
    	out = cython_4d_array_weave_forward(X, weave_param['num_zeros'], weave_param['filter_size'])
    return out, cache

def zero_weave_fast_forward(X,weave_param):
    cache = (X, weave_param)
    if len(X.shape) <= 2:
        out = cython_2d_zero_weave_forward(X, weave_param['num_zeros'], weave_param['filter_size'])
    if len(X.shape) == 3:
        out = cython_3d_zero_weave_forward(X, weave_param['num_zeros'], weave_param['filter_size'])
    return out, cache

def array_weave_fast_backward(dx, cache):
    X, weave_param = cache
    num_img, num_filters, height, width = X.shape
    num_zeros = weave_param['num_zeros']
    filter_size = weave_param['filter_size']
    dout = cython_array_weave_backward(dx, num_img, num_filters, 
    	height, width, num_zeros, filter_size)
    return dout