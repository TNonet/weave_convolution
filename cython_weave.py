from cython_weave_pyx import *
import numpy as np

def array_weave_fast_forward(X, weave_param):
    cache = (X, weave_param)
    out =  cython_array_weave_forward(X, weave_param['num_zeros'],
    	weave_param['filter_size'])
    return out, cache

def array_weave_fast_backward(dx, cache):
    X, weave_param = cache
    num_img, num_filters, height, width = X.shape
    num_zeros = weave_param['num_zeros']
    filter_size = weave_param['filter_size']
    dout = cython_array_weave_backward(dx, num_img, num_filters, 
    	height, width, num_zeros, filter_size)
    return dout