import numpy as np

def zero_weave_forward(arr, weave_param):
    """
    input array size = (num_img, num_filters, height, width)
    weave_param = {'num_zeros': num_zeros}
    output array size = (num_img, num_filters, 
                        height*(num_zeros + 1) - num_zeros, 
                        width*(num_zeros + 1) - num_zeros)
    Ex: 
    a = [[ 1  2  3  4  5]
        [ 6  7  8  9 10]
        [11 12 13 14 15]
        [16 17 18 19 20]
        [21 22 23 24 25]]
    >>> b = zero_weave(a, {'num_zeros':2})
    b = [[ 1.  0.  0.  6.  0.  0. 11.  0.  0. 16.  0.  0. 21.]
        [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        [ 2.  0.  0.  7.  0.  0. 12.  0.  0. 17.  0.  0. 22.]
        [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        [ 3.  0.  0.  8.  0.  0. 13.  0.  0. 18.  0.  0. 23.]
        [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        [ 4.  0.  0.  9.  0.  0. 14.  0.  0. 19.  0.  0. 24.]
        [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        [ 5.  0.  0. 10.  0.  0. 15.  0.  0. 20.  0.  0. 25.]]
    
    """
    num_zeros = weave_param['num_zeros']
    cache = (arr, weave_param)
    
    num_img, num_filters, height, width = arr.shape
    
    out = np.zeros([num_img,
    	num_filters,
    	height*(num_zeros + 1) - num_zeros,
    	width*(num_zeros + 1) - num_zeros])

    slice_jump = num_zeros + 1
    out[:,:,::slice_jump,::slice_jump] = arr
    return out, cache

def zero_weave_backwards(dx, cache):
    (_, weave_param) = cache
    num_zeros = weave_param['num_zeros']
    slice_jump = num_zeros + 1
    dout = dx[:,:,::slice_jump,::slice_jump]
    return dout

#def generate_weave_multiplier(arr, weave_param):
    
def array_weave_forwards(arr, weave_param):
    """
    input array size = (num_img, num_filters, height, width)
    output array size = (num_img, num_filters, 
                        height*(num_zeros + 1) - num_zeros, 
                        width*(num_zeros + 1) - num_zeros)
    Ex: 
    a = [[ 1  2  3  4  5]
        [ 6  7  8  9 10]
        [11 12 13 14 15]
        [16 17 18 19 20]
        [21 22 23 24 25]]
    >>> b = zero_weave(a, {'num_zeros':2})
    b = [[ 1.  4.  0.  2.  5.  0.  3.  0.  1.  4.  0.  2.  5.]
        [16. 19.  0. 17. 20.  0. 18.  0. 16. 19.  0. 17. 20.]
        [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        [ 6.  9.  0.  7. 10.  0.  8.  0.  6.  9.  0.  7. 10.]
        [21. 24.  0. 22. 25.  0. 23.  0. 21. 24.  0. 22. 25.]
        [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        [11. 14.  0. 12. 15.  0. 13.  0. 11. 14.  0. 12. 15.]
        [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        [ 1.  4.  0.  2.  5.  0.  3.  0.  1.  4.  0.  2.  5.]
        [16. 19.  0. 17. 20.  0. 18.  0. 16. 19.  0. 17. 20.]
        [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        [ 6.  9.  0.  7. 10.  0.  8.  0.  6.  9.  0.  7. 10.]
        [21. 24.  0. 22. 25.  0. 23.  0. 21. 24.  0. 22. 25.]]

    
    """
    num_zeros = weave_param['num_zeros']
    filter_size = weave_param['filter_size']
    cache = (arr, weave_param)
    
    num_img, num_filters, height, width = arr.shape
    
    out = np.zeros([num_img,
                      num_filters,
                      height*(num_zeros + 1) - num_zeros,
                      width*(num_zeros + 1) - num_zeros])
    #This needs to be generalized to other filter sizes
    expand_dist = 2*filter_size+2
    slice_jump = num_zeros + 1
    
    for i in range(height):
        for j in range(width):
            temp_val = arr[:,:,i,j]
            big_i = filter_size * i 
            big_j = filter_size * j
            for i_change in [-expand_dist, 0, expand_dist]:
                for j_change in [-expand_dist, 0, expand_dist]:
                    if (min(big_i+i_change,big_j+j_change) >= 0 
                        and max(big_i+i_change,big_j+j_change) < height*(num_zeros + 1) - num_zeros):
                        out[:,:,big_i+i_change,big_j+j_change] = temp_val

    out[:,:,::slice_jump,::slice_jump] = 0
                     
    return out, cache

def array_weave_backwards(dx, cache):
    org_arr, weave_param = cache
    num_zeros = weave_param['num_zeros']
    filter_size = weave_param['filter_size']
    
    expand_dist = 2*filter_size+2
    slice_jump = num_zeros + 1
    
    num_img, num_filters, height, width = org_arr.shape
    
    dout = np.zeros([num_img,
                   num_filters,
                   height,
                   width]) 
    for i in range(height):
        for j in range(width):
            big_i = filter_size * i 
            big_j = filter_size * j
            for i_change in [-expand_dist, 0, expand_dist]:
                for j_change in [-expand_dist, 0, expand_dist]:
                    if (min(big_i+i_change,big_j+j_change) >= 0 
                        and max(big_i+i_change,big_j+j_change) < height*(num_zeros + 1) - num_zeros):
                    	if i_change != 0 and j_change != 0:
                        	dout[:,:,i,j] += dx[:,:,big_i+i_change,big_j+j_change]
    return dout

def array_sum_fowards(arr1, arr2):
    cache = (arr1, arr2)
    sum_array = arr1 + arr2
    return sum_array, cache

def array_sum_backwards(dx, cache):
    return dx