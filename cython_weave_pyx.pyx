import numpy as np
cimport numpy as np
cimport cython

# DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t
    
@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def cython_array_weave_backward(np.ndarray[DTYPE_t, ndim=4] dx,
                                  int num_img, int num_filters,
                                  int height, int width,
                                  int num_zeros, int filter_size):
    
    cdef int expand_dist = 2*filter_size + 2
    cdef int slice_jump = num_zeros + 1
    cdef int HH = dx.shape[2]

    cdef int img, layer, i, j, big_i, big_j
    cdef int i_change, j_change
    
    cdef np.float64_t temp_val
    cdef int temp_x
    cdef int temp_y

    cdef np.ndarray[DTYPE_t, ndim=4] dout = np.zeros([num_img,
                                                num_filters,
                                                height,
                                                width])
    for img in range(num_img):
        for layer in range(num_filters):
            for i in range(height):
                for j in range(width):
                    big_i = filter_size * i 
                    big_j = filter_size * j
                    i_change, j_change = -expand_dist, -expand_dist
                    while i_change < expand_dist:
                        while j_change < expand_dist:
                            if (min(big_i+i_change,big_j+j_change) >= 0 
                                and max(big_i+i_change,big_j+j_change) < HH):
                                    if i_change != 0 and j_change != 0:
                                        dout[img,layer,i,j] += dx[img,layer,big_i+i_change,big_j+j_change]
                            j_change += expand_dist
                        i_change += expand_dist
    return dout

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def cython_4d_array_weave_forward(np.ndarray[DTYPE_t, ndim=4] arr,
                                int num_zeros, int filter_size):
    """
    Cython funciton for preforming array_weave forwards that is much faster!
    """
    cdef int num_img = arr.shape[0]
    cdef int num_filters = arr.shape[1]
    cdef int height = arr.shape[2]
    cdef int width = arr.shape[3] 
    
    cdef int expand_dist = 2*filter_size + 2
    cdef int slice_jump = num_zeros + 1

    cdef int img, layer, i, j, big_i, big_j, i_change, j_change
    
    cdef np.float64_t temp_val
    
    cdef int HH = height*(num_zeros + 1) - num_zeros
    cdef int WW = width*(num_zeros + 1) - num_zeros

    cdef np.ndarray[DTYPE_t, ndim=4] out = np.zeros([num_img,
                                                num_filters,
                                                HH,
                                                WW])

    cdef int temp_x
    cdef int temp_y

    for img in range(num_img):
        for layer in range(num_filters):
            for i in range(height):
                for j in range(width):
                    temp_val = arr[img,layer,i,j]
                    big_i = filter_size * i 
                    big_j = filter_size * j
                    #######
#                     i_change, j_change = -expand_dist, -expand_dist
#                     while i_change < expand_dist:
#                         while j_change < expand_dist:
#                             if (min(big_i+i_change,big_j+j_change) >= 0 
#                                 and max(big_i+i_change,big_j+j_change) < HH):
#                                     if i_change != 0 and j_change != 0:
#                                         out[img,layer,big_i+i_change,big_j+j_change] = temp_val
#                             j_change += expand_dist
#                         i_change += expand_dist
                    #First Column
                    temp_x = big_i-expand_dist
                    temp_y = big_j-expand_dist
                    if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                        out[img,layer,temp_x,temp_y] = temp_val
                    temp_x = big_i
                    if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                        out[img,layer,temp_x,temp_y] = temp_val
                    temp_x = big_i + expand_dist 
                    if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                        out[img,layer,temp_x,temp_y] = temp_val
                    #Second Column (Only 2 Points)
                    temp_x = big_i-expand_dist
                    temp_y = big_j
                    if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                        out[img,layer,temp_x,temp_y] = temp_val
                    temp_x = big_i + expand_dist 
                    if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                        out[img,layer,temp_x,temp_y] = temp_val
                    #Third Column
                    temp_x = big_i-expand_dist
                    temp_y = big_j+expand_dist
                    if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                        out[img,layer,temp_x,temp_y] = temp_val
                    temp_x = big_i
                    if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                        out[img,layer,temp_x,temp_y] = temp_val
                    temp_x = big_i + expand_dist 
                    if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                        out[img,layer,temp_x,temp_y] = temp_val

    return out


@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def cython_3d_array_weave_forward(np.ndarray[DTYPE_t, ndim=3] arr,
                                int num_zeros, int filter_size):
    """
    Cython funciton for preforming array_weave forwards that is much faster!
    """
    cdef int num_filters = arr.shape[0]
    cdef int height = arr.shape[1]
    cdef int width = arr.shape[2] 
    
    cdef int expand_dist = 2*filter_size + 2
    cdef int slice_jump = num_zeros + 1

    cdef int layer, i, j, big_i, big_j, i_change, j_change
    
    cdef np.float64_t temp_val
    
    cdef int HH = height*(num_zeros + 1) - num_zeros
    cdef int WW = width*(num_zeros + 1) - num_zeros

    cdef np.ndarray[DTYPE_t, ndim=3] out = np.zeros([num_filters,
                                                HH,
                                                WW])

    cdef int temp_x
    cdef int temp_y

    for layer in range(num_filters):
        for i in range(height):
            for j in range(width):
                temp_val = arr[layer,i,j]
                big_i = filter_size * i 
                big_j = filter_size * j
                #First Column
                temp_x = big_i-expand_dist
                temp_y = big_j-expand_dist
                if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                    out[layer,temp_x,temp_y] = temp_val
                temp_x = big_i
                if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                    out[layer,temp_x,temp_y] = temp_val
                temp_x = big_i + expand_dist 
                if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                    out[layer,temp_x,temp_y] = temp_val
                #Second Column (Only 2 Points)
                temp_x = big_i-expand_dist
                temp_y = big_j
                if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                    out[layer,temp_x,temp_y] = temp_val
                temp_x = big_i + expand_dist 
                if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                    out[layer,temp_x,temp_y] = temp_val
                #Third Column
                temp_x = big_i-expand_dist
                temp_y = big_j+expand_dist
                if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                    out[layer,temp_x,temp_y] = temp_val
                temp_x = big_i
                if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                    out[layer,temp_x,temp_y] = temp_val
                temp_x = big_i + expand_dist 
                if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                    out[layer,temp_x,temp_y] = temp_val

    return out

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def cython_2d_array_weave_forward(np.ndarray[DTYPE_t, ndim=2] arr,
                                int num_zeros, int filter_size):
    """
    Cython funciton for preforming array_weave forwards that is much faster!
    """
    cdef int height = arr.shape[0]
    cdef int width = arr.shape[1] 
    
    cdef int expand_dist = 2*filter_size + 2
    cdef int slice_jump = num_zeros + 1

    cdef int i, j, big_i, big_j, i_change, j_change
    
    cdef np.float64_t temp_val
    
    cdef int HH = height*(num_zeros + 1) - num_zeros
    cdef int WW = width*(num_zeros + 1) - num_zeros

    cdef np.ndarray[DTYPE_t, ndim=2] out = np.zeros([HH, WW])

    cdef int temp_x
    cdef int temp_y

    for i in range(height):
        for j in range(width):
            temp_val = arr[i,j]
            big_i = filter_size * i 
            big_j = filter_size * j
            #######
            #First Column
            temp_x = big_i-expand_dist
            temp_y = big_j-expand_dist
            if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                out[temp_x,temp_y] = temp_val
            temp_x = big_i
            if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                out[temp_x,temp_y] = temp_val
            temp_x = big_i + expand_dist 
            if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                out[temp_x,temp_y] = temp_val
            #Second Column (Only 2 Points)
            temp_x = big_i-expand_dist
            temp_y = big_j
            if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                out[temp_x,temp_y] = temp_val
            temp_x = big_i + expand_dist 
            if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                out[temp_x,temp_y] = temp_val
            #Third Column
            temp_x = big_i-expand_dist
            temp_y = big_j+expand_dist
            if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                out[temp_x,temp_y] = temp_val
            temp_x = big_i
            if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                out[temp_x,temp_y] = temp_val
            temp_x = big_i + expand_dist 
            if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                out[temp_x,temp_y] = temp_val
    return out


@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def cython_3d_zero_weave_forward(np.ndarray[DTYPE_t, ndim=3] arr,
                                int num_zeros, int filter_size):
    """
    Cython funciton for preforming array_weave forwards that is much faster!
    """
    cdef int height = arr.shape[1]
    cdef int width = arr.shape[2] 
    cdef int num_layers = arr.shape[0]
    
    cdef int slice_jump = num_zeros + 1

    cdef int i, j, big_i, big_j, layer
    
    cdef int HH = height*(num_zeros + 1) - num_zeros
    cdef int WW = width*(num_zeros + 1) - num_zeros

    cdef np.ndarray[DTYPE_t, ndim=3] out = np.zeros([num_layers,HH, WW])

    for layer in range(num_layers):
        for i in range(height):
            for j in range(width):
                big_i = slice_jump * i 
                big_j = slice_jump * j
                out[layer,big_i, big_j] = arr[layer,i,j]
    return out


@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def cython_2d_zero_weave_forward(np.ndarray[DTYPE_t, ndim=2] arr,
                                int num_zeros, int filter_size):
    """
    Cython funciton for preforming array_weave forwards that is much faster!
    """
    cdef int height = arr.shape[0]
    cdef int width = arr.shape[1] 
    
    cdef int slice_jump = num_zeros + 1

    cdef int i, j, big_i, big_j
    
    cdef int HH = height*(num_zeros + 1) - num_zeros
    cdef int WW = width*(num_zeros + 1) - num_zeros

    cdef np.ndarray[DTYPE_t, ndim=2] out = np.zeros([HH, WW])


    for i in range(height):
        for j in range(width):
            big_i = slice_jump * i 
            big_j = slice_jump * j
            out[big_i, big_j] = arr[i,j]
    return out
