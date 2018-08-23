import numpy as np
cimport numpy as np
cimport cython

# DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t


@cython.boundscheck(False)
cdef cython_array_weave_forwards(np.ndarray[DTYPE_t, ndim=4] arr,int num_zeros, int filter_size):
	"""
	Cython funciton for preforming array_weave forwards that is much faster!
	"""
	cdef int num_img, num_filtes, height, width = array.shape 
	cdef expand_dist = 2*filter_size + 2
	cdef slice_jump = num_zeros + 1

	cdef i, j, int big_i, big_j, i_change, j_change

	cdef np.array([DTYPE_t, ndim=4]) temp_val

    cdef np.array[DTYPE_t, ndim=4] out = np.zeros([num_img,
                      							num_filters,
							                    height*(num_zeros + 1) - num_zeros,
							                    width*(num_zeros + 1) - num_zeros])

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
                     
    return out


