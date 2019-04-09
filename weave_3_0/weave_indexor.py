import numpy as np


def array_weave_model(data_shape, filter_shape, channel_loc):
	"""
	Brute force weaving of a non-sensical example from
		>>> x.shape
			(num_exampels, width, height, channels)
		OR:
		>>> x.shape 
			(num_examples, channels, width, height)

	data_shape = x.shape
	channel_loc selectes between {'channel_first', 'channel_last'}
	filter_shape = (N,N) or N representing the dimensions of the
		filters being used

	array weave returns a 4D array, array_weave_indexor, of size
	 (num_filters, height, width, input_tensor_dimension = len(x.shape))


	"""
	if channel_loc == "channel_first":
		_,channels,width,height = data_size
	elif channel_loc == "channel_last":
		_,width,height,channels = data_size
	else:
		raise ValueError('channel_loc must be either "channel_last" or "channel_first".')

	if type(filter_shape) == list:
		assert len(filter_shape) == 2:
		assert filter_shape[0] == filter_shape[1]
		filter_shape = filter_shape[0]
		
	N_arr = brute_weave([channels,width,height], filter_size)


def brute_weave(N_arr_shape, filter_size):
	"""

	"""


	channels,width,height = N_arr_shape

	N_arr_weave = np.zeros([])
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
            #Second Column (Only 2 Points) Unless include_center == 1!
            temp_x = big_i-expand_dist
            temp_y = big_j
            if (min(temp_x,temp_y) >= 0 and max(temp_x,temp_y) < HH):
                out[temp_x,temp_y] = temp_val
            if include_center == 1:
                temp_x = big_i
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

def create_part_I_zero_weave_matrix(input_shape, weave_param):
    """
    Createes properly sized 4D indexing tensor, I_zero_weave that can be applied to
    tf.gather_nd(x, I_zero_weave) that will return a tensor that is identical to
    x = explode_tensor(tensor, num_zeros = 2)

    However, this takes a while to build in favor of fast execution time in the graph
    thus we ust set a max number of images before we build.

    Creates an matrix of size (num_images, num_filters, height, width, input_tensor_dimension (4))
    """

    num_filters,num_rows,num_cols = input_shape
    num_zeros = weave_param['num_zeros']
    tensor_dimension = len(input_shape)
    
    if tensor_dimension != 3:
        raise ValueError('Must operate on 4D tensors')
    if num_rows != num_cols:
        raise ValueErrro('Must operate ona square image')
        
    HH = WW = num_rows*(1+num_zeros) - num_zeros
     

    N_arr = np.arange(1,num_cols**2 * num_filters + 1).reshape(
        [num_filters,num_rows,num_cols]).astype(float)
    
    N_zero_weave = np.zeros([num_filters,HH,WW])
    N_zero_weave[:,::3,::3] = N_arr
    I_part = np.zeros([num_filters,HH,WW,tensor_dimension])

    N_zero_weave[np.where(N_zero_weave == 0)] = -1

    for fil in range(num_filters):
        for i in range(HH):
            for j in range(WW):
                i_j_loc = np.where(N_zero_weave[fil,i,j] == N_arr)
                if len(i_j_loc[0]) > 0:
                    I_part[fil,i,j] = i_j_loc
                else:
                    I_part[fil,i,j] = np.array([0,0,HH//weave_param['num_zeros']])
   
    return I_part.astype(int)

def create_part_I_array_weave_matrix(input_shape, weave_param, include_center = 0):
    """
    createses properly sized 4D indexing tensor, I_array_weave that can be applied to
    each 3D tensor of the inputs
    tf.gather_nd(x, I_array_weave) that will return a tensor that is identical to
    x, cache = array_weave_fast_forward(tensor, num_zeros = 2)

    However, this takes a while to build in favor of fast execution time in the graph
    thus

    Creates an matrix of size (num_images, num_filters, height, width, input_tensor_dimension (4))
    """
    num_filters,num_rows,num_cols = input_shape

    tensor_dimension = len(input_shape)
    if tensor_dimension != 3:
        raise ValueError('Must operate on 4D tensors')
    if num_rows != num_cols:
        raise ValueErrro('Must operate on a square image')
    
    N_arr = np.arange(1,num_filters*num_cols**2 + 1).reshape(
        [num_filters,num_rows,num_cols]).astype(float)
    N_weave,_ = array_weave_fast_forward(N_arr, weave_param, include_center = include_center)
    _,num_rows, num_cols = N_weave.shape
        
    N_weave[np.where(N_weave == 0)] = -1
    
    I_part = np.zeros([num_filters,num_rows,num_cols,tensor_dimension])

    for fil in range(num_filters):
        for i in range(num_rows):
            for j in range(num_cols):
                i_j_loc = np.where(N_weave[fil,i,j] == N_arr)
                if len(i_j_loc[0]) > 0:
                    I_part[fil,i,j] = i_j_loc
                else:
                    I_part[fil,i,j] = np.array([0,0,num_rows//weave_param['num_zeros']])
   
    return I_part.astype(int)