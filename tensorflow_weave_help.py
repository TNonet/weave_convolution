import numpy as np
import tensorflow as tf
from cython_weave import *


def create_part_I_zero_weave_matrix(input_shape, weave_param):
    """
    creases properly sized 5D indexing tensor, I_zero_weave that can be applied to
    tf.gather_nd(x, I_zero_weave) that will return a tensor that is identical to
    x = explode_tensor(tensor, num_zeros = 2)

    However, this takes a while to build in favor of fast execution time in the graph
    thus we ust set a max number of images before we build.

    Creates an matrix of size (num_images, num_filters, height, width, input_tensor_dimension (4))

    >>>>>>
    Should work on loading in the data not just building it each time
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

def create_part_I_array_weave_matrix(input_shape, weave_param):
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
    N_weave,_ = array_weave_fast_forward(N_arr, weave_param)
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

def create_full_I_zero_weave_matrix_fast(input_shape, max_num_images, weave_param):
    """
    createses properly sized 5D indexing tensor, I_array_weave that can be applied to
    tf.gather_nd(x, I_array_weave) that will return a tensor that is identical to
    x, cache = array_weave_fast_forward(tensor, num_zeros = 2)

    However, this takes a while to build in favor of fast execution time in the graph
    thus

    Creates an matrix of size (num_images, num_filters, height, width, input_tensor_dimension (4))
    """
    num_images,num_filters,num_rows,num_cols = input_shape
    num_images = max(max_num_images, num_images)

    tensor_dimension = len(input_shape)
    if tensor_dimension != 4:
        raise ValueError('Must operate on 4D tensors')
    if num_rows != num_cols:
        raise ValueErrro('Must operate ona square image')
    
    N_arr = np.arange(1,num_cols**2 + 1).reshape(
        [num_rows,num_cols]).astype(float)
    N_weave,_ = zero_weave_fast_forward(N_arr, weave_param)
    num_rows, num_cols = N_weave.shape
        
    N_weave[np.where(N_weave == 0)] = -1
    
    I_temp = np.zeros([num_rows,num_cols,tensor_dimension])
    for i in range(num_cols):
        for j in range(num_rows):
            i_j_loc = np.where(N_weave[i,j] == N_arr)
            if len(i_j_loc[0]) > 0:
                I_temp[i,j] = np.array([-2,-3,i_j_loc[0],i_j_loc[1]])
            else:
                I_temp[i,j] = np.array([0,0,0,num_cols//(weave_param['num_zeros'])])
    
    I_full = np.zeros([num_images,num_filters,num_rows,num_cols,tensor_dimension])
    
    for img in range(num_images):
        for layer in range(num_filters):
            I_temp_current = np.copy(I_temp)
            I_temp_current[np.where(I_temp_current == -2)] = img
            I_temp_current[np.where(I_temp_current == -3)] = layer
            I_full[img,layer] = I_temp_current
                        
    return I_full.astype(int)

def create_full_I_zero_weave_matrix(input_shape, max_num_images, weave_param):
    """
    creases properly sized 5D indexing tensor, I_zero_weave that can be applied to
    tf.gather_nd(x, I_zero_weave) that will return a tensor that is identical to
    x = explode_tensor(tensor, num_zeros = 2)

    However, this takes a while to build in favor of fast execution time in the graph
    thus we ust set a max number of images before we build.

    Creates an matrix of size (num_images, num_filters, height, width, input_tensor_dimension (4))

    >>>>>>
    Should work on loading in the data not just building it each time
    """
    num_images ,num_filters,num_rows,num_cols = input_shape
    num_images = max(num_images, max_num_images)
    num_zeros = weave_param['num_zeros']
    tensor_dimension = len(input_shape)
    
    if tensor_dimension != 4:
        raise ValueError('Must operate on 4D tensors')
    if num_rows != num_cols:
        raise ValueErrro('Must operate ona square image')
        
    HH = WW = num_rows*(1+num_zeros) - num_zeros
     

    N_arr = np.arange(1,num_cols**2 * num_filters * num_images + 1).reshape(
        [num_images,num_filters,num_rows,num_cols]).astype(float)
    
    N_zero_weave = np.zeros([num_images,num_filters,HH,WW])
    N_zero_weave[:,:,::3,::3] = N_arr
    I_full = np.zeros([num_images,num_filters,HH,WW,tensor_dimension])

    N_zero_weave[np.where(N_zero_weave == 0)] = -1

    for img in range(num_images):
        print(img)
        for fil in range(num_filters):
            for i in range(HH):
                for j in range(WW):
                    i_j_loc = np.where(N_zero_weave[img,fil,i,j] == N_arr)
                    if len(i_j_loc[0]) > 0:
                        I_full[img,fil,i,j] = i_j_loc
                    else:
                        I_full[img,fil,i,j] = np.array([0,0,0,HH//weave_param['num_zeros']])
   
    return I_full.astype(int)
        
def create_full_I_array_weave_matrix_fast(input_shape, max_num_images, weave_param):
    """
    createses properly sized 5D indexing tensor, I_array_weave that can be applied to
    tf.gather_nd(x, I_array_weave) that will return a tensor that is identical to
    x, cache = array_weave_fast_forward(tensor, num_zeros = 2)

    However, this takes a while to build in favor of fast execution time in the graph
    thus

    Creates an matrix of size (num_images, num_filters, height, width, input_tensor_dimension (4))
    """
    num_images,num_filters,num_rows,num_cols = input_shape
    num_images = max(max_num_images, num_images)

    tensor_dimension = len(input_shape)
    if tensor_dimension != 4:
        raise ValueError('Must operate on 4D tensors')
    if num_rows != num_cols:
        raise ValueErrro('Must operate ona square image')
    
    N_arr = np.arange(1,num_cols**2 + 1).reshape(
        [num_rows,num_cols]).astype(float)
    N_weave,_ = array_weave_fast_forward(N_arr, weave_param)
    num_rows, num_cols = N_weave.shape
        
    N_weave[np.where(N_weave == 0)] = -1
    print(num_cols//(weave_param['num_zeros']+1))
    
    I_temp = np.zeros([num_rows,num_cols,tensor_dimension])
    for i in range(num_cols):
        for j in range(num_rows):
            i_j_loc = np.where(N_weave[i,j] == N_arr)
            if len(i_j_loc[0]) > 0:
                I_temp[i,j] = np.array([-2,-3,i_j_loc[0],i_j_loc[1]])
            else:
                I_temp[i,j] = np.array([0,0,0,num_cols//(weave_param['num_zeros'])])
    
    I_full = np.zeros([num_images,num_filters,num_rows,num_cols,tensor_dimension])
    
    for img in range(num_images):
        for layer in range(num_filters):
            I_temp_current = np.copy(I_temp)
            I_temp_current[np.where(I_temp_current == -2)] = img
            I_temp_current[np.where(I_temp_current == -3)] = layer
            I_full[img,layer] = I_temp_current
                        
    return I_full.astype(int)

def create_full_I_array_weave_matrix(input_shape, max_num_images, weave_param):
    """
    createses properly sized 5D indexing tensor, I_array_weave that can be applied to
    tf.gather_nd(x, I_array_weave) that will return a tensor that is identical to
    x, cache = array_weave_fast_forward(tensor, num_zeros = 2)

    However, this takes a while to build in favor of fast execution time in the graph
    thus

    Creates an matrix of size (num_images, num_filters, height, width, input_tensor_dimension (4))
    """
    num_images,num_filters,num_rows,num_cols = input_shape
    num_images = max(max_num_images, num_images)

    tensor_dimension = len(input_shape)
    if tensor_dimension != 4:
        raise ValueError('Must operate on 4D tensors')
    if num_rows != num_cols:
        raise ValueErrro('Must operate ona square image')
    
    N_arr = np.arange(1,num_cols**2 * num_filters * num_images + 1).reshape(
        [num_images,num_filters,num_rows,num_cols]).astype(float)
    N_weave,_ = array_weave_fast_forward(N_arr, weave_param)
    _,_,num_rows, num_cols = N_weave.shape
    I_full = np.zeros([num_images,num_filters,num_rows,num_cols,tensor_dimension])
        
    N_weave[np.where(N_weave == 0)] = -1
    
    for img in range(num_images):
        print(img)
        for fil in range(num_filters):
            for i in range(num_cols):
                for j in range(num_rows):
                    i_j_loc = np.where(N_weave[img,fil,i,j] == N_arr)
                    if len(i_j_loc[0]) > 0:
                        I_full[img,fil,i,j] = i_j_loc
                    else:
                        I_full[img,fil,i,j] = np.array([0,0,0,num_cols//(weave_param['num_zeros'])])
                        
    return I_full.astype(int)

def explode_width(tensor, num_zeros = 2):
    h_zero = tf.zeros_like(tensor)
    stack = [tensor]
    for i in range(num_zeros):
        stack.append(h_zero)
    h_expand = tf.reshape(tf.stack(stack, 2),[tf.shape(tensor)[0], tf.shape(tensor)[1]*(num_zeros+1)])[:,:-2]
    return h_expand

def explode_tensor(tensor, num_zeros = 2):
    h_expand = explode_width(tensor, num_zeros=num_zeros)
    rot = tf.transpose(h_expand)
    rot1 = explode_width(rot, num_zeros=num_zeros)
    return rot1