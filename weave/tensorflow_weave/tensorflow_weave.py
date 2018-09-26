import numpy as np

import keras
import tensorflow as tf
import tensorflow.python.keras
from ..cython_weave.cython_weave import *
from tensorflow_weave_help import *


from keras import backend as K
K.set_image_data_format('channels_first')

class ZeroWeave(keras.layers.Layer):
    """
    Stil need to work on multy dimensions
    """

    def __init__(self, num_zeros = 2, filter_size = 3):
        super(ZeroWeave, self).__init__()
        self.num_zeros = num_zeros
        self.filter_size = filter_size
        
    def build(self, input_shape):
        """
        input_shape = (None, depth, height, width)
        """

        _, num_filters, height, width = input_shape
        if height != width:
            raise ValueError('Must operate on a square image')

        self.tensor_indexor = create_part_I_zero_weave_matrix((num_filters,height,width),
            {'num_zeros':self.num_zeros,'filter_size':self.filter_size})

        self.tensor_indexor = tf.convert_to_tensor(self.tensor_indexor)

        def fn(x):
            return tf.gather_nd(x, self.tensor_indexor)

        self.fn = fn

        super(ZeroWeave, self).build(input_shape)
        
    def call(self, inputs):
        """
        Must expand input by a empty column to remove derivates
        """
        cat_inputs = tf.concat([inputs, tf.zeros_like(inputs)], axis = 3)

        return tf.map_fn(self.fn, cat_inputs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1],
                input_shape[2]*(self.num_zeros + 1) - self.num_zeros,
                input_shape[3]*(self.num_zeros + 1) - self.num_zeros)

class ArrayWeave(keras.layers.Layer):
    """
    Layer that takes input of peripherial convolution layer and 
    returns a weave of the image to be merged with the local convolution layers
    """

    def __init__(self, filter_size = 3, num_zeros = 2, include_center = False):
        super(ArrayWeave, self).__init__()
        #Dimensions of Large output layer
        self.include_center = include_center
        self.filter_size = filter_size
        self.num_zeros = num_zeros
        
    def build(self, input_shape):
        """
        input_shape = (None, depth, height, width)
        """
        _, num_filters, height, width = input_shape
        if height != width:
            raise ValueError('Must operate on a square image')

        self.tensor_indexor = create_part_I_array_weave_matrix((num_filters,height,width),
            {'num_zeros':self.num_zeros,'filter_size':self.filter_size},
            include_center = self.include_center)
        self.tensor_indexor = tf.convert_to_tensor(self.tensor_indexor)

        def fn(x):
            return tf.gather_nd(x, self.tensor_indexor)

        self.fn = fn

        super(ArrayWeave, self).build(input_shape)
        
    def call(self, inputs):
        """
        Must expand input by a empty column to remove derivates
        """
        cat_inputs = tf.concat([inputs, tf.zeros_like(inputs)], axis = 3)

        return tf.map_fn(self.fn, cat_inputs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1],
                input_shape[2]*(self.num_zeros + 1) - self.num_zeros,
                input_shape[3]*(self.num_zeros + 1) - self.num_zeros)

class old_ZeroWeave(keras.layers.Layer):
    """
    Stil need to work on multy dimensions
    """

    def __init__(self, num_img, num_zeros = 2, filter_size = 3):
        super(ZeroWeave, self).__init__()
        self.num_img = num_img
        self.num_zeros = num_zeros
        self.filter_size = filter_size
        
    def build(self, input_shape, max_num_image = 16):
        """
        input_shape = (None, depth, height, width)
        """
        _, num_filters, height, width = input_shape

        if height != width:
            raise ValueError('Must operate on a square image')

	    # try:
	    # 	self.tensor_indexor = load_full_I_zero_matrix(input_shape,
	    #                                                           max_num_image,
	    #                                                           {'num_zeros':self.num_zeros,
	    #                                                            'filter_size':self.filter_size})
	    # except:
        self.tensor_indexor = create_full_I_zero_weave_matrix_fast(input_shape,
        	max_num_image,
        	{'num_zeros':self.num_zeros,
        	'filter_size':self.filter_size})

        self.tensor_indexor = tf.convert_to_tensor(self.tensor_indexor)

        super(ZeroWeave, self).build(input_shape)

    def call(self, inputs):
        cat_inputs = tf.concat([inputs, tf.zeros_like(inputs)], axis = 3)
        return tf.gather_nd(cat_inputs, self.tensor_indexor[:self.num_img])
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1],
                input_shape[2]*(self.num_zeros + 1) - self.num_zeros,
                input_shape[3]*(self.num_zeros + 1) - self.num_zeros)
    
class old_ArrayWeave(keras.layers.Layer):
    """
    Layer that takes input of peripherial convolution layer and 
    returns a weave of the image to be merged with the local convolution layers
    """

    def __init__(self, num_img, filter_size = 3, num_zeros = 2):
        super(ArrayWeave, self).__init__()
        #Dimensions of Large output layer
        self.filter_size = filter_size
        self.num_zeros = num_zeros
        self.num_img = num_img
        #self.output_dim = self.compute_output_shape(inputs)
        
    def build(self, input_shape, max_num_image = 16):
        """
        input_shape = (None, depth, height, width)
        """
        _, num_filters, height, width = input_shape
        if height != width:
            raise ValueError('Must operate on a square image')

        # try:
        # 	self.tensor_indexor = load_full_I_array_weave_matrix(input_shape,
	       #                                                         max_num_image,
	       #                                                         {'num_zeros':self.num_zeros,
	       #                                                          'filter_size':self.filter_size})
        # except:
        self.tensor_indexor = create_full_I_array_weave_matrix_fast(input_shape,
        	max_num_image,
        	{'num_zeros':self.num_zeros,
        	'filter_size':self.filter_size})

        self.tensor_indexor = tf.convert_to_tensor(self.tensor_indexor)
        super(ArrayWeave, self).build(input_shape)
        
    def call(self, inputs):
        """
        Must expand input by a empty column to remove derivates
        """
        cat_inputs = tf.concat([inputs, tf.zeros_like(inputs)], axis = 3)
        return tf.gather_nd(cat_inputs, self.tensor_indexor[:self.num_img])
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1],
                input_shape[2]*(self.num_zeros + 1) - self.num_zeros,
                input_shape[3]*(self.num_zeros + 1) - self.num_zeros)

    def get_config(self):
        base_config = super(ArrayWeave, self).get_config()
        base_config['output_dim'] = self.output_dim

    @classmethod
    def from_config(cls, config):
        return cls(**config)