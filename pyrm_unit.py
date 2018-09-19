from tensorflow_weave import *
import numpy as np

import keras
import tensorflow as tf
import tensorflow.python.keras
from keras.layers import Conv2D, Add, ZeroPadding2D, MaxPool2D

def pyrm_weave(inpts, num_filters, filter_ratio = 1, 
			include_center = 0, pad_state = True, filter_size = (3,3),
			pure_combine = False, max_pool = True):
	"""
	The full function that preforms the either the pyrm_weave_unit if 
	the input is a single 5D tensor, or preforms the merging if
	the input is list of two 5D tensors. This would mean that each
	input should be treated as a seperate image as is being combined
	with the weave method.

	if type(inpts) is list:
		if not pure_combine: 
			inpts[0] -> conv (local) --> ArrayWeave -_
					                                   \
			                                            >--> Add --> Conv (join) --> ouput
					                                   /
			inpts[1] -> conv (perip) --> ZeroWeave ---
		if pure_combine:
			inpts[0] --> ArrayWeave -_
	                                   \
	                                    >--> Add --> Conv (join) --> ouput
	                                   /
			inpts[1] --> ZeroWeave ---
	if type(inpts) is tensor:
			 _-> conv (local) --> ArrayWeave -_
			/                                  \
	inpts -|                                    >--> Add --> Conv (join) --> ouput
			\                                  /
			 --> conv (perip) --> ZeroWeave ---					
	"""

	if type(inpts) is not list:
		#print('First round should be a unit')
		if pure_combine:
			raise ValueError('pure_combine can not function on singel input')
		return pyrm_weave_unit(inpts, num_filters = num_filters,
			filter_ratio = filter_ratio, include_center = include_center,
			pad_state = pad_state, filter_size = filter_size, max_pool = max_pool)
	else:
		#Size Check:
		if len(inpts) != 2:
			raise ValueError('Must operate on only two (possible) tensors')
		if inpts[0].shape.as_list() != inpts[1].shape.as_list():
			raise ValueError('Must operate on tensors of the same size')
		#Tensors should be ready to operate on!
		if pure_combine:
			return pyrm_weave_pure_combine(inpts, num_filters = num_filters,
				include_center  = include_center, filter_size = filter_size,
				max_pool = max_pool)
		else:
			return pyrm_weave_combine(inpts, num_filters = num_filters,
				filter_ratio = filter_ratio, include_center = include_center,
				pad_state = pad_state, filter_size = filter_size, max_pool = max_pool)


def pyrm_weave_combine(inputs, num_filters, filter_ratio = 1, 
			   include_center = 0, pad_state = True, filter_size = (3,3),
			   max_pool = True):
	"""
	inputs[0] -> conv (local) --> ArrayWeave -_
			                                   \
	                                            >--> Add --> Conv (join) --> ouput
			                                   /
	inputs[1] -> conv (perip) --> ZeroWeave ---
	"""

	s_stride = (1,1)
	l_stride = filter_size
	pad_size = int((filter_size[0] - 1)/2)
	num_filters_join = int(num_filters*filter_ratio)

	if num_filters_join < 1:
		raise ValueError('There must be at least one filter joining the Array and Zero Weave Layers')

	x0 = inputs[0]
	x1 = inputs[1]

	if pad_state:
		x0 = ZeroPadding2D(padding=(pad_size,pad_size))(x0)
		x1 = ZeroPadding2D(padding=(pad_size,pad_size))(x1)
	else:
		pass

	x_per = Conv2D(num_filters,
	               kernel_size = filter_size,
	               strides=(1,1),
	               padding='valid',
	               activation='relu')(x0)

	x_loc = Conv2D(num_filters,
	               kernel_size= filter_size,
	               strides=(1,1),
	               padding='valid',
	               activation = 'relu')(x1)

	x_zero = ZeroWeave()(x_loc)
	x_weave = ArrayWeave(include_center = include_center)(x_per)

	x = Add()([x_weave, x_zero])

	x = ZeroPadding2D(padding=(pad_size,pad_size))(x)

	x = Conv2D(num_filters_join,
	           kernel_size= filter_size,
	           strides=l_stride,
	           padding='valid',
	           activation = 'relu')(x)

	if max_pool:
		x = MaxPool2D() (x)

	return x

def pyrm_weave_pure_combine(inputs, num_filters,
							include_center = 0, filter_size = (3,3),
							max_pool = True):
	"""
	inputs[0] --> ArrayWeave -_
	                           \
	                            >--> Add --> Conv (join) --> ouput
	                           /
	inputs[1] --> ZeroWeave ---
	"""
	x_loc = inputs[0]
	x_per = inputs[1]

	x_zero = ZeroWeave()(x_loc)
	x_weave = ArrayWeave(include_center = include_center)(x_per)

	x = Add()([x_weave, x_zero])

	x = ZeroPadding2D(padding=(pad_size,pad_size))(x)

	x = Conv2D(num_filters_join,
	           kernel_size= filter_size,
	           strides=l_stride,
	           padding='valid',
	           activation = 'relu')(x)

	if max_pool:
		x = MaxPool2D() (x)

	return x

def pyrm_weave_unit(inputs, num_filters, filter_ratio = 1, 
			        include_center = 0, pad_state = True,
			        filter_size = (3,3), max_pool = True):
	"""
	A helper function that creates the standard pyrm_weave_unit

			 _-> conv (local) --> ArrayWeave -_
			/                                  \
	input -|                                    >--> Add --> Conv (join) --> ouput
			\                                  /
			 --> conv (perip) --> ZeroWeave ---

	s_stride is the stride of the small local and perip filters
	l_stride is the stride of the large joining filter to return a filter of the same size as input

	"""
	s_stride = (1,1)
	l_stride = filter_size
	pad_size = int((filter_size[0] - 1)/2)
	num_filters_join = int(num_filters*filter_ratio)

	if num_filters_join < 1:
		raise ValueError('There must be at least one filter joining the Array and Zero Weave Layers')

	if pad_state:
		x = ZeroPadding2D(padding=(pad_size,pad_size))(inputs)
	else:
		x = inputs


	x_per = Conv2D(num_filters,
	               kernel_size = filter_size,
	               strides=(1,1),
	               padding='valid',
	               activation='relu')(x)

	x_loc = Conv2D(num_filters,
	               kernel_size= filter_size,
	               strides=(1,1),
	               padding='valid',
	               activation = 'relu')(x)

	x_zero = ZeroWeave()(x_loc)
	x_weave = ArrayWeave(include_center = include_center)(x_per)

	x = Add()([x_weave, x_zero])

	x = ZeroPadding2D(padding=(pad_size,pad_size))(x)

	x = Conv2D(num_filters_join,
	           kernel_size= filter_size,
	           strides=l_stride,
	           padding='valid',
	           activation = 'relu')(x)

	if max_pool:
		x = MaxPool2D() (x)


	return x


