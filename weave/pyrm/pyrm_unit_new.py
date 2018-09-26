
from ..tensorflow_weave.tensorflow_weave import *
import numpy as np
import keras
import tensorflow as tf
import tensorflow.python.keras
from keras.layers import Conv2D, Add, ZeroPadding2D

def pyrm_unit(inputs,
	n_filters,
	device,
	disjoint = True,
	pure_combine =  False,
	center = False,
	r_combine = 1,
	pre_pad = True,
	filter_size = (3,3)):
	"""
	inputs -> (List of) 4-D Tensors with shape (None, F, HH, WW)

	parameters:
	n_filters -> (non-negative integer) the number of filters for layer
	device -> (List of) string names for GPUs available to unit
	disjoint -> (boolean) wheter inputs is a list of multiple 
				tensors to combine or one to duplicate!
	pure_combine -> (boolean) whether inputs should be combined with
				ArrayWeave and ZeroWeave without convoultion
	center -> (boolean) whether the ArrayWeave includes center
	r_combine -> (non-negative integer) the raito of filters from
				Weave Conv Layers to the Combination Layers
	pre_pad -> (boolean) whether inputs should be padded before 
				convolution
	static filter_size -> (3,3)
				#Would like to generalize to different filters

	Wrapper for pyrm_weave_unit

	if disjoint:
		if not pure_combine: 
			inputs[0] -> conv (local) --> ArrayWeave -
			                                           >--> Add --> conv (join) --> ouput
			inputs[1] -> conv (perip) --> ZeroWeave --
		if pure_combine:
			inputs[0] --> ArrayWeave -
	                                  >--> Add --> conv (join) --> ouput
			inputs[1] --> ZeroWeave --
	else:
		     --> conv (local) --> ArrayWeave -
	inputs[0]                                >--> Add --> conv (join) --> ouput
		     --> conv (perip) --> ZeroWeave --					
	"""

	if disjoint:
		if len(inputs) != 2:
			raise ValueError('Must operate on only two (possible) tensors')
		if inputs[0].shape.as_list() != inputs[1].shape.as_list():
			raise ValueError('Must operate on tensors of the same size')
		if pure_combine:
			with tf.name_scope('pyrm_weave_disjoint_pure_combine_unit'):
				return pyrm_weave_disjoint_pure_combine(inputs = inputs,
														device = device,
														n_filters = n_filters,
														device = device,
														filter_ratio = filter_ratio, 
														center = center,
														filter_size = filter_size):
		else:
			with tf.name_scope('pyrm_weave_disjoint_not_pure_combine_unit'):
				return pyrm_weave_disjoint_not_pure_combine(inputs = inputs,
															n_filters = n_filters,
															device = device,
															filter_ratio = filter_ratio, 
															center = center,
															r_combine = r_combine,
															pre_pad = pre_pad,
															filter_size = filter_size):
	else:
		if pure_combine:
			raise ValueError('Purely combing a single image is just a normal convolution layer')
		with tf.name_scope('pyrm_weave_joint_unit'):
			return pyrm_weave_joint(inputs = inputs,
									n_filters = n_filters,
									device = device,
									center = center,
									r_combine = r_combine,
									pre_pad = pre_pad,
									filter_size = filter_size):


def pyrm_weave_joint(inputs,
	n_filters,
	device, 
	center = False,
	r_combine = 1,
	pre_pad = True,
	filter_size = (3,3)):
	"""
			 --> conv (local) --> ArrayWeave -
	inputs[0]                                 >--> Add --> conv (join) --> ouput
		     --> conv (perip) --> ZeroWeave --		
	"""
	s_stride = (1,1)
	l_stride = filter_size
	pad_size = int((filter_size[0] - 1)/2)
	num_filters_join = int(n_filters*r_combine)

	if num_filters_join < 1:
		raise ValueError('There must be at least one filter joining the Array and Zero Weave Layers')

	if pre_pad:
		x = ZeroPadding2D(padding=(pad_size,pad_size))(inputs)
	else:
		x = inputs


	x_per = Conv2D(n_filters,
	               kernel_size = filter_size,
	               strides=(1,1),
	               padding='valid',
	               activation='relu')(x)

	x_loc = Conv2D(n_filters,
	               kernel_size= filter_size,
	               strides=(1,1),
	               padding='valid',
	               activation = 'relu')(x)

	x_zero = ZeroWeave()(x_loc)
	x_weave = ArrayWeave(include_center = center)(x_per)

	x = Add()([x_weave, x_zero])

	x = ZeroPadding2D(padding=(pad_size,pad_size))(x)

	x = Conv2D(num_filters_join,
	           kernel_size= filter_size,
	           strides=l_stride,
	           padding='valid',
	           activation = 'relu')(x)
	return x


def pyrm_weave_disjoint_not_pure_combine(inputs,
	n_filters,
	device,
	center = False,
	r_combine = 1,
	pre_pad = True,
	filter_size = (3,3)):
	"""
	inputs[0] -> conv (local) --> ArrayWeave -
	                                          >--> Add --> Conv (join) --> ouput
	inputs[1] -> conv (perip) --> ZeroWeave --
	"""
	s_stride = (1,1)
	l_stride = filter_size
	pad_size = int((filter_size[0] - 1)/2)
	num_filters_join = int(n_filters*r_combine)

	if num_filters_join < 1:
		raise ValueError('There must be at least one filter joining the Array and Zero Weave Layers')

	x0 = inputs[0]
	x1 = inputs[1]

	if pad_state:
		x0 = ZeroPadding2D(padding=(pad_size,pad_size))(x0)
		x1 = ZeroPadding2D(padding=(pad_size,pad_size))(x1)
	else:
		pass

	x_per = Conv2D(n_filters,
	               kernel_size = filter_size,
	               strides=(1,1),
	               padding='valid',
	               activation='relu')(x0)

	x_loc = Conv2D(n_filters,
	               kernel_size= filter_size,
	               strides=(1,1),
	               padding='valid',
	               activation = 'relu')(x1)

	x_zero = ZeroWeave()(x_loc)
	x_weave = ArrayWeave(include_center = center)(x_per)

	x = Add()([x_weave, x_zero])

	x = ZeroPadding2D(padding=(pad_size,pad_size))(x)

	x = Conv2D(num_filters_join,
	           kernel_size= filter_size,
	           strides=l_stride,
	           padding='valid',
	           activation = 'relu')(x)
	return x


def pyrm_weave_disjoint_pure_combine(inputs,
	n_filters,
	device,
	filter_ratio = 1, 
	center = False,
	r_combine = 1,
	pre_pad = True,
	filter_size = (3,3)):
	"""
	inputs[0] --> ArrayWeave -
	                          >--> Add --> Conv (join) --> ouput
	inputs[1] --> ZeroWeave --
	"""
	mid_pad_size = 1
	l_stride = (3,3)
	num_filters_join = int(n_filters*r_combine)
	
	x_loc = inputs[0]
	x_per = inputs[1]

	x_zero = ZeroWeave()(x_loc)
	x_weave = ArrayWeave(include_center = center)(x_per)

	x = Add()([x_weave, x_zero])

	x = ZeroPadding2D(padding=(mid_pad_size,mid_pad_size))(x)

	x = Conv2D(num_filters_join,
	           kernel_size= filter_size,
	           strides=l_stride,
	           padding='valid',
	           activation = 'relu')(x)

	return x

