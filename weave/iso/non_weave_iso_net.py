#from tensorflow_weave import *
from .non_weave_iso_unit import *
import numpy as np

import keras
import tensorflow as tf
import tensorflow.python.keras
from keras.layers import Conv2D, Add, ZeroPadding2D, MaxPool2D, Input, Flatten, Dense
from keras.models import Model


def in_line_net(num_layers, num_filters, method,
				filter_size=(3, 3), max_pool_alt=False,
				mid_layer=100):
	"""
	Returns a model that has either the same theoretical train time as
	a pyrm-net fully optimized on sufficent GPUS or the same number of
	parameters as a pyrm-net.
	"""
	if method == 'time':
		num_layers = num_layers
		raise ValueError('Not Implemented Yet')
		#return iso_time_net(num_layers, num_filters, filter_size, max_pool_alt, mid_layer)
	elif method == 'param':
		return iso_param_net(num_layers, num_filters, filter_size, max_pool_alt, mid_layer)
	else:
		raise ValueError("Method must be eithr 'time' or 'param'.")


def iso_param_net(num_layers, num_filters, filter_size, max_pool_alt, mid_layer):
	
	inputs = Input(shape=(3,32,32))
	pad_size = (filter_size[0]-1)/2

	conv_layer_size = 2 ** (num_layers - 1)	

	x = non_weave_unit(inputs,num_filters,filter_size=filter_size, pad_size=pad_size)

	for layer in range(1,num_layers):
		num_filters *= 2
		if (layer+1) % 2 and max_pool_alt:
			conv_layer_size /= 2
			x = MaxPool2D()(x)
		elif max_pool_alt:
			conv_layer_size /= 2
		else:
			conv_layer_size /= 2
			x = MaxPool2D()(x)

		x = non_weave_unit(x, num_filters,filter_size=filter_size, pad_size=pad_size)

	x = MaxPool2D()(x)
	x = Flatten()(x)
	x = Dense(mid_layer, activation = 'relu')(x)
	predictions = Dense(10, activation='softmax')(x)

	# This creates a model that includes
	# the Input layer and three Dense layers
	model = Model(inputs=[inputs], outputs=predictions)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


def iso_time_net(num_layers, num_filters, filter_size, max_pool_alt, mid_layer):
	"""
	Returns a model that has the same theoretical training time as a pyrm-net
	"""
	inputs = Input(shape=(3,32,32))
	pad_size = (filter_size[0]-1)/2

	if max_pool_alt:
		pool = 2
	else:
		pool = 1

	x = non_weave_unit(inputs, num_filters, filter_size, pad_size)

	layer_count = 1
	for layer in range(1, num_layers):

		if layer_count % pool == 0:
			x = MaxPool2D()(x)
		else:
			pass
		x = non_weave_unit(x, num_filters, filter_size, pad_size)

	#x = MaxPool2D()(x)
	x = Flatten()(x)
	x = Dense(mid_layer, activation='relu')(x)
	predictions = Dense(10, activation='softmax')(x)

	# This creates a model that includes
	# the Input layer and three Dense layers
	model = Model(inputs=[inputs], outputs=predictions)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
