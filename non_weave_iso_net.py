from tensorflow_weave import *
import numpy as np

import keras
import tensorflow as tf
import tensorflow.python.keras
from keras.layers import Conv2D, Add, ZeroPadding2D, MaxPool2D

def in_line_net(num_layers, num_filters, method,
				filter_size = (3,3), max_pool_alt = True,
				mid_layer = 100):
	"""
	Returns a model that has either the same theoretical train time as
	a pyrm-net fully optimized on sufficent GPUS or the same number of
	parameters as a pyrm-net.
	"""
	if method = 'time':
		num_layers = num_layers
		return iso_time_net(num_layers, num_filters, filter_size, max_pool_alt, mid_layer)
	elif method = 'param':
		num_layers = 2 ** (num_layers) - 1
		reutnr iso_param_net(num_layers, num_filters, filter_size, max_pool_alt, mid_layer)
	else:
		raise ValueError("Method must be eithr 'time' or 'param'.")


def iso_time_net(num_layers, num_filters, filter_size, max_pool_alt, mid_layer):
	"""
	Returns a model that has the same theoretical training time as a pyrm-net
	"""
	inputs = Input(shape=(3,32,32))
	tf.cast(inputs, dtype=tf.float64)

	if max_pool_alt:
		pool = 2
	else:
		pool = 1

	x = ZeroPadding2D(padding=(pad_size,pad_size))(x)

	x = Conv2D(2* num_filters,
					kernel_size = filter_size,
	               	strides=(1,1),
	               	padding='valid',
	               	activation='relu')(x)
	x = Conv2D(num_filters,
					kernel_size = filter_size,
	               	strides=(1,1),
	               	padding='valid',
	               	activation='relu')(x)

	layer_count = 1
	for layer in range(1, num_layers):
		x = ZeroPadding2D(padding=(pad_size,pad_size))(x)
		if layer_count % pool == 0:
			x = MaxPool2D()(x)
		else:
			pass
		x = Conv2D(2* num_filters,
					kernel_size = filter_size,
	               	strides=(1,1),
	               	padding='valid',
	               	activation='relu')(x)
		x = Conv2D(num_filters,
					kernel_size = filter_size,
	               	strides=(1,1),
	               	padding='valid',
	               	activation='relu')(x)

	x = Flatten()(x)
	x = Dense(mid_layer, activation = 'relu')(x)
	predictions = Dense(10, activation='softmax')(x)

	# This creates a model that includes
	# the Input layer and three Dense layers
	model = Model(inputs=[inputs], outputs=predictions)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


def iso_param_net(num_layers, num_filters, filter_size, max_pool_alt, mid_layer):
	x_first = Conv2D(2* num_filters,
					kernel_size = filter_size,
	               	strides=(1,1),
	               	padding='valid',
	               	activation='relu')(inputs)
	x = Conv2D(num_filters,
					kernel_size = filter_size,
	               	strides=(1,1),
	               	padding='valid',
	               	activation='relu')(x_first)

	max_pool_counter = 0
	for layer in range(1,num_layers):
		if max_pool_alt:
			max_pool_counter += 1
		else:
			pass
		if max_pool_counter 




	x = Flatten()(x)
	x = Dense(mid_layer, activation = 'relu')(x)
	predictions = Dense(10, activation='softmax')(x)

	# This creates a model that includes
	# the Input layer and three Dense layers
	model = Model(inputs=[inputs], outputs=predictions)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model







