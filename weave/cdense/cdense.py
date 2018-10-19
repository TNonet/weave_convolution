from cdense_layer import *
import cdense_connect
import numpy as np
from keras.layers import MaxPool2D, Input, Flatten, Dense
from keras.models import Model
from ..weave_unit.weave_unit import *

def cDense(inputs,
	n_units,
	n_filters, 
	r_combine = 1,
	pure_combine = False, 
	connection_type = 'diverg', 
	pre_pad = True, 
	center = True, 
	filter_size = (3,3)):
	"""
	inputs --> list of tensors
	n_units --> number of units in layer
	n_filters --> number of filters in unit
	pure_combine --> (boolean) pure combine or simple array weave
	connection_type --> (string) (default) 'diverg' or 'lane' 
					'shuffle' is not implemented but could be a method
					to decrease overfitting
	pad_type --> (string) (default) 'valid' or 'same'
	center --> (boolean) include center peripherial filter
	filter_size --> (tuple) (default) (3,3)
	"""

	devices = ['/gpu:0', '/gpu:0']
	if type(inputs) != list:
		inputs = [inputs]
	print('len(inputs) = %d' % len(inputs))
	print('Inputs', inputs)
	input_dim = len(inputs)
	output = []
	with tf.name_scope('cDense_layer'):
		if input_dim > 2 * n_units:
			raise ValueError('Currently cannot make a layer with less than half units as the previous layer\n')
		else:
			input_layer = cdense_connect._map(inputs = inputs, n_units = n_units, connection_type = connection_type)

		for unit_index in range(n_units):
			output.append(weave_unit(input_layer[unit_index],
									n_filters = n_filters,
									devices = devices,
									disjoint = True,
									pure_combine = pure_combine,
									center = center,
									r_combine = r_combine,
									pre_pad = pre_pad,
									filter_size = filter_size))

	return output















