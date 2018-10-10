from keras.layers import Conv2D, Add, ZeroPadding2D
from pyrm_unit_new import *
from keras.layers import Dropout, BatchNormalization


def pyrmlayer(inputs,
	n_units,
	n_filters,
	ava_devices,
	disjoint = True,
	pure_combine =  False,
	batch_norm = True,
	drop = 0.2,
	center = False,
	r_combine = 1,
	pre_pad = True,
	filter_size = (3,3)):
	"""
	inputs:
	inputs -> (List of) (4-D Tesnors) with shape (None, F, HH, WW)	

	parameters:
	gpu_only -> (boolean) whether layers should operate on CPU's if there
				is a need for GPUS
	ava_devices -> (List of) string names for GPUs available to Layer
	center -> (boolean) whether the ArrayWeave includes center
	n_filters -> (non-negative integer) the number of filters for the first
				set of convolution filters in this layer.
	"""
	layer_out = []

	assert(inputs > 0, 'Need to have at least one input to layer')
	if disjoint:
		assert(len(inputs) % 2 ==  0, 'Need to have a 2^N number of inputs for each layer')
		assert(len(inputs) == 2*n_units, "Need to have 2*n_units as inputs")
	else:
		assert(len(inputs) == n_units, 'Need a unit for each input')

	for ind in range(1,len(inputs)):
		assert(inputs[ind-1].shape == inputs[ind].shape, "Each input must have the same shape")


	for unit in range(n_units):
		if disjoint:
			if batch_norm:
				unit_input_1 = BatchNormalization(axis =1)(inputs[2*unit])
				unit_input_2 = BatchNormalization(axis =1)(inputs[2*unit + 1])
			else:
				unit_input_1 = inputs[2*unit]
				unit_input_2 = inputs[2*unit + 1]
			if drop:
				unit_input_1 = Dropout(drop)(unit_input_1)
				unit_input_2 = Dropout(drop)(unit_input_2)
			else:
				pass

			unit_input = [unit_input_1,unit_input_2]
		else:
			if batch_norm:
				unit_input_1 = BatchNormalization(axis =1)(inputs[unit])
			else:
				unit_input_1 = inputs[unit]
			if drop:
				unit_input_1 = Dropout(drop)(unit_input_1)
			else:
				pass
			unit_input = [unit_input_1,unit_input_1]

		unit_devices = [ava_devices[2*unit], ava_devices[2*unit+1]]
		x_temp = pyrm_unit(unit_input,
							n_filters = n_filters, 
							devices = unit_devices,
							disjoint = disjoint,
							pure_combine = pure_combine,
							center = center,
							r_combine = r_combine,
							pre_pad = pre_pad,
							filter_size = filter_size)
		layer_out.append(x_temp)

	return layer_out

