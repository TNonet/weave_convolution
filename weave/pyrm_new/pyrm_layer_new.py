from keras.layers import Conv2D, Add, ZeroPadding2D


def pyrmlayer(inputs,
	n_units,
	n_filters,
	ava_gpu,
	disjoint = True,
	pure_combine =  False,
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
	ava_gpu -> (List of) string names for GPUs available to Layer
	center -> (boolean) whether the ArrayWeave includes center
	n_filters -> (non-negative integer) the number of filters for the first
				set of convolution filters in this layer.
	"""
	layer_out = []


	for unit in range(n_units):
		unit_input = [inputs[2*unit],inputs[2*unit+1]]
		unit_devices = [ava_gpu[2*unit], ava_gpu[2*unit+1]]
		x_temp = pyrm_unit(unit_input,
							n_filters = n_filters, 
							unit_devices,
							disjoint = disjoint,
							pure_combine = pure_combine,
							center = center,
							r_combine = r_combine,
							pre_pad = pre_pad,
							filter_size = filter_size)
		layer_out.append(x_temp)

	return layer_out

