from pyrm_layer_new import pyrmlayer
import numpy as np
from keras.layers import MaxPool2D, Input, Flatten, Dense
from keras.models import Model

def pyrm_net(inputs,
			n_layers,
			n_filters_start,
			n_gpus,
			r_filter = 2,
			r_combine = 2,
			max_pool_loc = 2,
			pure_combine = False,
			pre_pad = True,
			min_dim = 8,
			center = False,
			gpu_only = False,
			filter_size = (3,3)):
	"""
	inputs:
	inputs -> List of (4-D Tensor) with shape (None, 3, H, W) representing RGB images.

	parameters:
	input_size -> (Tupe of length 3) representing the size of the image array
				of a single image.
	n_gpu -> (non-negative integer) number of gpus available on machine
				can limit number of layers for GPU calcs
	gpu_only -> (boolean) whether layers should operate on CPU's if there
				is a need for GPUS
	center -> (boolean) whether the ArrayWeave includes center
	min_size -> (non-negative integer) the minimum dimension that ArrayWeave
				can operate on.
	n_layers -> (non-negative integer) the numbers of layers in a pyrm-net:
				n_layers = min(n_layers,
				 			fn(gpu_only,n_gpu) #Can we use gpus? if so how big?
				 			fn(max_pool_loc,min_size)) #how small can we make the image?
	n_filters -> (non-negative integer) the number of filters on the first layer
	r_filters -> (non-negative integer) the ratio at which the number of filters
				changes as layers are advanced
	max_pool_loc -> (non-negative integer) the number of layers passed since
				the first layer for the next MaxPool operation
	end_max_pool -> (boolean) whether a max_pool operat
	"""

	#Disjoint Layers take two input tensors.
	layer_size = len(inputs)/2
	input_size = inputs[0].shape.as_list()
	min_length = input_size[2]

	#Determine the Number of Layers:
	if gpu_only:
		gpu_layers = int(np.log2(n_gpus))
	else:
		gpu_layers = float('inf')

	print('GPU settings allow for %f layers' % gpu_layers)

	size_layers = int(np.log2(min_length/float(min_dim)))*(max_pool_loc+1)

	print('Minimum output size allow for %d layers' % size_layers)
	n_layers = int(min(n_layers,gpu_layers,size_layers))

	print('Number of layers %d' % n_layers)

	#Creating Device Library
	#n_gpus take first n spots of library and back fill with CPU
	ava_devices = ['/gpu:%d' % i for i in range(n_gpus)]

	assert(layer_size == 2 ** (n_layers - 1))

	print('First Layer Size: %d (Number of Units)' % layer_size)

	if not gpu_only:
		for _ in range(2*layer_size - n_gpus):
			ava_devices.append('/cpu:0')
	else:
		pass

	#Creating Layers
	#First Layer Dependent on input type
	n_filters = n_filters_start
	layer_out = pyrmlayer(inputs,
							n_units=layer_size,
							n_filters=n_filters,
							ava_devices = ava_devices,
							disjoint = True,
							pure_combine =  pure_combine,
							batch_norm = False,
							drop = False,
							center = center,
							r_combine = r_combine,
							pre_pad = pre_pad,
							filter_size = filter_size)

	print('First Layer Output Size: %d (Number of Tensors)' % len(layer_out))

	for layer in range(1,n_layers):
		if layer % max_pool_loc == 0:
			layer_in = [MaxPool2D()(tense) for tense in layer_out]
		else:
			layer_in = layer_out

		layer_size /= 2
		n_filters *= r_filter

		layer_out = pyrmlayer(layer_in,
								n_units = layer_size,
								n_filters = n_filters,
								ava_devices = ava_devices,
								disjoint = True,
								pure_combine = pure_combine,
								center = center,
								r_combine = r_combine,
								pre_pad = pre_pad,
								filter_size = filter_size)

	print('Final Layer Size %d' % len(layer_out))
	assert(len(layer_out) == 1)
	x = layer_out[0]

	return x
