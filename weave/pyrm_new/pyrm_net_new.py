from pyrm_layer_new import pyrmlayer
import numpy as np
from keras.layers import MaxPool2D, Input, Flatten, Dense
from keras.models import Model

def pyrm_net(input_size,
			n_layers,
			n_filters_start,
			n_gpus,
			inputs = False,
			r_filter = 2,
			r_combine = 2,
			max_pool_loc = 2,
			pure_combine = False,
			pre_pad = True,
			end_max_pool = True,
			min_dim = 8,
			center = False,
			gpu_only = False,
			filter_size = (3,3)):
	"""
	inputs:
	inputs -> (4-D Tensor) with shape (None, 3, H, W) representing RGB images

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

	#Determine the Number of Layers:
	gpu_layers = float(max(gpu_only * int(np.log2(n_gpus)), (1 - gpu_only) * float('inf')))
	print('GPU settings allow for %f layers' % gpu_layers)
	min_length = min(input_size[1:])
	#Size Layer is still werid as it determines size_layers based on max_pool_loc even if it doenst use max pool
	#Need to Fix!
	size_layers = (int(np.log2(min_length/float(min_dim))) - end_max_pool)*max_pool_loc
	print('Minimum output size allow for %d layers' % size_layers)
	n_layers = int(min(n_layers,gpu_layers,size_layers))

	print('Number of layers %d' % n_layers)

	#Creating Device Library
	#n_gpus take first n spots of library and back fill with CPU
	ava_devices = ['/gpu:%d' % i for i in range(n_gpus)]

	layer_size = 2 ** (n_layers - 1)

	print('First Layer Size: %d (Number of Units)' % layer_size)

	if not gpu_only:
		for _ in range(2*layer_size - n_gpus):
			ava_devices.append('/cpu:0')
	else:
		pass

	#Creating Layers
	#First Layer Dependent on input type

	if inputs == False:
		first_disjoint = False
		inputs = Input(shape=(3,32,32))
		layer_in = [inputs for _ in range(layer_size)]
	else:
		if len(inputs) == 2*layer_size:
			layer_in = inputs
			first_disjoint = True
		else:
			raise ValueError('With tensor input (size {}) must match layer_size {})'.format(len(inputs), 2*layer_size))

	n_filters = n_filters_start
	layer_out = pyrmlayer(layer_in,
							n_units=layer_size,
							n_filters=n_filters,
							ava_devices = ava_devices,
							disjoint = first_disjoint,
							pure_combine =  pure_combine,
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
	if end_max_pool:
		x = MaxPool2D()(layer_out[0])
	else:
		x = layer_out[0]

	# x = Flatten()(x)
	# x = Dense(100, activation = 'relu')(x)
	# predictions = Dense(10, activation='softmax')(x)

	# # This creates a model that includes
	# # the Input layer and three Dense layers
	# model = Model(inputs=[inputs], outputs=predictions)
	# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return x
