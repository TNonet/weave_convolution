import numpy as np

import keras
import tensorflow as tf
import tensorflow.python.keras
from pyrm_unit import *
from keras.layers import Input, Conv2D, Dense, Flatten, Add, ZeroPadding2D, add, MaxPool2D
from keras.models import Model


def build_pyrm_net(num_layers, num_filters, mid_layer = 100, pure_combine = False):
	inputs = Input(shape=(3,32,32))
	tf.cast(inputs, dtype=tf.float64)


	layer_size = 2 ** (num_layers - 1)
	###First Layer is Different!
	prev_layer_out = []
	for _ in range(layer_size):
		prev_layer_out.append(pyrm_weave(inputs, num_filters, pure_combine = pure_combine))

	for layer in range(1,num_layers):
		layer_size /= 2
		layer_out = []
		for ind in range(layer_size):
			layer_input = [prev_layer_out[2*ind], prev_layer_out[2*ind+1]]
			layer_out.append(pyrm_weave(layer_input, num_filters, pure_combine = pure_combine))
		prev_layer_out = layer_out

	print(len(prev_layer_out))

	x = prev_layer_out[0]
	x = Flatten()(x)
	x = Dense(mid_layer, activation = 'relu')(x)
	predictions = Dense(10, activation='softmax')(x)

	# This creates a model that includes
	# the Input layer and three Dense layers
	model = Model(inputs=[inputs], outputs=predictions)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

