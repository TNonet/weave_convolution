#from tensorflow_weave import *
import numpy as np

import keras
import tensorflow as tf
import tensorflow.python.keras
from keras.layers import Conv2D, Add, ZeroPadding2D, MaxPool2D, Input, Flatten, Dense
from keras.models import Model


def non_weave_unit(inputs, num_filters, filter_size, pad_size, pad = True):
	"""
	Depending on where the unit is being used to keep the number of parameters equal
	the number of filters must change

	I
	"""
	name = 'non_weave_unit'
	if pad:
		name += '_pad'
	else:
		pass

	with tf.name_scope(name):
		if pad:
			x = ZeroPadding2D(padding=(pad_size,pad_size))(inputs)
		else:
			x = inputs

		x = Conv2D(2* num_filters,
						kernel_size = filter_size,
		               	strides=(1,1),
		               	padding='valid',
		               	activation='relu')(x)

		x = ZeroPadding2D(padding=(pad_size,pad_size))(x)

		x = Conv2D(num_filters,
						kernel_size = filter_size,
		               	strides=(1,1),
		               	padding='valid',
		               	activation='relu')(x)

	return x