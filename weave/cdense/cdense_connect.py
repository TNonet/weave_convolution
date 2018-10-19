import tensorflow as tf

def _map(inputs, n_units, connection_type):
	"""
	Creates an list of tensors from inputs that
	will be used by cdense.
	"""
	if len(inputs) == n_units:
		output = [[inputs[i-1],inputs[i]] for i in range(n_units)]
	elif len(inputs) < n_units:
		output = []
		input_dim = len(inputs)
		#Units that can connect staight forwards
		output = [[inputs[i-1],inputs[i]] for i in range(input_dim)]
		n_units -= input_dim
		current_input_index = 0
		#Begin successively connecting units to the least conneceted input
		while n_units > 0:
			output.append([inputs[current_input_index], inputs[current_input_index]])
			n_units -= 1
			current_input_index += 1
			#Run to end of list --> All units are equally connected. (Got to beginning)
			if current_input_index == input_dim:
				current_input_index = 0
	else: #(len(inputs) > n_units)
		collpase_units = len(inputs) - n_units
		if n_units == 1:
			pass
		elif collpase_units % 2 == 1:
			raise ValueError('Must be an even difference between previous layer and next layer')
		output = [[inputs[2*i],inputs[2*i + 1]] for i in range(collpase_units)]
		for i in range(2*collpase_units, n_units):
			if i == 2*collpase_units:
				output.append([inputs[-1],inputs[i]])
			else:
				output.append([inputs[i-1],inputs[i]])
	if connection_type == 'shuffle':
		return _shuffle(output)
	else:
		return output

def _shuffle(inputs):
	"""
	Shuffles the inputs to create a random mapping
	between layers.
	"""
	expand_inputs = tf.stack(inputs, axis = -1)
	shuffled_inputs = tf.random_shuffle(expand_inputs)
	return tf.unstack(shuffled_inputs)
