import tensorflow as tf

DEFAULT_ACTIVATION = tf.nn.relu
DEFAULT_INITIALIZER = tf.initializers.glorot_normal()
BN_TRAINING = True


def layer(op):
	"""Decorator for chaining components of layer"""

	def layer_decorated(self, *args, **kwargs):

		name = kwargs.setdefault('name', 'no_given_name')

		terminal_length = self.terminals

		if terminal_length == 0:
			raise RuntimeError(f'No input variables found for layer {name}')
		elif terminal_length == 1:
			layer_input = self.terminals[0]
		else:
			layer_input = list(self.terminals)

		layer_output = op(self, layer_input, *args, **kwargs)

		self.feed(layer_output)

		return self


class Network:

	def __init__(self):
		self.terminals = list()
		self._build()

	def _build(self, is_training):
		raise NotImplementedError('Must be implemented by the subclass.')

	def feed(self, tensor):
		self.terminals = [tensor]

		return self

	@layer
	def conv(self, inputs, filters, kernal_size=3, strides=1, padding='SAME',
			 activation=DEFAULT_ACTIVATION, use_bias=False, kernel_initializer=DEFAULT_INITIALIZER,
			 name=None, reuse=tf.AUTO_REUSE):
		output = tf.layers.conv2d(
			inputs=inputs,
			filters=filters,
			kernal_size=kernal_size,
			strides=strides,
			padding=padding,
			activation=activation,
			use_bias=use_bias,
			kernel_initializer=kernel_initializer,
			name=name,
			reuse=reuse,
		)

		return output

	@layer
	def conv_nn(self, inputs, filters, rate=1, strides=[1, 1, 1, 1], padding='SAME',
				activation=DEFAULT_ACTIVATION, use_bias=False, kernel_initializer=DEFAULT_INITIALIZER,
				name=None, reuse=tf.AUTO_REUSE):
		with tf.variable_scope(name):
			kernels = tf.get_variable(name='kernel', shape=filters, initializer=kernel_initializer)

			x = tf.nn.conv2d(inputs, kernels, dilations=[1, rate, rate, 1], strides=strides, padding=padding)

			if use_bias:
				bias = tf.get_variable(name='bias', shape=[filters[3]], initializer=tf.constant_initializer(value=0))
				x = tf.nn.bias_add(x, bias)

			if activation is not None:
				x = activation(x)

			return x

	@layer
	def batch_normalization(self, inputs, name=None, training=True, activation=DEFAULT_ACTIVATION):
		output = tf.layers.batch_normalization(
			inputs=inputs,
			momentum=0.95,
			epsilon=1e-5,
			training=training,
			name=name,
		)

		if activation is not None:
			output = activation(output)

		return output
