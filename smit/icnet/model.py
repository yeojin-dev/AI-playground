import tensorflow as tf

from .network import Network


class ICNet(Network):

	def __init__(self, cfg, image_reader=None):

		self.cfg = cfg
		self.mode = cfg.mode
		self.num_classes = cfg.param['num_classes']
		self.ignore_label = cfg.param['ignore_label']
		self.loss_weight = (cfg.LAMBDA1, cfg.LABMDA2, cfg.LABMDA3)
		self.reservoir = dict()
		self.losses = None
		self.start_epoch = 0
		self.step_ph = tf.placeholder(dtype=tf.float32, shape=())

		if self.mode == 'train':
			self.images, self.labels = image_reader.dataset.get_next()
			h, w = self.images.get_shape().as_list()[1:3]
			size = (
				int((h + 1) / 2),
				int((w + 1) / 2),
			)
			self.images2 = tf.image.resize_bilinear(self.images, size=size, align_corners=True)

			super().__init__()
		else:
			raise NotImplementedError('N/A, except for train mode')

	def _res_bottleneck(self, inputs, ch_in, ch_out, strides=(1, 1, 1, 1), name=None, reuse=tf.AUTO_REUSE):

		need_tr = (ch_in != ch_out) or (strides != (1, 1, 1, 1))

		if need_tr:
			scope = name + '_sc'
			with tf.variable_scope(scope, reuse=reuse):
				(self.feed(inputs)
					.conv_nn(
						filters=(1, 1, ch_in, ch_out),
						strides=strides,
						name='ch_modifier',
						activation=None,
						reuse=reuse
					)
					.batch_normalization(name='bn', activation=None))

				sc_out = self.terminals[0] + 0.0
		else:
			sc_out = inputs + 0.0

		depth4 = ch_out // 4
		scope = name + '_mb'
		with tf.variable_scope(scope):
			(self.feed(inputs)
				.conv_nn(filters=(1, 1, ch_in, depth4), strides=strides, activation=None, name='1x1_1', reuse=reuse)
				.batch_normalization(name='1x1_1bn')
				.conv_nn(filters=(3, 3, depth4, depth4), activation=None, name='3x3_2', reuse=reuse)
				.batch_normalization(name='3x3_2bn')
				.conv_nn(filters=(1, 1, depth4, ch_out), name='1x1_3', activation=None, reuse=reuse)
				.batch_normalization(name='1x1_3bn', activation=None))

			mb_out = self.terminals[0] + 0.0

		with tf.variable_scope(name):
			output = sc_out + mb_out
			self.feed(output).activator()

			return self.terminals[0]

	def _branch2(self, inputs, name, reuse=False):

		with tf.variable_scope(name):
			(self.feed(inputs)
				.conv(filters=32, kernel_size=3, strides=2, activation=None, name='conv1_1', reuse=reuse)
				.batch_nomalization(name='conv1_1bn')
				.conv(filters=32, kernel_size=3, strides=1, activation=None, name='conv1_2', reuse=reuse)
				.batch_nomalization(name='conv1_2bn')
				.conv(filters=64, kernel_size=3, strides=1, activation=None, name='conv1_3', reuse=reuse)
				.batch_nomalization(name='conv1_3bn')
				.max_pool(poll_size=3, name='max_pool'))

			x = self._res_bottleneck(self.terminals[0], 64, 128, name='conv2_1')
			x = self._res_bottleneck(x, 128, 128, name='conv2_2')
			x = self._res_bottleneck(x, 128, 128, name='conv2_3')
			x = self._res_bottleneck(x, 128, 256, strides=2, name='conv3_1')

			self.reservoir['conv3_1'] = x + 0.0

			return 0

	def _pyramid_pool(self, inputs, name, reuse=tf.AUTO_REUSE):

		size = tf.shape(inputs)[1:3]

		pool1 = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
		pool1 = tf.image.resize_bilinear(pool1, size=size, align_corners=True)

		part = list()
		h_w = tf.div(size, tf.constant(2))
		h, w = h_w[0], h_w[1]
		for ht in range(2):
			for wd in range(2):
				id_h1, id_h2 = h * ht, h * (ht + 1)
				id_w1, id_w2 = w * wd, w * (wd + 1)
				part.append(tf.reduce_mean(inputs[:, id_h1:id_h2, id_w1:id_w2, :], axis=[1, 2], keepdims=True))

		pool2 = tf.concat(part[:2], axis=2)
		for index in range(1, 2):
			pos = index * 2
			row = tf.concat(part[pos:pos + 2], axis=2)
			pool2 = tf.concat([pool2, row], axis=1)
		pool2 = tf.image.resize_bilinear(pool2, size=size, align_corners=True)

		part = list()
		h_w = tf.div(size, tf.constant(3))
		h, w = h_w[0], h_w[1]
		for ht in range(3):
			for wd in range(3):
				id_h1, id_h2 = h * ht, h * (ht + 1)
				id_w1, id_w2 = w * wd, w * (wd + 1)
				part.append(tf.reduce_mean(inputs[:, id_h1:id_h2, id_w1:id_w2, :], axis=[1, 2], keepdims=True))

		pool3 = tf.concat(part[:3], axis=2)
		for index in range(1, 3):
			pos = index * 3
			row = tf.concat(part[pos:pos + 3], axis=2)
			pool3 = tf.concat([pool3, row], axis=1)
		pool3 = tf.image.resize_bilinear(pool3, size=size, align_corners=True)

		part = list()
		h_w = tf.div(size, tf.constant(6))
		h, w = h_w[0], h_w[1]
		for ht in range(6):
			for wd in range(6):
				id_h1, id_h2 = h * ht, h * (ht + 1)
				id_w1, id_w2 = w * wd, w * (wd + 1)
				part.append(tf.reduce_mean(inputs[:, id_h1:id_h2, id_w1:id_w2, :], axis=[1, 2], keepdims=True))

		pool6 = tf.concat(part[:6], axis=2)
		for index in range(1, 6):
			pos = index * 6
			row = tf.concat(part[pos:pos + 6], axis=2)
			pool6 = tf.concat([pool6, row], axis=1)
		pool6 = tf.image.resize_bilinear(pool6, size=size, align_corners=True)

		with tf.variable_scope(name, reuse=reuse):
			out = tf.add_n([inputs, pool6, pool3, pool2, pool1])

			(self.feed(out)
			 	.conv(filters=256, kernel_size=1, strides=1, activation=None, name='1x1Pool')
			 	.batch_normalization(name='1x1Poolbn'))

			return self.terminals[0] + 0.0

	def _res_bottleneck_d(self, inputs, ch_in, ch_out, rate=2, strides=(1, 1, 1, 1), name=None, reuse=tf.AUTO_REUSE):

		need_tr = (ch_in != ch_out) or (strides != (1, 1, 1, 1))

		if need_tr:
			scope = name + '_sc'

			with tf.variable_scope(scope, reuse=reuse):

				(self.feed(inputs)
					.conv_nn(
						filters=(1, 1, ch_in, ch_out),
						strides=strides,
						activation=None,
						name='ch_modifier',
						reuse=reuse
					)
					.batch_normalization(name='bn', activation=None))

				sc_out = self.terminals[0] + 0.0
		else:
			sc_out = inputs + 0.0

		scope = name + '_mb'
		depth4 = ch_out // 4
		with tf.variable_scope(scope):
			(self.feed(inputs)
				.conv_nn(filters=(1, 1, ch_in, depth4), strides=strides, activation=None, name='1x1_1', reuse=reuse)
				.batch_normalization(name='1x1_1bn')
			 	.d_conv(filters=(3, 3, depth4, depth4), rate=rate, activation=None, name='d3x3_2', reuse=reuse)
				.batch_normalization(name='3x3_2bn')
				.conv_nn(filters=(1, 1, depth4, ch_out), name='1x1_3', activation=None, reuse=reuse)
				.batch_normalization(name='1x1_3bn', activation=None))

			mb_out = self.terminals[0] + 0.0

		with tf.variable_scope(name):
			output = sc_out + mb_out
			self.feed(output).activator()

			return self.terminals[0]

	def _branch4(self, inputs, name, reuse=False):

		new_size = tf.shape(inputs)[1:3] // 2

		with tf.variable_scope(name):
			self.feed(inputs).resize_bilinear(size=new_size, name='conv3_1_reduce')

			x = self._res_bottleneck(self.terminals[0], 256, 256, name='conv3_2')

			x = self._res_bottleneck(x, 256, 256, name='conv3_3')
			x = self._res_bottleneck(x, 256, 256, name='conv3_3')

			x = self._res_bottleneck_d(x, 256, 512, name='conv4_1')
			x = self._res_bottleneck_d(x, 512, 512, name='conv4_2')
			x = self._res_bottleneck_d(x, 512, 512, name='conv4_3')
			x = self._res_bottleneck_d(x, 512, 512, name='conv4_4')
			x = self._res_bottleneck_d(x, 512, 512, name='conv4_5')
			x = self._res_bottleneck_d(x, 512, 512, name='conv4_6')

			x = self._res_bottleneck_d(x, 512, 1024, rate=4, name='conv5_1')
			x = self._res_bottleneck_d(x, 1024, 1024, rate=4, name='conv5_2')
			x = self._res_bottleneck_d(x, 1024, 1024, rate=4, name='conv5_3')

			x = self._pyramid_pool(x, name='PyPool')  # returns 256 channels output

			self.reservoir['conv5_3'] = x + 0.0

			return 0

	def _branch1(self, inputs, name, reuse=False):

		with tf.variable_scope(name):
			(self.feed(inputs)
				.conv(filters=32, kernel_size=3, strides=2, activation=None, name='3x3_1', reuse=reuse)
			 	.batch_normalization(name='3x3_1bn')
			 	.conv(filters=32, kernel_size=3, strides=2, activation=None, name='3x3_2', reuse=reuse)
			 	.batch_normalization(name='3x3_2bn')
			 	.conv(filters=64, kernel_size=3, strides=2, activation=None, name='3x3_3', reuse=reuse)
			 	.batch_normalization(name='3x3_3bn'))

			self.reservoir['conv3'] = self.terminals[0] + 0.0

			return 0

	def _fusion_module(self, small_tensor, large_tensor, s_ch, l_ch, name, reuse=tf.AUTO_REUSE):

		large_size = tf.shape(large_tensor)[1:3]

		with tf.variable_scope(name, reuse=reuse):

			self.feed(small_tensor).resize_bilinear(large_size, name='interp')

			self.reservoir[name+'_out'] = self.terminals[0] + 0.0

			(self.conv_nn(filters=(3, 3, s_ch, 128), rate=2, activation=None, name='3x3', reuse=reuse)
				.batch_normalization(activation=None, name='3x3bn'))

			f_small = self.terminals[0] + 0.0

			(self.feed(large_tensor)
			 	.conv_nn(filters=(1, 1, l_ch, 128), activation=None, name='1x1', reuse=reuse)
				.batch_normalization(activation=None, name='1x1bn'))

			f_used = tf.add(f_small, self.terminals[0])

			self.feed(f_used).activator(name='activation')

			return self.terminals[0] + 0.0
