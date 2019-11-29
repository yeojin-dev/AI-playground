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
