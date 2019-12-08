import os

import tensorflow as tf
import tensorflow.bitwise as tw

from network import Network


class ICNet(Network):

	def __init__(self, cfg, train_reader, eval_reader):

		self.cfg = cfg
		self.mode = cfg.mode
		self.num_classes = cfg.param['num_classes']
		self.ignore_label = cfg.param['ignore_label']
		self.loss_weight = (cfg.LAMBDA1, cfg.LAMBDA2, cfg.LAMBDA3)
		self.reservoir = dict()
		self.losses = None
		self.start_epoch = 0

		self.sum_loss = tf.placeholder(dtype=tf.float32, shape=(5,))
		self.sum_acc = tf.placeholder(dtype=tf.float32, shape=(3,))

		self.eps = tf.constant(1e-5)

		self.train_reader = train_reader.dataset
		self.eval_reader = eval_reader.dataset

		self.handle = tf.placeholder(tf.string, shape=[])
		self.iterator = tf.data.Iterator.from_string_handle(self.handle, self.train_reader.output_types)

		self.images, self.labels = self.iterator.get_next()
		self.images.set_shape([None, None, None, 3])
		self.labels.set_shape([None, None, None, 1])

		size = tf.div(
			tf.add(tf.shape(self.images)[1:3], tf.constant(1)),
			tf.constant(2),
		)

		self.images2 = tf.image.resize_bilinear(self.images, size=size, align_corners=True)

		super().__init__()

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
				.batch_normalization(name='conv1_1bn')
				.conv(filters=32, kernel_size=3, strides=1, activation=None, name='conv1_2', reuse=reuse)
				.batch_normalization(name='conv1_2bn')
				.conv(filters=64, kernel_size=3, strides=1, activation=None, name='conv1_3', reuse=reuse)
				.batch_normalization(name='conv1_3bn')
				.max_pool(pool_size=3, name='max_pool'))

			x = self._res_bottleneck(self.terminals[0], 64, 128, name='conv2_1')
			x = self._res_bottleneck(x, 128, 128, name='conv2_2')
			x = self._res_bottleneck(x, 128, 128, name='conv2_3')
			x = self._res_bottleneck(x, 128, 256, strides=(1, 2, 2, 1), name='conv3_1')

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
			x = self._res_bottleneck(x, 256, 256, name='conv3_4')

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

	def _branch1(self, inputs, name, reuse=tf.AUTO_REUSE):

		with tf.variable_scope(name):
			(self.feed(inputs)
				.conv(filters=32, kernel_size=3, strides=2, activation=None, name='3x3_1')
			 	.batch_normalization(name='3x3_1bn')
			 	.conv(filters=32, kernel_size=3, strides=2, activation=None, name='3x3_2')
			 	.batch_normalization(name='3x3_2bn')
			 	.conv(filters=64, kernel_size=3, strides=2, activation=None, name='3x3_3')
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

	def _tail(self, inputs, name, reuse=tf.AUTO_REUSE):

		size = tf.multiply(tf.shape(inputs)[1:3], tf.constant(2))

		with tf.variable_scope(name, reuse=reuse):

			self.feed(inputs).resize_bilinear(size, name='interp')

			self.reservoir[name+'_out'] = self.terminals[0] + 0.0

	def _get_mask(self, gt, num_classes, ignore_label):

		class_mask = tf.less_equal(gt, num_classes-1)
		not_ignore_mask = tf.not_equal(gt, ignore_label)
		mask = tf.logical_and(class_mask, not_ignore_mask)
		indices = tf.squeeze(tf.where(mask), 1)

		return indices

	def _loss(self, name, reuse=tf.AUTO_REUSE):
		# prediction
		tensors = (self.reservoir['sub4_out'], self.reservoir['sub2_out'], self.reservoir['sub1_out'])

		losses = list()
		predictions = list()
		labels = list()

		with tf.variable_scope(name, reuse=reuse):
			for index in range(len(tensors)):
				(self.feed(tensors[index])
				 	.conv(
						filters=self.num_classes,
						kernel_size=1,
						strides=1,
						use_bias=True,
						activation=None,
						name='cls_{}'.format(index)
					)
				)
				predictions.append(self.terminals[0] + 0.0)

			# resizing labels
			for index in range(len(predictions)):
				size = tf.shape(predictions[index])[1:3]
				(self.feed(self.labels)
				 	.resize_nn(size, name='interp_{}'.format(index)))
				labels.append(tf.squeeze(self.terminals[0], axis=[3]))

			self.reservoir['digits_out'] = predictions[-1]

			# ignore label process and loss calc.
			t_loss = 0.0
			for index in range(len(labels)):
				gt = tf.reshape(labels[index], (-1,))
				indices = self._get_mask(gt, self.num_classes, self.ignore_label)  # get label pos. with not ignore label
				gt = tf.cast(tf.gather(gt, indices), tf.int32)  # only not-ignore_label ground-truth

				pred = tf.reshape(predictions[index], (-1, self.num_classes))
				pred = tf.gather(pred, indices)  # only not-ignore_label prediction

				loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt))
				t_loss += self.loss_weight[index] * loss
				losses.append(loss)

			losses.append(t_loss)

			return losses

	def _confusion_matrix(self, pred, gt):
		merged_maps = tw.bitwise_or(tw.left_shift(gt, 8), pred)
		hist = tf.bincount(tf.reshape(merged_maps, (-1,)))
		nonzero = tf.squeeze(tf.cast(tf.where(tf.not_equal(hist, 0)), dtype=tf.int32))

		pred, gt = tw.bitwise_and(nonzero, 255), tw.right_shift(nonzero, 8)

		class_cnt = self.num_classes
		indices = class_cnt * gt + pred
		shape = class_cnt * class_cnt

		conf_matrix = tf.sparse_to_dense(indices, (shape,), tf.gather(hist, nonzero), 0)

		return tf.cast(tf.reshape(conf_matrix, (class_cnt, class_cnt)), dtype=tf.float32)

	def _mIoU(self, pred, gt):

		conf_mat = self._confusion_matrix(pred, gt)  # 11-person, 12-rider

		row_sum = tf.squeeze(tf.reduce_sum(conf_mat, axis=1))
		col_sum = tf.squeeze(tf.reduce_sum(conf_mat, axis=0))
		gt_class_num = tf.cast(tf.count_nonzero(row_sum), dtype=tf.float32)
		diag = tf.squeeze(tf.diag_part(conf_mat))

		union = row_sum + col_sum - diag + self.eps
		mIoU = tf.truediv(tf.reduce_sum(tf.truediv(diag, union)), gt_class_num)

		return mIoU, conf_mat

	def _inference(self):

		pred = self.reservoir['digits_out']

		pred = tf.reshape(pred, (-1,))
		labels = tf.reshape(self.labels, (-1,))  # flattening

		mask = tf.not_equal(labels, self.ignore_label)
		indices = tf.squeeze(tf.where(mask), 1)

		gt = tf.cast(tf.gather(labels, indices), tf.int32)
		pred = tf.cast(tf.gather(pred, indices), tf.int32)

		mIoU, conf_mat = self._mIoU(pred, gt)
		# person-11, rider-12
		union = tf.reduce_sum(conf_mat[11, :])
		personIoU = tf.cond(
			tf.equal(union, 0),
			lambda: 0.0,
			lambda: tf.truediv(
				conf_mat[11, 11],
				union + tf.reduce_sum(conf_mat[:, 11]) - conf_mat[11, 11] + self.eps,
			),
		)
		union = tf.reduce_sum(conf_mat[12, :])
		riderIoU = tf.cond(
			tf.equal(union, 0),
			lambda: 0.0,
			lambda: tf.truediv(
				conf_mat[12, 12],
				union + tf.reduce_sum(conf_mat[:, 12]) - conf_mat[12, 12] + self.eps,
			)
		)

		return mIoU, personIoU, riderIoU

	def optimizer(self):
		# weight-decay, learning-rate control, optimizer selection with bn training
		if self.cfg.WEIGHT_DECAY != 0.0:
			l2_weight = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() if ('bn' not in var.name)])
			loss_to_opt = self.losses[-1] + self.cfg.WEIGHT_DECAY * l2_weight
			self.losses.append(loss_to_opt)

		if self.cfg.LR_CONTROL is 'poly':
			base_lr = tf.constant(self.cfg.LEARNING_RATE)
			learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - self.step_ph / self.cfg.MAX_ITERATION), self.cfg.POWER))
		else:
			learning_rate = self.cfg.LEARNING_RATE

		opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
		train_op = opt.minimize(self.losses[-1])

		if self.cfg.BN_LEARN:
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			train_op = tf.group([train_op, update_ops])

		# create session
		gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC', visible_device_list='0')
		config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
		self.sess = tf.Session(config=config)
		self.sess.run(tf.global_variables_initializer())

		self.train_handle = self.sess.run(self.train_reader.string_handle())
		self.eval_handle = self.sess.run(self.eval_reader.string_handle())

		# check-point processing
		self.saver = tf.train.Saver()
		ckpt_loc = self.cfg.ckpt_dir
		self.ckpt_name = os.path.join(ckpt_loc, 'ICnetModel')

		ckpt = tf.train.get_checkpoint_state(ckpt_loc)
		if ckpt and ckpt.model_checkpoint_path:
			import re
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)

			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.start_step = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
			print("---------------------------------------------------------")
			print(" Success to load checkpoint - {}".format(ckpt_name))
			print(" Session starts at step - {}".format(self.start_step))
			print("---------------------------------------------------------")
		else:
			if not os.path.exists(ckpt_loc):
				os.makedirs(ckpt_loc)
			self.start_step = 0
			print("**********************************************************")
			print("  [*] Failed to find a checkpoint - Start from the first")
			print(" Session starts at step - {}".format(self.start_step))
			print("**********************************************************")

		# summary and summary Writer
		_ = tf.summary.scalar("Total_Loss", self.sum_loss[3])
		_ = tf.summary.scalar("Branch-4 Loss", self.sum_loss[0])
		_ = tf.summary.scalar("Branch-2 Loss", self.sum_loss[1])
		_ = tf.summary.scalar("Branch-1 Loss", self.sum_loss[2])
		_ = tf.summary.scalar("Mean IoU", self.sum_acc[0])
		_ = tf.summary.scalar("Person IoU", self.sum_acc[1])
		_ = tf.summary.scalar("Rider IoU", self.sum_acc[2])

		self.summaries = tf.summary.merge_all()

		self.writer = tf.summary.FileWriter(self.cfg.log_dir, self.sess.graph)

		# inference and evaluation
		IoUs = self._inference()
		Images = (self.images, self.labels)

		return train_op, self.losses, self.summaries, self.reservoir['digits_out'], IoUs, Images

	def save(self, global_step):
		self.saver.save(self.sess, self.ckpt_name, global_step)
		print('The checkpoint has been created, step: {}'.format(global_step))

	def _build(self, is_training=True):
		self._branch2(self.images2, name='br2')
		self._branch4(self.reservoir['conv3_1'], name='br4')
		self._branch1(self.images, name='br1')

		x = self._fusion_module(
			self.reservoir['conv5_3'],
			self.reservoir['conv3_1'],
			256,
			256,
			name='sub4',
		)
		x = self._fusion_module(
			x,
			self.reservoir['conv3'],
			128,
			64,
			name='sub2',
		)

		self._tail(x, name='sub1')

		self.losses = self._loss(name='loss')
