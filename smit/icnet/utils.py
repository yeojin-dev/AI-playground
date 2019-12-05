import multiprocessing as mp
import os

import numpy as np
import tensorflow as tf


class Config:
	CITYSCAPES_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cityscapes')
	CITYSCAPES_TRAIN_LIST = os.path.join(CITYSCAPES_DATA_DIR, 'trainAttribute.txt')
	CITYSCAPES_EVAL_LIST = os.path.join(CITYSCAPES_DATA_DIR, 'valAttribute.txt')

	IMG_MEAN = np.array((124.68, 116.779, 103.939), dtype=np.float32)

	cityscapes_param = {
		'name': 'cityscapes',
		'num_classes': 19,
		'ignore_label': 255,
		'eval_size': [1024, 2048],
		'eval_steps': 500,
		'eval_list': CITYSCAPES_EVAL_LIST,
		'train_list': CITYSCAPES_TRAIN_LIST,
		'data_dir': CITYSCAPES_DATA_DIR,
	}

	dataset_param = {
		'name': 'YOUR_OWN_DATASET',
		'num_classes': 0,
		'ignore_label': 0,
		'eval_size': [0, 0],
		'eval_steps': 0,
		'eval_list': '/PATH/TO/YOUR_EVAL_LIST',
		'train_dir': '/PATH/TO/YOUR_TRAIN_LIST',
		'data_dir': '/PATH/TO/YOUR_DATA_DIR',
	}

	TRAIN_SIZE = [720, 720]
	TRAIN_EPOCHS = 500  # number of epochs for weight decay
	WEIGHT_DECAY = 0.0001  # 0.0 for turn off
	LR_CONTROL = 'None'  # ploy, linear, exponential, None
	LEARNING_RATE = 0.1
	POWER = 0.9
	MAX_ITERATION = 30000
	BN_LEARN = True

	# loss function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss + weight_decay
	LAMBDA1 = 0.4
	LAMBDA2 = 0.4
	LAMBDA3 = 1.0

	BATCH_SIZE = 8
	BUFFER_SIZE = 2
	N_WORKERS = min(mp.cpu_count(), BATCH_SIZE)

	def __init__(self, args):
		print('SETUP CONFIGURATIONS...')

		if args.dataset == 'cityscapes':
			self.param = self.cityscapes_param
		else:
			raise NotImplementedError('N/A, except for cityscapes dataset')

		self.dataset = args.dataset

		if args.use_aug:
			self.random_scale, self.random_mirror = True, True
		else:
			self.random_scale, self.random_mirror = False, False

		self.mode = args.mode

		self.ckpt_dir = args.ckpt_dir
		self.log_dir = args.log_dir

	def display(self):
		print('\nconfigurations:')
		for configuration in dir(self):
			configuration_value = getattr(self, configuration)
			if (not configuration.startswith('__')
					and not callable(configuration_value)
					and not isinstance(configuration_value, dict)):
				print('{:30} {}'.format(configuration, configuration_value))
			if configuration == 'param':
				print(configuration)
				for k, v in getattr(self, configuration).items():
					print('\t{:27} {}'.format(k, v))
		print('\n')


def _read_attribute_list(data_dir, data_list):
	with open(data_list, 'r') as f:

		att_file = list()

		for line in f:
			try:
				attributes = line[:-1].split(' ')
			except ValueError(attributes):
				attributes = line[:-1].split(' ')

			if not tf.gfile.Exists(attributes[0]):
				raise ValueError(f'Failed to find file: {attributes[0]}')

			if not tf.gfile.Exists(attributes[1]):
				raise ValueError(f'Failed to find file: {attributes[1]}')

			att_file.append(attributes)

	return att_file


def _image_scaling(img, label):
	scale = tf.random_uniform((1,), minval=0.5, maxval=2.0, dtype=tf.float32, seed=None)
	new_shape = tf.squeeze(tf.to_int32(tf.to_float(tf.shape(img)[:-1] * scale)))

	img = tf.image.resize_images(img, new_shape)
	label = tf.squeeze(
		tf.image.resize_nearest_neighbor(tf.expand_dims(label, axis=0), new_shape),
		axis=0,
	)

	return img, label


def _image_cropping(image, label, cfg):
	crop_h, crop_w = cfg.TRAIN_SIZE
	ignore_label = cfg.param['ignore_label']

	label = tf.cast(label, dtype=tf.float32)
	label -= ignore_label
	combined = tf.concat(axis=1, values=[image, label])
	image_shape = tf.shape(image)

	trg_height = tf.maximum(crop_h, image_shape[0])
	off_height = tf.to_int32((trg_height - image_shape[0]) / 2)
	trg_width = tf.maximum(crop_w, image_shape[1])
	off_width = tf.to_int32((trg_width - image_shape[1]) / 2)

	combined_pad = tf.image.pad_to_bounding_box(
		combined,
		offset_height=off_height,
		offset_width=off_width,
		target_height=trg_height,
		target_width=trg_width,
	)

	combined_crop = tf.image.random_crop(combined_pad, [crop_h, crop_w, 4])

	img_crop = combined_crop[:, :, :-1]

	label_crop = tf.expend_dims(combined_crop[:, :, :-1], axis=2)
	label_crop += ignore_label
	label_crop = tf.cast(label_crop, dtype=tf.uint8)

	img_crop.set_shape((crop_h, crop_w, 3))
	label_crop.set_shape((crop_h, crop_w, 1))

	return img_crop, label_crop


def _train_process(attributes, cfg):
	image_filename = attributes[0]
	label_filename = attributes[1]

	img_contents = tf.read_file(image_filename)
	label_contents = tf.read_file(label_filename)

	img = tf.cast(tf.image.decode_image(img_contents, channels=3), dtype=tf.float32)
	img -= cfg.IMG_MEAN
	label = tf.cast(tf.image.decode_image(label_contents, channels=1), dtype=tf.float32)

	img.set_shape((cfg.param['eval_size'][0], cfg.param['eval_size'][1], 3))
	label.set_shape((cfg.param['eval_size'][0], cfg.param['eval_size'][1], 1))

	return img, label


def _eval_preprocess(attributes, cfg):
	image_filename = attributes[0]
	label_filename = attributes[1]

	img_contents = tf.read_file(image_filename)
	label_contents = tf.read_file(label_filename)

	img = tf.cast(tf.image.decode_png(img_contents, channels=3), dtype=tf.float32)
	img -= cfg.IMG_MEAN
	label = tf.cast(tf.image.decode_png(label_contents, channels=1), dtype=tf.float32)

	img.set_shape((cfg.param['eval_size'][0], cfg.param['eval_size'][1], 3))
	label.set_shape((cfg.param['eval_size'][0], cfg.param['eval_size'][1], 1))

	return img, label


class ImageReader:

	def __init__(self, cfg):

		mode = cfg.mode

		self.attribute_list = _read_attribute_list(cfg.param['data_dir'], cfg.param[mode + '_list'])
		print('read attribute file - {}'.format(cfg.param[mode + '_list']))

		if mode == 'train':
			self.dataset = self._train_dataset(cfg)
		elif mode == 'eval':
			self.dataset = self._eval_dataset(cfg)
		else:
			raise NotImplementedError('N/A, other than train or eval')

	def _train_dataset(self, cfg):

		batch_size = cfg.BATCH_SIZE
		batch_size_n = batch_size * cfg.BUFFER_SIZE

		dataset = (
			tf.data.Dataset.from_tensor_slices(self.attribute_list)
				.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=len(self.attribute_list)))
				.map(lambda x: _train_process(x, cfg), num_parallel_calls=cfg.N_WORKERS)
				.batch(batch_size, drop_remainder=True)
				.prefetch(buffer_size=batch_size_n)
				.make_one_shot_iterator()
		)

		return dataset

	def _eval_dataset(self, cfg):

		batch_size = cfg.BATCH_SIZE
		batch_size_n = batch_size * cfg.BUFFER_SIZE

		dataset = (
			tf.data.Dataset.from_tensor_slices(self.attribute_list)
				.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=len(self.attribute_list)))
				.map(lambda x: _eval_preprocess(x, cfg), num_parallel_calls=cfg.N_WORKERS)
				.batch(batch_size, drop_remainder=True)
				.prefetch(buffer_size=batch_size_n)
				.make_one_shot_iterator()
		)

		return dataset
