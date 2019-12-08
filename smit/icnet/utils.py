import multiprocessing as mp
import os

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import tensorflow as tf


class Config:
	CITYSCAPES_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cityscapes')
	CITYSCAPES_TRAIN_LIST = os.path.join(CITYSCAPES_DATA_DIR, 'trainAttribute.txt')
	CITYSCAPES_EVAL_LIST = os.path.join(CITYSCAPES_DATA_DIR, 'valAttribute.txt')

	IMG_MEAN = np.array((124.68, 116.779, 103.939), dtype=np.float32)

	label_colors = [
		[128, 64, 128],  # 0 = road
		[244, 35, 232],  # 1 = sidewalk
		[70, 70, 70],  # 2 = building
		[102, 102, 156],  # 3 = wall
		[190, 153, 153],  # 4 = fence
		[153, 153, 153],  # 5 = pole
		[250, 170, 30],  # 6 = traffic light
		[220, 220, 0],  # 7 = traffic sign
		[107, 142, 35],  # 8 = vegetation
		[152, 251, 152],  # 9 = terrain
		[70, 130, 180],  # 10 = sky
		[220, 20, 60],  # 11 = person
		[255, 0, 0],  # 12 = rider
		[0, 0, 142],  # 13 = car
		[0, 0, 70],  # 14 = truck
		[0, 60, 100],  # 15 = bus
		[0, 80, 100],  # 16 = train
		[0, 0, 230],  # 17 = motocycle
		[119, 10, 32],  # 18 = bicycle
	]

	cityscapes_param = {
		'name': 'cityscapes',
		'num_classes': 19,
		'ignore_label': 255,
		'eval_size': [1024, 2048],
		'eval_steps': 500,
		'eval_list': CITYSCAPES_EVAL_LIST,
		'train_list': CITYSCAPES_TRAIN_LIST,
		'data_dir': CITYSCAPES_DATA_DIR,
		'label_colors': label_colors,
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
	SAVE_PERIOD = 2  # every N epochs
	WEIGHT_DECAY = 0.0001  # 0.0 for turn off
	LR_CONTROL = 'None'  # ploy, linear, exponential, None
	LEARNING_RATE = 0.001
	POWER = 0.9
	MAX_ITERATION = 30000
	BN_LEARN = True

	# loss function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss + weight_decay
	LAMBDA1 = 0.4
	LAMBDA2 = 0.4
	LAMBDA3 = 1.0

	BATCH_SIZE = 16
	BUFFER_SIZE = 4
	N_WORKERS = min(mp.cpu_count(), BATCH_SIZE)

	def __init__(self, args):

		if args.dataset == 'cityscapes':
			self.param = self.cityscapes_param
		else:
			raise NotImplementedError('N/A, except for cityscapes dataset')

		self.dataset = args.dataset

		if args.use_aug:
			self.random_scale, self.random_mirror = True, True
		else:
			self.random_scale, self.random_mirror = False, False

		self.ckpt_dir = args.ckpt_dir
		self.log_dir = args.log_dir
		self.res_dir = args.res_dir

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
	scale = tf.random_uniform([1], minval=0.5, maxval=2.0, dtype=tf.float32, seed=None)
	new_shape = tf.squeeze(tf.to_int32(tf.to_float(tf.shape(img)[:-1]) * scale))

	img = tf.image.resize_images(img, new_shape)
	label = tf.image.resize_nearest_neighbor(
		tf.expand_dims(label, axis=0),
		new_shape,
	)
	label = tf.squeeze(label, axis=0)

	return img, label


def _image_mirroring(img, label):
	prob = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]

	flip = lambda: (tf.image.flip_left_right(img), tf.image.flip_left_right(label))
	as_is = lambda: (img, label)

	img, label = tf.cond(tf.less_equal(prob, 0.5), flip, as_is)

	return img, label


def _image_cropping(image, label, cfg):
	crop_h, crop_w = cfg.TRAIN_SIZE
	ignore_label = cfg.param['ignore_label']

	label = tf.cast(label, dtype=tf.float32)
	label -= ignore_label
	combined = tf.concat(axis=2, values=[image, label])
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

	label_crop = tf.expand_dims(combined_crop[:, :, -1], axis=2)
	label_crop += ignore_label
	label_crop = tf.cast(label_crop, dtype=tf.uint8)

	# set static tensor input for easy tensor shape operation
	# img_crop.set_shape((crop_h, crop_w, 3))
	# label_crop.set_shape((crop_h, crop_w, 1))

	return img_crop, label_crop


def _train_process(attributes, cfg):
	image_filename = attributes[0]
	label_filename = attributes[1]

	img_contents = tf.read_file(image_filename)
	label_contents = tf.read_file(label_filename)

	img = tf.cast(tf.image.decode_png(img_contents, channels=3), dtype=tf.float32)
	img -= cfg.IMG_MEAN
	label = tf.cast(tf.image.decode_png(label_contents, channels=1), dtype=tf.float32)

	if cfg.random_mirror:
		img, label = _image_mirroring(img, label)
	if cfg.random_scale:
		img, label = _image_scaling(img, label)

	img, label = _image_cropping(img, label, cfg)

	return img, label


def _eval_preprocess(attributes, cfg):
	image_filename = attributes[0]
	label_filename = attributes[1]

	img_contents = tf.read_file(image_filename)
	label_contents = tf.read_file(label_filename)

	img = tf.cast(tf.image.decode_png(img_contents, channels=3), dtype=tf.float32)
	img -= cfg.IMG_MEAN
	label = tf.cast(tf.image.decode_png(label_contents, channels=1), dtype=tf.uint8)

	# set static tensor input for easy tensor shape operation
	# img.set_shape((cfg.param['eval_size'][0], cfg.param['eval_size'][1], 3))
	# label.set_shape((cfg.param['eval_size'][0], cfg.param['eval_size'][1], 1))

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


class Visualizer:

	def __init__(self, cfg):

		self.queue = mp.Queue()
		self.queue.put(0)

		self.res_dir = cfg.res_dir
		if not os.path.exists(self.res_dir):
			os.makedirs(self.res_dir)

		self.num_classes = cfg.param['num_classes']
		self.label_colors = cfg.param['label_colors']
		self.img_mean = cfg.IMG_MEAN

	def _save_image(self, image, ids):

		filename = os.path.join(self.res_dir, f'{ids}.png')
		img = np.clip(image, 0.0, 255.0).astype(np.uint8)
		Image.fromarray(img).save(filename)

	def _show_image(self, image, ids):

		item = self.queue.get()

		if item == 0:
			# First Run - Plot the image and queue the Axes object
			q_d = plt.imshow(image / 255.0)
			ax = plt.gca()

			ax.set_xticks([])
			ax.set_yticks([])

			ax.set_xlabel(f'Image of Id : {ids}')
			plt.ion()
			plt.show()
			plt.pause(0.0001)
			self.queue.put(q_d)
		else:
			# Use the queued Axes object to plot the image
			item.set_data(image / 255.0)

			ax = plt.gca()
			ax.set_xlabel(f'Image of Id : {ids}')

			plt.draw()
			plt.pause(0.0001)
			self.queue.put(item)

	def _save_and_show(self, images, preds, ids):

		b, h, w = np.shape(images[0])[0:-1]
		select = int((b - 1) * np.random.uniform())

		# select sample in batch
		img, gt, pred = images[0][select], images[1][select].astype(np.int32), preds[select].astype(np.int32)
		gt[gt == 255] = 19
		label_colors = np.append(self.label_colors, [[0, 0, 0]], axis=0)
		num_classes = 20

		index = np.reshape(pred, (-1,))
		one_hot = np.eye(self.num_classes)[index]
		pred_img = np.reshape(np.matmul(one_hot, self.label_colors), (h, w, 3))

		index = np.reshape(gt, (-1,))
		one_hot = np.eye(num_classes)[index]
		gt_img = np.reshape(np.matmul(one_hot, label_colors), (h, w, 3))

		img = np.clip((img + self.img_mean), 0.0, 255.0).astype(np.uint8)
		Img = Image.fromarray(img)

		pred_img = np.clip(pred_img, 0.0, 255.0).astype(np.uint8)
		Pred_img = Image.fromarray(pred_img)

		gt_img = np.clip(gt_img, 0.0, 255.0).astype(np.uint8)
		Gt_img = Image.fromarray(gt_img)

		p_image = Image.blend(Img, Pred_img, 0.3)
		g_image = Image.blend(Img, Gt_img, 0.3)

		filename = os.path.join(self.res_dir, f'img{ids}.png')
		Img.save(filename)
		filename = os.path.join(self.res_dir, f'prd{ids}.png')
		p_image.save(filename)
		filename = os.path.join(self.res_dir, f'gt{ids}.png')
		g_image.save(filename)

		self._show_image(np.array(p_image, np.float32), ids)

	def save_and_show(self, images, preds, ids):

		prc = mp.Process(target=self._save_and_show, args=(images, preds, ids))
		prc.daemon = True
		prc.start()

		prc.join()
