import argparse
import time

import numpy as np

from model import ICNet
from utils import Config, ImageReader, Visualizer


def get_arguments():
	parser = argparse.ArgumentParser(description='Implementation for ICNet Semantic Segmentation')

	parser.add_argument(
		'--dataset',
		type=str,
		default='cityscapes',
		help='which dataset to trained with',
		choices=['cityscapes', 'ade20k', 'others'],
		required=False,
	)

	parser.add_argument(
		'--use_aug',
		default=True,
		help="whether to use data augmentation for inputs during the training.",
		required=False,
	)

	parser.add_argument(
		'--log_dir',
		type=str,
		default='./logs',
		help='the directory where the Training logs are located',
		required=False,
	)

	parser.add_argument(
		'--ckpt_dir',
		type=str,
		default='./ckpt',
		help='the directory where the checkpoint files are located',
		required=False,
	)

	parser.add_argument(
		'--res_dir',
		type=str,
		default='./res',
		help='the directory where the result images are located',
		required=False,
	)

	return parser.parse_args()


class TrainConfig(Config):
	def __init__(self, arguments):
		Config.__init__(self, arguments)
		self.mode = 'train'


class EvalConfig(Config):
	def __init__(self, arguments):
		Config.__init__(self, arguments)
		self.mode = 'eval'


def main():
	args = get_arguments()

	print('SETUP TrainConfig...')
	train_cfg = TrainConfig(args)
	train_cfg.display()

	# print('SETUP EvalConfig...')
	eval_cfg = EvalConfig(args)
	# eval_cfg.display()

	train_reader = ImageReader(train_cfg)
	eval_reader = ImageReader(eval_cfg)

	train_net = ICNet(train_cfg, train_reader, eval_reader)

	_train_op, _losses, _summaries, _Preds, _IoUs, _Images = train_net.optimizer()

	vis = Visualizer(eval_cfg)

	global_step = train_net.start_step
	epoch_step = int(len(train_reader.attribute_list) / train_cfg.BATCH_SIZE + 0.5)
	start_epoch = int(global_step / epoch_step)
	save_step = int(epoch_step * train_cfg.SAVE_PERIOD)

	all_steps = int(len(eval_reader.attribute_list) / (eval_cfg.BATCH_SIZE))
	g_eval_step = 0

	train_fd = {train_net.handle: train_net.train_handle}
	eval_fd = {train_net.handle: train_net.eval_handle}

	for epochs in range(start_epoch, train_cfg.TRAIN_EPOCHS):

		epoch_loss = None
		start_batch = global_step % epoch_step

		print(f'Start batch - {start_batch}')
		print(f'Epoch step - {epoch_step}')

		for steps in range(start_batch, epoch_step):
			start_time = time.time()

			_, losses = train_net.sess.run([_train_op, _losses], feed_dict=train_fd)

			if epoch_loss is None:
				epoch_loss = np.array(losses)
			else:
				epoch_loss += np.array(losses)

			if global_step % save_step == 0:
				train_net.save(global_step)

			global_step += 1

			duration = time.time() - start_time
			msg = (
				f'''step {global_step} \t total loss = {losses[3]:.3f}, sub4 = {losses[0]:.3f}, '''
				f'''sub24 = {losses[1]:.3f}, sub124 = {losses[2]:.3f}, val_loss: {losses[4]:.3f}'''
				f'''({duration:.3f} sec/step)'''
			)
			print(msg)

		epoch_loss /= (epoch_step - start_batch)
		accuracy = None

		for steps in range(all_steps - 1):
			start_time = time.time()

			IoUs = train_net.sess.run(_IoUs, feed_dict=eval_fd)

			if accuracy is None:
				accuracy = np.array(IoUs)
			else:
				accuracy += np.array(IoUs)

			g_eval_step += 1

			duration = time.time() - start_time
			msg = (
				f'''step {steps} \t mean_IoU = {IoUs[0]:.3f}, Person_IoU = {IoUs[1]:.3f}, '''
				f'''Rider_IoU = {IoUs[2]:.3f}, ({duration:.3f} sec/step)'''
			)
			print(msg)

		IoUs, Preds, Images = train_net.sess.run([_IoUs, _Preds, _Images], feed_dict=eval_fd)
		accuracy += np.array(IoUs)
		accuracy /= all_steps

		g_eval_step += 1

		vis.save_and_show(Images, Preds, g_eval_step)

		feed_dict = {train_net.sum_loss: epoch_loss, train_net.sum_acc: accuracy}
		summaries = train_net.sess.run(_summaries, feed_dict=feed_dict)
		train_net.writer.add_summary(summaries, epochs)


if __name__ == '__main__':
	main()
