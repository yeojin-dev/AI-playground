import argparse
import time

from .model import ICNet
from .utils import Config, ImageReader


def get_arguments():
	parser = argparse.ArgumentParser(description='Implementation for ICNet Semantic Segmentation')

	parser.add_argument(
		'--data_set',
		type=str,
		default='cityscapes',
		help='which dataset to trained with',
		choices=['cityscapes', 'ade20k', 'others'],
		required=False,
	)

	parser.add_argument(
		'--mode',
		type=str,
		default='train',
		help='which mode for ICNet',
		choices=['train', 'eval'],
		required=False
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

	return parser.parse_args()


class TrainConfig(Config):
	def __init__(self, arguments):
		Config.__init__(self, arguments)

	LAMBDA1 = 0.4
	LAMBDA2 = 0.4
	LAMBDA3 = 1.0

	BATCH_SIZE = 8
	LEARNING_RATE = 0.001

	SAVE_PERIOD = 500


def main():
	args = get_arguments()

	cfg = TrainConfig(args)
	cfg.display()

	train_reader = ImageReader(cfg)
	train_net = ICNet(cfg, train_reader)

	train_op, losses, summaries = train_net.optimizer()

	global_step = train_net.start_step
	epoch_step = int(len(train_reader.attribute_list) / cfg.BATCH_SIZE + 0.5)
	start_epoch = int(global_step / epoch_step)
	start_batch = global_step % epoch_step

	for epochs in range(start_epoch, cfg.TRAIN_EPOCHS):
		for steps in range(start_batch, epoch_step):
			start_time = time.time()

			feed_dict = {train_net.step_ph: global_step}
			_, loss_ptr, summary_ptr = train_net.sess.run([train_op, losses, summaries], feed_dict=feed_dict)
			train_net.writer.add_summary(summary_ptr, global_step)

			if global_step % cfg.SAVE_PERIOD == 0:
				train_net.save(global_step)

			global_step += 1

			duration = time.time() - start_time
			msg = (
				f'''step {global_step} \t total loss = {loss_ptr[3]:.3f}, sub4 = {loss_ptr[0]:.3f},'''
				f'''sub24 = {loss_ptr[1]:.3f}, sub124 = {loss_ptr[2]:.3f}, val_loss: {loss_ptr[4]:.3f}'''
				f'''({duration:.3f} sec/step)'''
			)
			print(msg)


if __name__ == '__main__':
	main()
