import argparse
import os

import tensorflow as tf

import network
import utils

FLAGS = None


def arg_process():

    parser = argparse.ArgumentParser('Implementation for MNIST handwritten digits 2019')

    parser.add_argument(
        '--data_path',
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help='The directory where the MNIST images were located',
        required=False,
    )
    parser.add_argument(
        '--img_name',
        type=str,
        default='train-images.idx3-ubyte',
        help='The file name for MNIST image',
        required=False,
    )
    parser.add_argument(
        '--label_name',
        type=str,
        default='train-labels.idx1-ubyte',
        help='The file name for MNIST labels',
        required=False,
    )
    parser.add_argument(
        '--num_epoch',
        type=int,
        default=10,
        help='Parameter for epoch',
        required=False,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='parameter for batch size',
        required=False,
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='parameter for learning rate',
        required=False,
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=2,
        help='parameter for layers',
    )
    parser.add_argument(
        '--num_nodes',
        nargs='+',
        type=int,
        default=[256, 256],
        help='parameter for nodes',
        required=False,
    )

    args, unknown = parser.parse_known_args()
    return args, unknown


def main():

    print('Starting MNIST Digit Recognition Learning')

    mnist_data = utils.mnist_data(FLAGS=FLAGS)
    net = network.MNIST(FLAGS=FLAGS)

    train_op, cost = net.optimizer()

    per_epoch = mnist_data.image_size // FLAGS.batch_size

    for index in range(FLAGS.num_epoch):
        mnist_data.shuffle()
        sum_cost = 0
        for step in range(per_epoch):
            digit, label = mnist_data.next_batch()
            feed_dict = {net.digit: digit, net.label: label}
            _, cost = net.sess.run([train_op, cost], feed_dict=feed_dict)
            sum_cost += cost
        mean_cost = sum_cost / float(per_epoch)
        print(f'Learning at f{index} epoch, Cost of {mean_cost:1.4f}')


if __name__ == '__main__':

    FLAGS, unparsed = arg_process()
    tf.app.run()
