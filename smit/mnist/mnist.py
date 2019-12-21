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
        '--log_dir',
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help='the directory where the MNIST images were located',
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
        default=3,
        help='parameter for layers',
    )
    parser.add_argument(
        '--num_nodes',
        nargs='+',
        type=int,
        default=[128, 128, 128],
        help='parameter for nodes',
        required=False,
    )
    parser.add_argument(
        '--num_depths',
        nargs='+',
        type=int,
        default=[16, 32, 64],
        help='number of output channels for each conv layer',
        required=False,
    )
    parser.add_argument(
        '--kernel_size',
        nargs='+',
        type=int,
        default=[(5, 5), (5, 5), (5, 5)],
        help='size of kernels for each conv layer',
        required=False,
    )
    parser.add_argument(
        '--nw_type',
        type=str,
        default='conv',  # or 'dense'
        help='the type of network(dense of conv)',
        required=False,
    )
    args, unknown = parser.parse_known_args()
    return args, unknown


def main(_):

    print('Starting MNIST Digit Recognition Learning')

    mnist_data = utils.mnist_data(FLAGS=FLAGS)
    net = network.MNIST(FLAGS=FLAGS)

    _train_op, _cost, _summary = net.optimizer()

    per_epoch = mnist_data.image_size // FLAGS.batch_size

    for index in range(FLAGS.num_epoch):
        mnist_data.shuffle()
        sum_cost = 0

        # Training
        for step in range(per_epoch):
            digit, label = mnist_data.next_batch()
            feed_dict = {net.digit: digit, net.label: label}
            _, cost = net.sess.run([_train_op, _cost], feed_dict=feed_dict)
            sum_cost += cost
        mean_cost = sum_cost / float(per_epoch)
        print(f'Learning at {index} epoch, Cost of {mean_cost:1.4f}')
        net.save(net.sess, index)

        # Evaluation
        batch_x, batch_y = mnist_data.eval_data()
        feed_dict_evaluation = {net.digit: batch_x, net.label: batch_y}
        out = net.inference
        _accuracy = net.evaluate(out, batch_y)
        accuracy = net.sess.run(_accuracy, feed_dict=feed_dict_evaluation)

        sum_ptr = net.sess.run(_summary, feed_dict={net.epoch_cost: mean_cost, net.epoch_eval: accuracy})
        net.writer.add_summary(sum_ptr, index)

        print(f'Evaluation at {index} epoch, accuracy of {accuracy:1.4f}')


if __name__ == '__main__':

    FLAGS, unparsed = arg_process()
    tf.app.run()
