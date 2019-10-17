import os
import sys

import numpy as np


def read_file(filename):
    return open(filename, 'rb')


class mnist_data:

    def __init__(self, FLAGS):
        # seek(offset, [where]), where(0:start, 1:current, 2:end)        
        self.image_file = read_file(os.path.join(FLAGS.data_path, FLAGS.img_name))
        self.image_size = int((os.path.getsize(self.image_file.name) - 16.0) / 784.0)

        self.eval_size = int(self.image_size * 0.1)
        self.image_size -= self.eval_size

        self.label_file = read_file(os.path.join(FLAGS.data_path, FLAGS.label_name))
        self.label_size = int((os.path.getsize(self.label_file.name) - 8.0))

        assert (self.image_size + self.eval_size) == self.label_size

        self.batch_size = FLAGS.batch_size
        self.image_index = np.arange(self.image_size)
        self.position = 0

        self.nw_type = FLAGS.nw_type

    def next_batch(self):
        images = list()
        labels = list()

        for _ in range(self.batch_size):
            one_hot = np.zeros(10)
            self.image_file.seek(16 + 784 * self.image_index[self.position])
            self.label_file.seek(8 + self.image_index[self.position])

            digit = np.fromfile(self.image_file, dtype=np.ubyte, count=784)

            if self.nw_type == 'dense':
                images.append(digit)
            elif self.nw_type == 'conv':
                digit = np.reshape(digit, (28, 28, 1))
                images.append(digit)
            else:
                raise NotImplementedError('not supported network type in next_batch.')

            one_hot[int(np.fromfile(self.label_file, dtype=np.ubyte, count=1))] = 1.0
            labels.append(one_hot)

            self.position += 1

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

    def shuffle(self):
        np.random.shuffle(self.image_index)
        self.position = 0

    def eval_data(self):
        images = list()
        labels = list()

        for index in range(self.eval_size):
            one_hot = np.zeros(10)
            self.image_file.seek(16 + 784 * (self.image_size + index), 0)
            self.label_file.seek(8 + (self.image_size + index), 0)

            if self.nw_type == 'dense':
                images.append(np.fromfile(self.image_file, dtype=np.ubyte, count=784))
            elif self.nw_type == 'conv':
                images.append(np.fromfile(self.image_file, dtype=np.ubyte, count=784).reshape((28, 28, 1)))
            else:
                sys.exit('not supported network type in eval_data.')

            one_hot[int(np.fromfile(self.label_file, dtype=np.ubyte, count=1))] = 1.0
            labels.append(one_hot)

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)
