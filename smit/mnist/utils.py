import os
import numpy as np


def read_file(filename):
    return open(filename, 'rb')


class mnist_data:

    def __init__(self, FLAGS):
        # seek(offset, [where]), where(0:start, 1:current, 2:end)        
        self.image_file = read_file(os.path.join(FLAGS.data_path, FLAGS.img_name))
        self.image_size = int((self.image_file.seek(0, 2) - 16.0) / 784.0)

        self.label_file = read_file(os.path.join(FLAGS.data_path, FLAGS.label_name))
        self.label_size = int((self.label_file.seek(0, 2) - 8.0))

        assert self.image_size == self.label_size

        self.batch_size = FLAGS.batch_size
        self.index = np.arange(self.image_size)
        self.position = 0

        self.one_hot = np.eye(10, dtype=np.float32)

    def next_batch(self):
        images = list()
        labels = list()

        for _ in range(self.batch_size):
            self.image_file.seek(16 + 784 * self.index[self.position])
            self.label_file.seek(8 + self.index[self.position])

            images.append(np.fromfile(self.image_file, dtype=np.ubyte, count=784))
            idx = int(np.fromfile(self.label_file, dtype=np.ubyte, count=1))
            labels.append(self.one_hot[idx])

            self.position += 1

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

    def shuffle(self):
        np.random.shuffle(self.index)
        self.position = 0
