import os
import matplotlib.pyplot as plt
import numpy as np


def show_image(images, labels):
    sq_row = int(np.sqrt(np.shape(images[0])))
    images = np.reshape(images, (-1, 28, 28))  # -1은 자동으로 값을 맞춘다는 의미

    pos = 0
    for row in range(sq_row):
        row_img = images[pos]
        pos += 1
        for col in range(sq_row-1):
            row_img = np.concatenate((row_img, images[pos]), axis=1)
            pos += 1
        if row == 0:
            show_img = row_img + 0
        else:
            show_img = np.concatenate((show_img, row_img), axis=1)
    
    plt.imshow(show_img, cmap='gray')
    plt.show()
    return 0


def read_file(filename):
    return open(filename, 'rb')


class mnist_data:

    def __init__(self, FLAGS):
        # seek(offset, [where]), where(0:start, 1:current, 2:end)        
        self.image_file = read_file(os.path.join(FLAGS.data_path, FLAGS.img_name))
        self.image_size = int((self.image_file.seek(0, 2) - 16.0) / 784.0)
        
        self.label_file = read_file(os.path.join(FLAGS.data_path, FLAGS.label_name))
        self.label_size = int((self.label_file.seek(0, 2) - 8.0))

        assert self.image_size = self.label_size

        self.batch_size = FLAGS.batch_size
        self.index = np.arrage(self.image_size)
        self.position = 0
    
    def next_batch(self):
        images = list()
        labels = list()

        for _ in range(self.batch_size):
            self.image_file.seek(16 + 784 * self.index[self.position])
            self.label_file.seek(8 + self.index[self.position])

            images.append(np.fromfile(self.image_file, dtype=np.ubyte, count=784))
            lables.append(np.fromfile(self.label_file, dtype=np.ubyte, count=1))

            self.position += 1
        
        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)
    
    def shuffle(self):
        np.random.shuffle(self.index)
        self.position = 0
