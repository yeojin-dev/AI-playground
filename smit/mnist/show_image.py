import numpy as np
import matplotlib.pyplot as plt


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
