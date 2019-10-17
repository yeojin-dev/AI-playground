import tensorflow as tf


weight_init = tf.initializers.glorot_normal()
bias_init = tf.constant_initializer(value=0)
default_activator = tf.nn.relu


class MNIST:

    def __init__(self, FLAGS):

        self.num_layers = FLAGS.num_layers
        self.num_nodes = FLAGS.num_nodes

        self.num_depths = FLAGS.num_depths
        self.kernel_size = FLAGS.kernel_size

        self.nw_type = FLAGS.nw_type

        self.digit = tf.placeholder(dtype=tf.float32)
        self.label = tf.placeholder(dtype=tf.float32)

        self.lr = FLAGS.learning_rate
        self._build_network()

    def _dense_layer(self, x, weight_shape, bias_shape, reuse=tf.AUTO_REUSE, name=None):

        with tf.variable_scope(name, reuse=reuse):

            W = tf.get_variable("W", weight_shape, initializer=weight_init)
            b = tf.get_variable("b", bias_shape, initializer=bias_init)

        return tf.nn.relu(tf.matmul(x, W) + b)

    def _dense_inference(self, digit):

        x = self._dense_layer(digit, (784, self.num_nodes[0]), (1, self.num_nodes[0]), name='input')

        for index in range(self.num_layers-1):
            name = f'hidden_{index+1}'
            x = self._dense_layer(
                x=x,
                weight_shape=(self.num_nodes[index], self.num_nodes[index+1]),
                bias_shape=(1, self.num_nodes[index+1]),
                name=name,
            )

        out = self._dense_layer(x, (self.num_nodes[-1], 10), (1, 10), name='output')

        return out

    def _conv_inference(self):

        self.digit.set_shape([None, None, None, 1])

        with tf.variable_scope('conv_1', reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d(
                inputs=self.digit,
                filters=self.num_depths[0],
                kernel_size=self.kernel_size[0],
                padding='SAME',
                activation=default_activator,
                kernel_initializer=weight_init,
            )
            x = tf.nn.max_pool(
                value=x,
                ksize=(1, 2, 2, 1),
                strides=(1, 2, 2, 1),
                padding='SAME',
            )

        for index in range(1, self.num_layers-1):
            name = f'conv_{index+1}'

            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                x = tf.layers.conv2d(
                    inputs=x,
                    filters=self.num_depths[index],
                    kernel_size=self.kernel_size[index],
                    padding='SAME',
                    activation=default_activator,
                    kernel_initializer=weight_init,
                )
                x = tf.nn.max_pool(
                    value=x,
                    ksize=(1, 2, 2, 1),
                    strides=(1, 2, 2, 1),
                    padding='SAME',
                )

        name = f'conv_{self.num_layers}'
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d(
                inputs=x,
                filters=self.num_depths[-1],
                kernel_size=self.kernel_size[-1],
                padding='SAME',
                activation=default_activator,
                kernel_initializer=weight_init,
            )

        name = 'output'
        divider = 2 ** (self.num_layers - 1)
        size = 28 // divider
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d(
                inputs=x,
                filters=10,  # number of digits(0 to 9)
                kernel_size=(size, size),
                activation=None,
                kernel_initializer=weight_init,
            )

        return tf.reshape(x, (-1, 10))

    def _get_loss(self, nw_out, label):

        x_entrophy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=nw_out, labels=label)
        loss = tf.reduce_mean(x_entrophy)

        return loss

    def _build_network(self):

        if self.nw_type == 'dense':
            self.inference = self._dense_inference()
        elif self.nw_type == 'conv':
            self.inference = self._conv_inference()
        else:
            raise NotImplementedError('nw_type shall be "conv" or "dense".')

        nw_out = self.inference
        self.cost = self._get_loss(nw_out, self.label)

    def optimizer(self):

        opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        train_op = opt.minimize(self.cost)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        return train_op, self.cost

    def evaluate(self, output, y):

        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy
