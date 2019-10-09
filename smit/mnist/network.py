import tensorflow as tf


weight_init = tf.initializers.glorot_normal()
bias_init = tf.constant_initializer(value=0)


class MNIST:

    def __init__(self, FLAGS):

        self.num_layers = FLAGS.num_layers
        self.num_nodes = FLAGS.num_nodes

        self.digit = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        self.lr = FLAGS.learning_rate
        self.build_network()

    def _layer(self, x, weight_shape, bias_shape, reuse=tf.AUTO_REUSE, name=None):

        with tf.variable_scope(name, reuse=reuse):

            W = tf.get_variable("W", weight_shape, initializer=weight_init)
            b = tf.get_variable("b", bias_shape, initializer=bias_init)

        return tf.nn.relu(tf.matmul(x, W) + b)

    def _inference(self, digit):

        x = self._layer(digit, (784, self.num_nodes[0]), (1, self.num_nodes[0]), name='input')

        for index in range(self.num_layers-1):
            name = f'hidden_{index-1}'
            x = self._layer(
                x=x,
                weight_shape=(self.num_nodes[index], self.num_nodes[index+1]),
                bias_shape=(1, self.num_nodes[index+1]),
                name=name,
            )

        out = self._layer(x, (self.num_nodes[-1], 10), (1, 10), name='output')

        return out

    def _get_loss(self, nw_out, label):

        x_entrophy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=nw_out, labels=label)
        loss = tf.reduce_mean(x_entrophy)

        return loss

    def _build_network(self):

        nw_out = self._inference(self.digit)
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
