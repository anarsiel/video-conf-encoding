import tensorflow as tf


class Network(object):
    def __init__(self, n_layers):
        """
        constructor
        :param n_layers: number of nodes in each layer of the network
        """
        # store the parameters of network
        self.params = []

        # Declare layer-wise weights and biases
        self.W1 = tf.Variable(
            tf.random.normal([n_layers[0], n_layers[1]], stddev=0.1),
            name='W1')
        #         self.b1 = tf.Variable(tf.random.normal([n_layers[1]], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32, seed=0), name='b1')
        self.b1 = tf.Variable(tf.zeros([1, n_layers[1]]))

        self.W2 = tf.Variable(
            tf.random.normal([n_layers[1], n_layers[2]], stddev=0.1),
            name='W2')
        #         self.b2 = tf.Variable(tf.random.normal([n_layers[2]], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32, seed=0), name='b2')
        self.b2 = tf.Variable(tf.zeros([1, n_layers[2]]))

        self.W3 = tf.Variable(
            tf.random.normal([n_layers[2], n_layers[3]], stddev=0.1),
            name='W3')
        #         self.b3 = tf.Variable(tf.random.normal([n_layers[3]], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32, seed=0), name='b3')
        self.b3 = tf.Variable(tf.zeros([1, n_layers[3]]))

        # Collect all initialized weights and biases in self.params
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x):
        """
        Forward pass of the network
        :param x: input data
        :return: predicted label
        """
        X_tf = tf.cast(x, dtype=tf.float32)
        Z1 = tf.matmul(X_tf, self.W1) + self.b1
        Z1 = tf.nn.relu(Z1)
        Z2 = tf.matmul(Z1, self.W2) + self.b2
        Z2 = tf.nn.relu(Z2)
        Z3 = tf.matmul(Z2, self.W3) + self.b3
        # 		Y = tf.nn.sigmoid(Z3)
        return Z3

    def loss(self, y_true, logits):
        """
        logits - Tensor of shape (batch_size, size_output)
        y_true - Tensor of shape (batch_size, size_output)
        """
        y_true_tf = tf.cast(tf.reshape(y_true, (-1, 1)), dtype=tf.float32)
        logits_tf = tf.cast(tf.reshape(logits, (-1, 1)), dtype=tf.float32)
        return tf.compat.v1.losses.sigmoid_cross_entropy(y_true_tf, logits_tf)

    def backward(self, x, y):
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
        with tf.GradientTape() as tape:
            predicted = self.forward(x)
            current_loss = self.loss(y, predicted)
        grads = tape.gradient(current_loss, self.params)
        optimizer.apply_gradients(zip(grads, self.params),
                                  global_step=tf.compat.v1.train.get_or_create_global_step())