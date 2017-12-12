import tensorflow as tf


class ValueEstimator:
	"""Implements a value function estimator using a single layer nn"""
    def __init__(self, env, transformer, learning_rate=0.01, scope="value_estimator"):
        self.env = env
        self.learning_rate = learning_rate
        self.processor = transformer

        with tf.variable_scope(scope):
            self._build_model()
            self._build_train_op()

    def _build_model(self):
        self.state = tf.placeholder(tf.float32, [None, 400], name="state")

        self.value = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(self.state, 0),
            num_outputs=1,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.value = tf.squeeze(self.value)

    def _build_train_op(self):
        self.target = tf.placeholder(tf.float32, name="target")
        self.loss = tf.reduce_mean(tf.squared_difference(self.value, self.target))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess):
        return sess.run(self.value, feed_dict={self.state: 
            self.processor.transform_state(state).reshape(-1,400)})

    def update(self, state, target, sess):
        feed_dict = {
            self.state: 
            self.processor.transform_state(state).reshape(-1,400),
            self.target: target
        }
        sess.run([self.train_op], feed_dict=feed_dict)