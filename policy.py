import tensorflow as tf


class PolicyEstimator:

    def __init__(self,
                 env,
                 transformer,
                 lamb=1e-5,
                 learning_rate=0.01,
                 scope="policy_estimator"):

        self.env = env
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.processor = transformer

        with tf.variable_scope(scope):
            self._build_model()
            self._build_train_op()

    def _build_model(self):
        self.state = tf.placeholder(tf.float32, [None, 400], name="state")

        self.mu = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(self.state, 0),
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.mu = tf.squeeze(self.mu)

        self.sigma = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(self.state, 0),
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.sigma = tf.squeeze(self.sigma)
        self.sigma = tf.nn.softplus(self.sigma) + 1e-5

        self.norm_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        self.action = self.norm_dist.sample(1)
        self.action = tf.clip_by_value(self.action,
                                       self.env.action_space.low[0],
                                       self.env.action_space.high[0])

    def _build_train_op(self):
        self.action_train = tf.placeholder(tf.float32, name="action_train")
        self.td_error_train = tf.placeholder(
            tf.float32, name="td_error_train")

        self.loss = tf.reduce_sum(-tf.log(
            self.norm_dist.prob(self.action_train) + 1e-5) *
            self.td_error_train - self.lamb * self.norm_dist.entropy())

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess):
        feed_dict = {self.state: self.processor.transform_state(
            state).reshape(-1, 400)}
        return sess.run(self.action, feed_dict=feed_dict)

    def update(self, state, action, td_error, sess):
        feed_dict = {
            self.state: self.processor.transform_state(state).reshape(-1, 400),
            self.action_train: action,
            self.td_error_train: td_error
        }
        sess.run([self.train_op], feed_dict=feed_dict)
