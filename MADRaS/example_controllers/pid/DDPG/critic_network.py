"""Critic-Network."""

import tensorflow as tf

LAYER1_SIZE = 300
LAYER2_SIZE = 600
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.0001


class CriticNetwork:
    """docstring for CriticNetwork."""

    def __init__(self, sess, state_dim, action_dim):
        """Init Method."""
        self.time_step = 0
        self.sess = sess
        # create q network
        self.state_input, self.action_input, self.q_value_output, self.net =\
            self.create_q_network(state_dim, action_dim)

        # create target q network (the same structure with q network)
        self.target_state_input, self.target_action_input, self.target_q_value_output, self.target_update =\
            self.create_target_q_network(state_dim, action_dim, self.net)

        self.create_training_method()

        # initialization
        # self.sess.run(tf.initialize_all_variables())
        self.sess.run(tf.global_variables_initializer())

        self.update_target()

    def create_training_method(self):
        """Define training optimizer."""
        self.y_input = tf.placeholder("float", [None, 1])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
        self.TD_error = tf.reduce_mean(tf.square(self.y_input - self.q_value_output))
        self.cost = self.TD_error + weight_decay
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
        self.action_gradients = tf.gradients(self.q_value_output, self.action_input)

    def create_q_network(self, state_dim, action_dim):
        """Q Network."""
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        state_input = tf.placeholder("float", [None, state_dim])
        action_input = tf.placeholder("float", [None, action_dim])
        regularization = tf.contrib.layers.l2_regularizer(scale=0.01)

        W1 = tf.get_variable("W1_c", shape=[state_dim, layer1_size],
                             regularizer=regularization)
        b1 = tf.get_variable("b1_c", shape=[layer1_size],
                             regularizer=regularization)
        W2 = tf.get_variable("W2_c", shape=[layer1_size, layer2_size],
                             regularizer=regularization)
        W2_action = tf.get_variable("W2_action", shape=[action_dim, layer2_size],
                                    regularizer=regularization)
        b2 = tf.get_variable("b2_c", shape=[layer2_size],
                             regularizer=regularization)
        W3 = tf.get_variable("W3_c", shape=[layer2_size, 1],
                             regularizer=regularization)
        b3 = tf.get_variable("b3_c", shape=[1], regularizer=regularization)

        layer1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
        layer1_norm = tf.layers.batch_normalization(layer1)
        layer2 = tf.nn.relu(tf.matmul(layer1_norm, W2) +
                            tf.matmul(action_input, W2_action) + b2)
        layer2_norm = tf.layers.batch_normalization(layer2)
        q_value_output = tf.identity(tf.matmul(layer2_norm, W3) + b3)

        return state_input, action_input, q_value_output, [W1, b1, W2,
                                                           W2_action,
                                                           b2, W3, b3]

    def create_target_q_network(self, state_dim, action_dim, net):
        """Target Q Network."""
        state_input = tf.placeholder("float", [None, state_dim])
        action_input = tf.placeholder("float", [None, action_dim])

        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer1 = tf.nn.relu(tf.matmul(state_input, target_net[0]) +
                            target_net[1])
        layer1_norm = tf.layers.batch_normalization(layer1)
        layer2 = tf.nn.relu(tf.matmul(layer1_norm, target_net[2]) +
                            tf.matmul(action_input, target_net[3]) +
                            target_net[4])
        layer2_norm = tf.layers.batch_normalization(layer2)
        q_value_output = tf.identity(tf.matmul(layer2_norm, target_net[5]) +
                                     target_net[6])

        return state_input,action_input,q_value_output,target_update

    def update_target(self):
        """Update Target Network."""
        self.sess.run(self.target_update)

    def train(self, y_batch, state_batch, action_batch):
        """Train Step."""
        self.time_step += 1
        self.sess.run(self.optimizer, feed_dict={
            self.y_input: y_batch,
            self.state_input: state_batch,
            self.action_input: action_batch})

    def gradients(self, state_batch, action_batch):
        """Get Network Gradients."""
        return self.sess.run(self.action_gradients, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch})[0]

    def target_q(self, state_batch, action_batch):
        """Output of the Target Q Network."""
        return self.sess.run(self.target_q_value_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_action_input: action_batch})

    def q_value(self, state_batch, action_batch):
        """Get Q-Value."""
        return self.sess.run(self.q_value_output, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch})

    def td_error(self, state, action, y):
        """TD Error Calculation."""
        return self.sess.run(self.TD_error, feed_dict={self.y_input: y,
                                                       self.state_input: state,
                                                       self.action_input: action})
