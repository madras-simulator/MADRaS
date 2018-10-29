"""Actor-Network."""

import tensorflow as tf

# Hyper Parameters
LAYER1_SIZE = 300
LAYER2_SIZE = 400
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 32


class ActorNetwork:
    """docstring for ActorNetwork."""

    def __init__(self, sess, state_dim, action_dim):
        """Init Method."""
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        # create actor network
        self.state_input, self.action_output, self.net =\
            self.create_network(state_dim, action_dim)

        # create target actor network
        self.target_state_input, self.target_action_output, self.target_update, self.target_net =\
            self.create_target_network(state_dim, action_dim, self.net)

        # define training rules
        self.create_training_method()

        # self.sess.run(tf.initialize_all_variables())
        self.sess.run(tf.global_variables_initializer())

        self.update_target()
        # self.load_network()

    def create_training_method(self):
        """Trainer."""
        self.q_gradient_input = tf.placeholder("float", [None,
                                                         self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output,
                                                 self.net,
                                                 -self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(
            list(zip(self.parameters_gradients, self.net)))

    def create_network(self, state_dim, action_dim):
        """Network Creater."""
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        state_input = tf.placeholder("float", [None, state_dim])
        regularization = tf.contrib.layers.l2_regularizer(scale=0.01)

        # ###################GET_VARIABLES##################
        W1 = tf.get_variable("W1", shape=[state_dim, layer1_size],
                             regularizer=regularization)
        b1 = tf.get_variable("b1", shape=[layer1_size],
                             regularizer=regularization,
                             initializer=tf.zeros_initializer)
        W2 = tf.get_variable("W2", shape=[layer1_size, layer2_size],
                             regularizer=regularization)
        b2 = tf.get_variable("b2", shape=[layer2_size],
                             regularizer=regularization,
                             initializer=tf.zeros_initializer)

        layer1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
        layer1_norm = tf.layers.batch_normalization(layer1)
        layer2 = tf.nn.relu(tf.matmul(layer1_norm, W2) + b2)
        layer2_norm = tf.layers.batch_normalization(layer2)

        W_trackpos = tf.get_variable("W_trackpos", shape=[layer2_size, 1],
                                     regularizer=regularization)
        b_trackpos = tf.get_variable("b_trackpos", shape=[1],
                                     regularizer=regularization,
                                     initializer=tf.zeros_initializer)
        W_vel = tf.get_variable("W_vel", shape=[layer2_size, 1],
                                regularizer=regularization)
        b_vel = tf.get_variable("b_vel", shape=[1],
                                regularizer=regularization,
                                initializer=tf.zeros_initializer)

        trackpos = tf.nn.relu(tf.matmul(layer2_norm, W_trackpos) + b_trackpos)
        trackpos_res = trackpos + tf.gather(state_input, indices=[20], axis=1)

        velocity = tf.nn.relu(tf.matmul(layer2_norm, W_vel) + b_vel)
        velocity_res = velocity + tf.gather(state_input, indices=[21], axis=1)
        action_output = tf.concat([trackpos_res, velocity_res], 1)

        return state_input, action_output, [W1, b1, W2, b2,
                                            W_trackpos, b_trackpos,
                                            W_vel, b_vel]

    def create_target_network(self, state_dim, action_dim, net):
        """Target Network Creater."""
        state_input = tf.placeholder("float", [None, state_dim])
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer1 = tf.nn.relu(tf.matmul(state_input, target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(tf.matmul(layer1, target_net[2]) + target_net[3])

        trackpos = tf.nn.relu(tf.matmul(layer2, target_net[4]) + target_net[5])
        trackpos_res = trackpos + tf.gather(state_input, indices=[20], axis=1)
        velocity = tf.nn.relu(tf.matmul(layer2, target_net[6]) + target_net[7])
        velocity_res = velocity + tf.gather(state_input, indices=[21], axis=1)

        action_output = tf.concat([trackpos_res, velocity_res], 1)
        return state_input, action_output, target_update, target_net

    def update_target(self):
        """Target update."""
        self.sess.run(self.target_update)

    def train(self, q_gradient_batch, state_batch):
        """Train Function."""
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.state_input: state_batch})

    def actions(self, state_batch):
        """Get Network Output."""
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: state_batch})

    def action(self, state):
        """Get output for a single obs."""
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: [state]})[0]

    def target_actions(self, state_batch):
        """Get Target Network Outputs."""
        return self.sess.run(self.target_action_output, feed_dict={
            self.target_state_input: state_batch})
