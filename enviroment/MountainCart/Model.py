import tensorflow as tf

class Model:
    """
    This class holds the TensorFlow operations and model definitions
    """
    def __init__(self, num_states, num_actions, batch_size):
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        # define the placeholders
        self.states = None
        self.actions = None
        # the output operations
        self.logits = None
        self.optimizer = None
        self.var_init = None
        # now setup the model
        self.define_model()

    def define_model(self):
        self.states = tf.placeholder(shape=[None, self.num_states], dtype=tf.float32)
        self.q_s_a = tf.placeholder(shape=[None, self.num_actions], dtype=tf.float32)
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self.states, 50, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 50, activation=tf.nn.relu)
        self.logits = tf.layers.dense(fc2, self.num_actions)
        loss = tf.losses.mean_squared_error(self.q_s_a, self.logits)
        self.optimizer = tf.train.AdamOptimizer().minimize(loss)
        self.var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        """
        Returns the output of the network (i.e. by calling the _logits operation) with an input of a single state.
        Note the reshaping operation that is used to ensure that the data has a size (1, num_states). This is called
        whenever action selection by the agent is required
        :param state:
        :param sess:
        :return:
        """
        return sess.run(self.logits, feed_dict={self.states:
                                                     state.reshape(1, self.num_states)})

    def predict_batch(self, states, sess):
        return sess.run(self.logits, feed_dict={self.states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self.optimizer, feed_dict={self.states: x_batch, self.q_s_a: y_batch})

