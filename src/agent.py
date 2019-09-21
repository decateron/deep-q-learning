"""
Module:
    agent.py
Overview:
    In this module, I build an agent (Deep Q Neural Network).
    This Neural Network will decide which action to take in this game state.
Classed:
    DQNetwork
"""



import tensorflow as tf



class DQNetwork:
    """
    This is the Deep Q-learning model.
    The model takes a stack of 4 frames as input.
    It passes through 3 convolution layers.
    Then it is flattened.
    Finally, it passes through 2 fully connected layers.

    It outputs a Q value for each action.
    """
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # I create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            
            # Target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            """
            First convolution layer:
                Exponential Linear Unit
            """

            # Input is 110x84x4
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32,
                                          kernel_size=[8,8],
                                          strides=[4,4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")
            
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
            
            """
            Second convolution layer:
                Exponential Linear Unit
            """

            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4,4],
                                          strides=[2,2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")

            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")            
            
            """
            Third convolution layer:
                Exponential Linear Unit
            """

            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=64,
                                          kernel_size=[3,3],
                                          strides=[2,2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
            
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            """
            First fully connected layer:
                Exponential Linear Unit
            """
            
            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")
            
            """
            Second fully connected layer:
                Linear Activation Unit
            """

            
            self.output = tf.layers.dense(inputs=self.fc, 
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=self.action_size, 
                                          activation=None,
                                          name="fc2")
            

  
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
