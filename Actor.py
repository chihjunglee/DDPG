"""
This class implements the policy neural network.  This class primarily needs to be able to (i) build a neural network with the
provided architecture, and (ii) find the gradient of an action with respect to the policy parameters.  This gradient
is part of the deterministic policy gradient and is need to improve the policy in the "learning" section.
I used the below reference for help determining the gradients and the required function for the actor network
but the creation of the actor neural networks is my own code written in Tensorflow.
Reference: https://github.com/pemami4911/deep-rl/tree/master/ddpg
"""

import tensorflow as tf
import math

# Actor/Critic Neural Network Architecture
LAYER_1_SIZE = 400
LAYER_2_SIZE = 300

class ActorNetwork(object):

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau



        # Create non_target policy network
        self.inputs, self.out, self.scaled_out = self.create_actor_network(nn_type = "non_target")

        # Find trainable variables that will be needed for gradient calculations
        self.network_params = tf.trainable_variables()
        print(self.network_params[0].get_shape().as_list())


        # Create target policy network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network(nn_type = "target")

        # Find trainable variables that will be needed for gradient calculations
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]


        # Below are some check that the trainable weights have the correct shape
        for i in range(len(self.target_network_params)):
            print(i)
            elem_1 = tf.multiply(self.network_params[i],self.tau)
            print("elem_1",elem_1.get_shape().as_list())
            elem_2 = tf.multiply(self.target_network_params[i],1.-self.tau)
            print("elem_2",elem_2.get_shape().as_list())
            tensor_sum = elem_1 + elem_2

        print("act_nn shape",elem_1)
        print("target_nn shape",elem_2)
        tensor_sum = (elem_1 + elem_2).get_shape().as_list()
        print("Sum shape",tensor_sum)

        # Create operation in graph to update target neural network
        self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau)) for i in range(len(self.target_network_params))]
        print("length",len(self.update_target_network_params))

        # Store critic gradient with respect to actions.  To be used in policy gradient update.
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Determine gradients of actions with respect to parameters of actor network.  Combine gradients of Q-value with respect
        # to action and action with respect to parameters of neural network.  The produce is negated to perform gradient ascent
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

        # Use ADAM to perform gradient ascent using the gradient calculated in the operation above and the trainable variables
        # in the network_params variable
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)


    # Create policy network with architecture provided.  The policy network is a mapping from states to actions.
    def create_actor_network(self, nn_type):

        # States are the input variable
        states = tf.placeholder(tf.float32,shape = [None,self.s_dim])

        # First layer
        h1_w = tf.Variable(tf.random_uniform([self.s_dim,LAYER_1_SIZE],minval=-1/math.sqrt(self.s_dim),maxval=1/math.sqrt(self.s_dim)))
        h1_b = tf.Variable(tf.random_uniform([LAYER_1_SIZE],minval=-1/math.sqrt(self.s_dim),maxval=1/math.sqrt(self.s_dim)))
        h1 = tf.nn.relu(tf.matmul(states,h1_w) + h1_b)

        # Second layer
        h2 = tf.layers.dense(inputs = h1, units = LAYER_2_SIZE, activation=tf.nn.relu)
        init = tf.random_uniform_initializer(minval = -.003, maxval = 0.003)
        act_not_scaled = tf.layers.dense(inputs = h2, units = self.a_dim, activation = tf.nn.tanh, kernel_initializer = init, bias_initializer = init)

        # Scale output to bounds of environment
        act_scaled = tf.multiply(act_not_scaled,self.action_bound)

        return states, act_not_scaled, act_scaled


    # The train function is how the "Learning" phase of the learning loop is implemented.  The policy neural network
    # parameters are updated in the diretion of the deterministic policy gradient
    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    # The predict function is used to find the action prescribed by the policy network given the current state.
    # The agent will experiment by adding noise to the output of this function
    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    # This function updates the target neural network by using values from the non_target network
    def update_target_network(self):
        # update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau)) for i in range(len(self.target_network_params))]
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
