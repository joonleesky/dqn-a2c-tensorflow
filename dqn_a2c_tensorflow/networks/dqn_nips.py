import tensorflow as tf
import tensorflow.contrib.layers as layers

from .network import Network

class DqnNips(Network):
    def __init__(self):
        super().__init__()


    def build_layer(self, name, inputs, output_size):
        with tf.variable_scope(name):
            self.l0 = inputs
            self.l1 = layers.conv2d(inputs = self.l0, 
                                    num_outputs = 16, 
                                    activation_fn = tf.nn.relu, 
                                    stride = 4, 
                                    kernel_size = 8, 
                                    padding = 'VALID')
            self.l2 = layers.conv2d(inputs = self.l1, 
                                    num_outputs = 32, 
                                    activation_fn = tf.nn.relu, 
                                    stride = 2, 
                                    kernel_size = 4, 
                                    padding = 'VALID')
            self.l2 = layers.flatten(self.l2)
            self.l3 = layers.fully_connected(inputs = self.l2, 
                                             num_outputs = 256, 
                                             activation_fn = tf.nn.relu)
            
            outputs = layers.fully_connected(inputs = self.l3, 
                                             num_outputs = output_size,
                                             activation_fn = None)

        return outputs

    