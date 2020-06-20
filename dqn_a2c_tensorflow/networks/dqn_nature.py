import tensorflow as tf
import tensorflow.contrib.layers as layers

from .network import Network

class DqnNature(Network):
    def __init__(self):
        super().__init__()
        
    def build_layer(self, name, inputs, output_size):
        with tf.variable_scope(name):
            self.header  = self.build_header('conv', inputs)
            self.outputs = self.build_tail('mlp', self.header, output_size)

        return self.outputs
    
    def build_header(self, name, inputs):
        with tf.variable_scope(name):
            self.l1 = layers.conv2d(inputs = inputs, 
                                    num_outputs = 32, 
                                    activation_fn = tf.nn.relu,
                                    stride = 4, 
                                    kernel_size = 8, 
                                    padding = 'VALID')
            self.l2 = layers.conv2d(inputs = self.l1, 
                                    num_outputs = 64, 
                                    activation_fn = tf.nn.relu, 
                                    stride = 2, 
                                    kernel_size = 4, 
                                    padding = 'VALID')
            self.l3 = layers.conv2d(inputs = self.l2, 
                                    num_outputs = 64, 
                                    activation_fn = tf.nn.relu, 
                                    stride = 1, 
                                    kernel_size = 3, 
                                    padding = 'VALID')
            outputs = layers.flatten(self.l3)
        return outputs
    
    def build_tail(self, name, inputs, output_size, last_layer_initializer = layers.xavier_initializer()):
        with tf.variable_scope(name):
            self.l4 = layers.fully_connected(inputs = inputs, 
                                             num_outputs = 512, 
                                             activation_fn = tf.nn.relu)
            
            outputs = layers.fully_connected(inputs = self.l4, 
                                             num_outputs = output_size,
                                             activation_fn = None,
                                             weights_initializer = last_layer_initializer)
        return outputs