class Network(object):
    """
    This class specifies the basic Network class.
    To define your own neural network, subclass this class and implement the functions below.
    See mlp.py for an example implementation.
    """

    def __init__(self):
        """
        Parameters
        ----------
        output_size: int
            number of the output for the nn
        """
        pass
    
    def build_layer(self, name, inputs, output_size):
        """
        Parameters
        ----------
        name: string
            name to indicate the network
        inputs: tf.placeholder
            shape of the input is as follows,
            Visual input    : [N, H, W, C]
            Non-visual input: [N, D]
        output_size: int
            size of the output
            
        Returns
        -------
        outputs: tf.placeholder
            shape of the output is [N, self.output_size]
        """