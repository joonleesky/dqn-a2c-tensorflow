import tensorflow as tf

class Agent(object):
    """
    This class specifies the basic Agent class. 
    To define your own agent, subclass this class and implement the functions below.
    See dqn.py for an example implementation.
    """

    def __init__(self, 
                 sess, 
                 env, 
                 config, 
                 network):
        """
        Parameters
        ----------
        sess: tf.Session
            tensorflow session to build the computation graph
        env: gym.Env
            mario environment with following openAI Gym API
        config: tf.app.flags
            configurations to map the hyperparameters for training
        network: neural network
            neural network to perform policy evaluation / control
        """
        # Basic Settings
        self.sess  = sess
        self.env   = env
        self.network  = network
        
        # Environment
        self.observation_dims = list(env.observation_space.shape)
        self.input_shape = [None] + self.observation_dims
        self.action_size = self.env.action_space.n
        
        # Training
        self.total_timesteps = config.total_timesteps
        self.num_actors = config.num_actors
        
        # Logging
        self.log_path = config.log_path
        self.log_freq = config.log_freq
        
        # Initialize the agent
        self.build_graph()
        self.build_logger()
        self.initialize_model()
        self.t = 0
        self.last_obs = self.env.reset()
        pass

    def build_graph(self):
        """
        Build agent's function approximators
        """
        pass
    
    def build_logger(self):
        """
        Build graph for the logging procedure
        """
        pass
    
    def initialize_model(self):
        """
        Initialize the variables of the model
        """
        pass

    def train(self):
        """
        Train the neural network with the trajectories
        """
        pass
    
    def play(self):
        """
        Evaluate the trained network with simulations
        (This is seperated from simulation since exploration is not necessary)
        """
        pass
    
    def predict(self, obs_t):
        """
        Parameters
        ----------
        obs_t: np.array()
            numpy array state at time step t
        
        Returns
        -------
        a_t: int()
            integral to indicate the action
        """
        pass

