import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from registry import create_env, create_network, create_agent

flags = tf.app.flags

flags.DEFINE_string('env', 'PongNoFrameskip-v4', 'Type of the environment')
flags.DEFINE_string('agent_type', 'dqn', 'Type of the agent [dqn, a2c]')
flags.DEFINE_string('network_type','nature', 'Type of the network architecture')

# Training
flags.DEFINE_integer('total_timesteps',int(2e6) , 'Number of total timestep to train the agent')
flags.DEFINE_integer('num_actors', 1, 'number of actors to simulate')

# Debug
flags.DEFINE_string('log_path', './results/pong_v4/dqn/test', 'Path of the log to save the experiment')
flags.DEFINE_integer('log_freq', 10000, 'Frequency of the logging procedure')


if __name__=='__main__':
    config = flags.FLAGS
    
    # Write down the configurations into json
    config_dict = config.flag_values_dict()
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    with open(str(config.log_path)+'/config.json', 'w') as f:
        json.dump(config_dict, f)
    
    # Create an environment
    # If actor is more than one, environment should be parallelized
    env = create_env(config)

    # Create the network
    network = create_network(config.network_type)
    
    # Initialize the Training Agent
    TrainAgent = create_agent(config.agent_type,
                              sess = tf.Session(), 
                              env  = env, 
                              config  = config, 
                              network = network)
    
    TrainAgent.train()
