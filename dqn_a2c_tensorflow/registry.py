from environments.atari_wrappers import wrap_deepmind
from environments.parallel_env   import ParallelEnv
import tensorflow as tf
import gym
import os

alg_type = {'off_policy': ['dqn'],
            'on_policy' : ['a2c']}

def create_env(config):
    env = gym.make(config.env)
    
    expt_dir = config.log_path + '/monitor'
    env = gym.wrappers.Monitor(env, expt_dir, force=True, video_callable=False)
    
    # If the replay buffer is needed (off-policy algorithms),
    #   observation is stored in (int) format to efficiently manage the memory 
    # Therefore, scaling function is passed on to the buffer's encoding function
    if config.agent_type in alg_type['off_policy']:
        env = wrap_deepmind(env, scale = False)
    else:
        env = wrap_deepmind(env, scale = True)
    
    # Parallelization is supported if number of actor is more than one
    if config.num_actors > 1:
        env = ParallelEnv(num_processes = config.num_actors, env = env)
        
    return env


from networks.dqn_nature import DqnNature
from networks.dqn_nips   import DqnNips

def create_network(network_type):
    network_type = network_type.lower()
    
    if network_type == 'nips':
        network = DqnNips()
    elif network_type == 'nature':
        network = DqnNature()

    return network


from agents.deepq.dqn import DQN
from agents.actor_critic.a2c import A2C

def create_agent(agent_type, *args, **kwargs):
    agent_type = agent_type.lower()
    
    if agent_type == 'dqn':
        agent = DQN(*args, **kwargs)
    elif agent_type == 'a2c':
        agent = A2C(*args, **kwargs)
    
    return agent