from collections import deque
import numpy as np

class ReplayBuffer(object):
    """ 
    Replay buffer stores the trajectories and encode those to feed it to the neural network
    Several memory optimization methods are listed below
    1. To efficiently manage the overlapping observations, observation is stored as pointer objects.
    2. Since the frame is managed with integer datatype, scaling the frame into [0,1] is only done before passed to hte model. 
    Storing the observation is seperated from storing the other information in order to efficiently store the next observation.
    """ 
    def __init__(self, size, observation_dims, scale):
        """
        Parameters
        ----------
        size: int
            size of the replay buffer
        observation_dims: list    
            list of the environment's observation dimensions
        scale: boolean
            whether to scale the observation into [0,1] 
        """
        # Initialize
        self.size = size
        self.observation_dims = observation_dims
        self.scale = scale
        self.idx = -1
        self.num_in_buffer = 0
        
        # Initialize the buffer
        self.obses   = deque(maxlen = self.size)
        self.actions = deque(maxlen = self.size)
        self.rewards = deque(maxlen = self.size)
        self.dones   = deque(maxlen = self.size)
        
    def store_obs(self, obs):
        self.obses.append(obs)
        self.idx = min(self.idx + 1, self.size-1)
        
    def store_effect(self, action, reward, done):
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        
        self.num_in_buffer = min(self.num_in_buffer+1, self.size)
        
    def sample(self, batch_size):
        if self.num_in_buffer < batch_size:
            assert('Replay buffer does not has enough data to sample')
        idxes = np.random.choice(np.arange(self.num_in_buffer-1), batch_size, replace = False)
        obs_batch  = np.array([self._encode_obs(idx) for idx in idxes])
        act_batch  = np.array([self.actions[idx] for idx in idxes], dtype = np.int32).reshape(-1)
        rew_batch  = np.array([self.rewards[idx] for idx in idxes], dtype = np.float32).reshape(-1)
        done_batch = np.array([self.dones[idx] for idx in idxes], dtype = np.float32).reshape(-1)
        next_obs_batch = np.array([self._encode_obs(idx+1) for idx in idxes])

        return obs_batch, act_batch, rew_batch, done_batch, next_obs_batch
    
    def _encode_obs(self, idx):
        if self.scale == True:
            return np.array(self.obses[idx]).astype(np.float32) / 255.0
        else:
            return np.array(self.obses[idx]).astype(np.float32)
    
    def encode_recent_obs(self):
        return np.expand_dims(self._encode_obs(self.idx), 0)
    
    
if __name__=='__main__':
    print('Test Replay Buffer')
    