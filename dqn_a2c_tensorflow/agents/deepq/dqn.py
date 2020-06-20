from dqn_a2c_tensorflow.agents.agent  import Agent
from dqn_a2c_tensorflow.utils.scheduler import LinearScheduler, ConstantScheduler
from dqn_a2c_tensorflow.utils.replay_buffer import ReplayBuffer
import tensorflow as tf
import numpy as np
import random

default_params = {
    'gamma'             : 0.99,
    'batch_size'        : 32,
    'learning_start'    : 10000,
    'q_update_freq'     : 4,
    'target_update_freq': 1000,
    'grad_clip_norm'    : 10.0,
    
    'optimizer'     : tf.train.AdamOptimizer,
    'lr_scheduler'  : ConstantScheduler(1e-4),
    'eps_scheduler' : LinearScheduler(initial_value = 1.0, final_value = 0.01, schedule_timesteps = int(1e6)),
    'buffer_size'   : int(1e5)
}

class DQN(Agent):
    def __init__(self, 
                 sess, 
                 env, 
                 config, 
                 network,
                 params = default_params):
        """
        Parameters
        ----------
        Note: needs to be descripted
        """
        self.params = params
        self.replay_buffer = ReplayBuffer(size = params['buffer_size'],
                                          observation_dims = list(env.observation_space.shape),
                                          scale = True)
        super(DQN, self).__init__(sess, env, config, network)
        pass
    
    def build_graph(self):
        ##################
        ## Placeholders ##
        ##################
        self.obs_t_ph     = tf.placeholder(tf.float32, shape = self.input_shape, name ='obs_t')
        self.obs_t1_ph    = tf.placeholder(tf.float32, shape = self.input_shape, name ='obs_t1')
        self.action_t_ph  = tf.placeholder(tf.int32,   shape = [None], name = 'action_t')
        self.reward_t_ph  = tf.placeholder(tf.float32, shape = [None], name = 'reward_t')
        self.done_mask_ph = tf.placeholder(tf.float32, shape = [None], name = 'done_t')
            
        self.learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')    
        
        #############
        ## Network ##
        #############
        self.q_network = self.network.build_layer('q_net',self.obs_t_ph, self.action_size)
        self.target_network = self.network.build_layer('target_net', self.obs_t1_ph, self.action_size)
        q_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'q_net')
        target_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'target_net')
        
        # Action for prediction
        self.action = tf.argmax(self.q_network, 1)
        
        # Set prediction and the target
        actions_one_hot = tf.one_hot(self.action_t_ph, self.action_size)
        pred_q_t    = tf.reduce_sum(self.q_network * actions_one_hot, 1)
        target_q_t1 = tf.reduce_max(self.target_network, 1)    
        target = self.reward_t_ph + self.params['gamma'] * target_q_t1 * (1 - self.done_mask_ph)
        
        ##########
        ## Loss ##
        ##########
        # Huber Loss with temporal difference bellman error
        delta = pred_q_t - target
        self.loss = tf.reduce_mean(tf.where(tf.abs(delta) < 1.0,
                                            tf.square(delta) * 0.5,
                                            tf.abs(delta) - 0.5))
        
        ##################
        ## Optimization ##
        ##################
        # Minimize the bellman error - only update the q network
        optimizer = self.params['optimizer'](self.learning_rate)
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, self.params['grad_clip_norm'])
        self.train_fn = optimizer.minimize(self.loss, var_list = q_network_vars)
        
        # Update the target network
        update_fn = []
        q_network_vars = sorted(q_network_vars, key=lambda var: var.name)
        target_network_vars = sorted(target_network_vars, key=lambda var: var.name)
        for q_net_var, target_net_var in zip(q_network_vars, target_network_vars):
            update_fn.append(tf.assign(target_net_var, q_net_var))
        self.update_fn = tf.group(*update_fn)
    
    def build_logger(self):
        # Loss placeholder is used to simplify the logging procedure
        self.loss_val   = tf.placeholder(tf.float32, name = 'loss')
        self.avg_reward = tf.placeholder(tf.float32, name = 'avg_reward')
        self.max_reward = tf.placeholder(tf.float32, name = 'max_reward')

        tf.summary.scalar('loss',self.loss_val)
        tf.summary.scalar('avg_reward', self.avg_reward)
        tf.summary.scalar('max_reward', self.max_reward)
        
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_path, self.sess.graph)
        
    def initialize_model(self):
        self.sess.run(tf.global_variables_initializer())        
    
    def _step(self):
        """
        Take a step in the enviornment and store the transition
        """
        # Store the last observation
        self.replay_buffer.store_obs(self.last_obs)
        
        # Epsilon-greedy
        eps = self.params['eps_scheduler'].value(self.t)
        if random.random() < eps or self.t < self.params['learning_start']:
            action = self.env.action_space.sample()
        else:
            obs  = self.replay_buffer.encode_recent_obs()
            action = self.predict(obs)
        # Take a step and store the transition
        next_obs, reward, done, info = self.env.step(action)
        self.replay_buffer.store_effect(action, reward, done)
        
        if done == True:
            self.last_obs = self.env.reset()
        else:
            self.last_obs = next_obs
    
    def train(self):
        while self.t < self.total_timesteps:
            self._step()
            # Update the actor & critic network
            if self.t >= self.params['learning_start']:
                # Update the actor(q) network
                if self.t % self.params['q_update_freq']:
                    obs_batch, act_batch, rew_batch, done_batch, next_obs_batch =\
                        self.replay_buffer.sample(self.params['batch_size'])
                    lr = self.params['lr_scheduler'].value(self.t)
                    feed_dict = {self.obs_t_ph:     obs_batch,
                                 self.action_t_ph:  act_batch,
                                 self.reward_t_ph:  rew_batch,
                                 self.done_mask_ph: done_batch,
                                 self.obs_t1_ph:    next_obs_batch,
                                 self.learning_rate: lr}        
                    loss, _ = self.sess.run([self.loss, self.train_fn], feed_dict = feed_dict)
        
                # Update the critic(target) network
                if self.t % self.params['target_update_freq']:
                    self.sess.run(self.update_fn)
                    
            # Log the training progress
            if self.t % self.log_freq == 0:
                print('Timestep:',self.t)
                if self.t < self.params['learning_start']:
                    loss = 0.0
                # Get rewards for recent 100 episodes
                episode_rewards = self.env.get_episode_rewards()[-100:]
                if len(episode_rewards) > 0:
                    feed_dict = {self.loss_val:   loss,
                                 self.avg_reward: np.mean(episode_rewards),
                                 self.max_reward: np.max(episode_rewards)}
                    summary = self.sess.run(self.merged, feed_dict = feed_dict)
                    self.writer.add_summary(summary, global_step = self.t)
                
            self.t += 1
    
    def play(self):
        pass
    
    def predict(self, obs_t):
        return self.sess.run(self.action, feed_dict = {self.obs_t_ph: obs_t})
