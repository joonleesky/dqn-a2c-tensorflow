from dqn_a2c_tensorflow.agents.agent  import Agent
from dqn_a2c_tensorflow.utils.initializer import normc_initializer
from dqn_a2c_tensorflow.utils.distribution import entropy
from dqn_a2c_tensorflow.utils.scheduler import ConstantScheduler
import tensorflow as tf
import numpy as np

default_params = {
    'nsteps': 5,
    'gamma' : 0.99,
    'vf_coef'  : 0.5,
    'ent_coef' : 0.01,
    'grad_clip_norm' : 0.5,
    'normalize_advantage': False,
    
    'optimizer': tf.train.RMSPropOptimizer,
    'alpha'    : 0.99,
    'epsilon'  : 1e-5,
    'lr_scheduler': ConstantScheduler(7e-4) 
}

class A2C(Agent):
    def __init__(self, 
                 sess, 
                 env, 
                 config, 
                 network,
                 params = default_params):
        """
        Agent class for the A2C algorithm
        
        Note
        ----
        current implementation only supports for the discrete action space
        
        Parameters
        ----------
        actor_network: neural network
            neural network to behave as a policy
        critic_network: neural network
            neural network to act as the baseline
        """
        self.params = params
        super(A2C, self).__init__(sess, env, config, network)
        pass
    
    def build_graph(self):
        ##################
        ## Placeholders ##
        ##################
        self.obs_ph    = tf.placeholder(tf.float32, shape = self.input_shape, name ='obs')
        self.action_ph = tf.placeholder(tf.int32,   shape = [None], name = 'action')
        self.adv_ph    = tf.placeholder(tf.float32, shape = [None], name = 'advantage')
        self.target_ph = tf.placeholder(tf.float32, shape = [None], name = 'critic_target')
            
        self.learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')    
        
        #############
        ## Network ##
        #############
        self.header = self.network.build_header('header', self.obs_ph)
        self.actor  = self.network.build_tail('actor_net', 
                                              self.header, 
                                              self.action_size,
                                              normc_initializer(0.01))
        self.critic = self.network.build_tail('critic_net',
                                              self.header, 
                                              1,
                                              normc_initializer(1.0))
        self.critic = tf.reshape(self.critic, [-1])
        
        # Sample action from the action distribution
        self.sampled_action  = tf.reshape(tf.random.multinomial(self.actor, num_samples = 1), [-1])
        
        ##########
        ## Loss ##
        ##########
        # Actor Loss (Policy Gradient)
        self.neglog_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.action_ph, logits = self.actor)
        self.actor_loss = tf.reduce_mean(self.neglog_prob * self.adv_ph)
        
        # Critic Loss (MSE error)
        self.critic_loss = tf.reduce_mean(tf.square(self.critic - self.target_ph))
        
        # Entropy Loss
        self.entropy_loss = tf.reduce_mean(entropy(self.actor))
        
        self.loss = self.actor_loss + self.params['vf_coef'] * self.critic_loss - self.params['ent_coef'] * self.entropy_loss
        
        ##################
        ## Optimization ##
        ##################
        optimizer = self.params['optimizer'](learning_rate = self.learning_rate, 
                                             decay   = self.params['alpha'], 
                                             epsilon = self.params['epsilon'])
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, self.params['grad_clip_norm'])
        self.update_op = optimizer.minimize(self.loss)
    
    def build_logger(self):
        # Loss placeholder is used to simplify the logging procedure
        self.actor_loss_val  = tf.placeholder(tf.float32, name = 'actor_loss')
        self.critic_loss_val = tf.placeholder(tf.float32, name = 'critic_loss')
        self.entropy_val     = tf.placeholder(tf.float32, name = 'entropy_loss')
        self.explained_var   = tf.placeholder(tf.float32, name = 'explained_var')
        self.avg_reward = tf.placeholder(tf.float32, name = 'avg_reward')
        self.max_reward = tf.placeholder(tf.float32, name = 'max_reward')
        self.avg_length = tf.placeholder(tf.float32, name = 'avg_length')
        
        tf.summary.scalar('actor_loss',   self.actor_loss_val)
        tf.summary.scalar('critic_loss',  self.critic_loss_val)
        tf.summary.scalar('entropy_loss', self.entropy_val)
        tf.summary.scalar('explained_var',self.explained_var)
        tf.summary.scalar('avg_reward',   self.avg_reward)
        tf.summary.scalar('max_reward',   self.max_reward)
        tf.summary.scalar('avg_length',   self.avg_length)
        
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_path, self.sess.graph)
    
    def initialize_model(self):
        self.sess.run(tf.global_variables_initializer())      
    
    
    def _sample_trajectory(self):
        obs_batch, act_batch, rew_batch, done_batch = [],[],[],[]
        for _ in range(self.params['nsteps']):
            actions = self.predict(self.last_obs)
            next_obs, reward, done, info = self.env.step(actions)
            
            obs_batch.append(self.last_obs)
            act_batch.append(actions)
            rew_batch.append(reward)
            done_batch.append(done)
            # Each environment will automatically be reset at the end of the game
            self.last_obs = next_obs
        
        return np.array(obs_batch), np.array(act_batch), np.array(rew_batch), np.array(done_batch)
    
    def _compute_target(self, last_v, rew_batch, done_batch):
        """n-step return bootstrapped from the last state"""
        target = []
        R = last_v
        # Compute discounted sum of rewards in backwards
        for rew, done in zip(rew_batch[::-1], done_batch[::-1]):
            R = rew + self.params['gamma'] * R * (1 - done)
            target.append(R)
        return np.concatenate(target[::-1])
    
    def _compute_advantage(self, obs_vec, target_vec):
        v_vec = self.sess.run(self.critic, feed_dict = {self.obs_ph : obs_vec})
        adv_vec = target_vec - v_vec
        
        if self.params['normalize_advantage']:
            adv_vec = (adv_vec - np.mean(adv_vec, axis=0)) / np.std(adv_vec, axis=0)
        return adv_vec
        
        
    def simulate(self):
        pass
    
    def train(self):
        # Training is done with synchronous multi-processing
        while self.t < self.total_timesteps:
            # Collect trajectories with actor
            obs_batch, act_batch, rew_batch, done_batch = self._sample_trajectory() 
            last_v = self.sess.run(self.critic, feed_dict = {self.obs_ph : self.last_obs})
            # Compute advantage from each observation
            obs_vec = np.concatenate(obs_batch)
            act_vec = np.concatenate(act_batch)
            target_vec = self._compute_target(last_v, rew_batch, done_batch)
            adv_vec = self._compute_advantage(obs_vec, target_vec)
            # Backpropagate
            lr = self.params['lr_scheduler'].value(self.t)
            feed_dict = {self.obs_ph: obs_vec,
                         self.action_ph: act_vec,
                         self.adv_ph: adv_vec,
                         self.target_ph: target_vec,      
                         self.learning_rate: lr}
            actor_loss, critic_loss, entropy, v, _ \
                = self.sess.run([self.actor_loss, self.critic_loss, self.entropy_loss, self.critic, self.update_op],
                                 feed_dict = feed_dict)
            
            # Log the training progress
            if self.t % self.log_freq == 0:
                print('Timestep:',self.t)
                # Get rewards for recent 100 episodes
                episode_rewards = self.env.get_episode_rewards()[-100:]
                episode_lengths = self.env.get_episode_lengths()[-100:]
                if len(episode_rewards) > 0:
                    feed_dict = {self.actor_loss_val  : actor_loss,
                                 self.critic_loss_val : critic_loss,
                                 self.entropy_val     : entropy,
                                 self.avg_reward: np.mean(episode_rewards),
                                 self.max_reward: np.max(episode_rewards),
                                 self.avg_length: np.mean(episode_lengths),
                                 self.explained_var: 1 - np.var(target_vec - v) / np.var(target_vec)}
                    summary = self.sess.run(self.merged, feed_dict = feed_dict)
                    self.writer.add_summary(summary, global_step = self.t)
            self.t += self.num_actors * self.params['nsteps']

    def play(self):
        pass
     
    def predict(self, obs_t):
        return self.sess.run(self.sampled_action, feed_dict = {self.obs_ph: obs_t})
