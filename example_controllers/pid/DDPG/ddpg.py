"""Deep Deterministic Policy Gradient Algorithm."""
import math
import random
import numpy as np
from copy import deepcopy
import tensorflow as tf

from OU import OU
from critic_network import CriticNetwork
from actor_network import ActorNetwork
from ReplayBuffer import ReplayBuffer
from display_utils import *

# Hyper Parameters:

REPLAY_BUFFER_SIZE = 100000
REPLAY_START_SIZE = 100
BATCH_SIZE = 32
GAMMA = 0.99


class DDPG:
    """Class DDPG."""

    def __init__(self, env_name, state_dim, action_dim, save_location, reward_threshold, thresh_coin_toss):
        """Init Method."""
        self.name = 'DDPG'  # name for uploading results
        self.env_name = env_name
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_location = save_location

        # Ensure action bound is symmetric
        self.time_step = 0
        self.sess = tf.InteractiveSession()

        self.actor_network = ActorNetwork(self.sess, self.state_dim, self.action_dim)
        self.critic_network = CriticNetwork(self.sess, self.state_dim, self.action_dim)

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process noise
        self.OU = OU()
        self.visualize = []

        self.episode_printer = StructuredPrinter()
        self.step_printer = StructuredPrinter()

        # loading networks
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(save_location)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
        self.reward_threshold = reward_threshold
        self.thresh_coin_toss = thresh_coin_toss

    def train(self):
        """Train Method."""
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.getBatch(BATCH_SIZE)
        state_batch = np.asarray([data.state for data in minibatch])
        action_batch = np.asarray([data.action for data in minibatch])
        reward_batch = np.asarray([data.reward for data in minibatch])
        next_state_batch = np.asarray([data.next_state for data in minibatch])
        done_batch = np.asarray([data.done for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch, [BATCH_SIZE, self.action_dim])

        # Calculate y_batch

        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch, next_action_batch)
        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch, [BATCH_SIZE, 1])
        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch, state_batch, action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch, action_batch_for_gradients)
        self.actor_network.train(q_gradient_batch, state_batch)
        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    def saveNetwork(self, i):
        """Saver."""
        self.saver.save(self.sess, self.save_location +
                        self.env_name + 'network' + str(i) +
                        'DDPG.ckpt', global_step=self.time_step)

    def action(self, state):
        """Compute action."""
        action = self.actor_network.action(state)
        return action

    def noise_action(self, state, epsilon):
        """Select action a_t according to the current policy and noise."""
        action = self.actor_network.action(state)
        temp_action = deepcopy(action)
        self.step_printer.data["pi_TrackPos"] = action[0]
        self.step_printer.data["pi_Velocity"] = action[1]
        noise_t = np.zeros(self.action_dim)
        noise_t[0] = epsilon * self.OU.function(action[0], 0.0, 0.60, 0.80)
        noise_t[1] = epsilon * self.OU.function(action[1], 0.5, 1.00, 0.10)
        action = action + noise_t
        action[0] = np.clip(action[0], -1, 1)
        action[1] = np.clip(action[1], 0, 1)
        self.visualize.append((temp_action[1],
                               self.critic_network.q_value(
                               state.reshape((1, self.state_dim)),
                               action.reshape((1, self.action_dim)))))
        return action, self.step_printer

    def perceive(self, temp_buffer, traj_reward):
        """Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer."""
        if (traj_reward > self.reward_threshold) or\
           (self.replay_buffer.num_experiences == 0):
            self.episode_printer.data["Replay_Buffer"] = "GOOD-Added"
            # print("Adding GOOD trajectory with reward %0.2f"%traj_reward)
            for sample in temp_buffer:
                if (not (math.isnan(sample[2]))):
                    next_action = self.actor_network.target_actions(
                        sample[3].reshape((1, self.state_dim)))
                    q_value = self.critic_network.target_q(
                        sample[3].reshape((1, self.state_dim)),
                        next_action.reshape((1, self.action_dim)))
                    y = sample[2] if sample[4] else sample[2] + GAMMA * q_value
                    TD = self.critic_network.td_error(
                        sample[0].reshape((1, self.state_dim)),
                        sample[1].reshape((1, self.action_dim)),
                        np.resize(y, [1, 1]))
                    self.replay_buffer.add(sample[0], sample[1],
                                           sample[2], sample[3], sample[4], TD)
        else:
            if random.uniform(0, 1) < self.thresh_coin_toss:
                self.episode_printer.data["Replay_Buffer"] = "BAD-Added"
                # print("Adding LUCKY BAD trajectory with reward%0.2f"%traj_reward)
                for sample in temp_buffer:
                    if (not (math.isnan(sample[2]))):
                        next_action = self.actor_network.target_actions(
                            sample[3].reshape((1, self.state_dim)))
                        q_value = self.critic_network.target_q(
                            sample[3].reshape((1, self.state_dim)),
                            next_action.reshape((1, self.action_dim)))
                        y = sample[2] if sample[4] else sample[2] + GAMMA * q_value
                        TD = self.critic_network.td_error(
                            sample[0].reshape((1, self.state_dim)),
                            sample[1].reshape((1, self.action_dim)),
                            np.resize(y, [1, 1]))
                        self.replay_buffer.add(sample[0], sample[1],
                                               sample[2], sample[3], sample[4], TD)
            else:
                self.episode_printer.data["Replay_Buffer"] = "BAD-Rejected"
                # print("REJECTING BAD trajectory with reward %0.2f"%traj_reward)

        self.time_step = self.time_step + 1
        # Store transitions to replay start size then start training
        if self.replay_buffer.count() > REPLAY_START_SIZE:
            self.train()
        return self.episode_printer
