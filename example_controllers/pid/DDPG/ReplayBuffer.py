"""Prioritized experience replay buffer."""
from queue import PriorityQueue
from madras_datatypes import Madras
import random

madras = Madras()


class experience(object):
    """Experience class format."""

    def __init__(self, state, action, reward, next_state, done, TD):
        """Init Method"""
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.TD = TD

    def __gt__(self, other):
        """Redifining >."""
        return (self.TD > other.TD)

    def __lt__(self, other):
        """Redifining <."""
        return (self.TD < other.TD)

    def __eq__(self, other):
        """Redifining ==."""
        return (self.TD == other.TD)

class ReplayBuffer(object):
    """Prioritized Experiance replay."""

    def __init__(self, buffer_size):
        """Init Method."""
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = PriorityQueue()
        self.mean_reward = 0.0

    def getBatch(self, batch_size):
        """Sampling."""
        if self.num_experiences < batch_size:
            return random.sample(list(self.buffer.queue), self.num_experiences)
        else:
            return random.sample(list(self.buffer.queue), batch_size)

    def size(self):
        """Get size."""
        return self.buffer_size

    def add(self, state, action, reward, new_state, done, TD):
        """Add Experience."""
        exp = experience(state, action, reward, new_state, done, TD)
        if self.num_experiences < self.buffer_size:
            self.buffer.put(exp)
            self.num_experiences += 1
        else:
            self.buffer.get()
            self.buffer.put(exp)
        self.mean_reward = ((self.mean_reward * (self.num_experiences - 1)) +
                            reward) / madras.floatX(self.num_experiences)

    def count(self):
        """If buffer is full.

        Return buffer size otherwise, return experience counter.
        """
        return self.num_experiences

    def getMeanReward(self):
        """Get Mean Reward."""
        return self.mean_reward
