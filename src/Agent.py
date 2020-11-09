import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DQN import DQN

class Agent:
    """define a general Agent class, which contains the common properties and methods of a reinforcement learning
    agent.
    """

    def __init__(self, env, mem_size=int(100000), epsilon=1., discount=0.95):


        """
        constructor for the general agent class

        :param env: the environment for learning
        :param q_table_name: specify a csv file for initialise the q_table, e.g. load an existing q table;
                             if None, q_table will be uniformly initialised to 0.9.
        :param q_count_name: specify a csv file for initialise the q_count, e.g. load an existing q count;
                             if None, q_count will be uniformly initialised to 0.9.
        :param epsilon: epsilon-greedy parameter
        :param discount: reward discount rate
        """

        # define class attributes
        self.env = env
        self.state_size = int(self.env.observation_space.shape[0])
        self.action_size = int(self.env.action_space.n)

        self.epsilon = epsilon
        self.discount = discount

        self.mem_size = mem_size
        self.mem_count = 0
        self.replay_memory = np.zeros((mem_size, self.state_size*2+2), dtype=np.float32)

        self.policy_network = DQN(self.state_size, self.action_size)
        self.target_network = DQN(self.state_size, self.action_size)
        self.target_network.set_weights(self.policy_network)

    def action_epsilon_greedy(self, state):
        """given a state, select a epsilon_greedy action"""

        # state = (np.expand_dims(state, 0))
        # Q_values = self.policy_network.predict(state)[0]

        Q_values = self.policy_network.predict(state)

        ch = Q_values.argmax()
        # print(Q_values, ch)

        # evenly distributed epsilon
        weights = np.ones(self.action_size) * self.epsilon / self.action_size

        # add the remaining weight on the greedy action
        weights[ch] += 1 - self.epsilon

        # return the action
        return np.random.choice(self.action_size, p=weights)

    def remember(self, state, new_state, action, reward):

        if self.mem_count < self.mem_size:
            index = self.mem_count

        else:
            index = self.mem_count % self.mem_size

        # print(self.mem_count, index)
        self.replay_memory[index, :] = *state, *new_state, action, reward
        self.mem_count += 1

    def batch_sample(self, batch_size):
        pool_size = min(self.mem_size, self.mem_count)
        memory = self.replay_memory[:pool_size,:]
        idx = np.random.choice(pool_size, batch_size, replace=False)
        batch = memory[idx]
        return batch




        



