import os
import numpy as np
from QN import QN
import matplotlib as mpl
from matplotlib import animation
import matplotlib.pyplot as plt
import _thread
from tensorflow.keras.models import load_model


# deal with matplotlib thread warning by using use a non-interactive backend
mpl.use('Agg')


# a helper function to store env.render() to gif, code downloaded from and modified base on
# https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
def save_frames_as_gif(frames, episode, path='./gif/'):
    _ = plt.figure(figsize=(6, 5), dpi=100)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    plt.margins(0.)
    plt.tight_layout(pad=0.,)
    plt.annotate("Episode %d"%episode, (15, 40), fontsize=25)

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    filename = 'Episode_%d.gif' % episode
    anim.save(path + filename, fps=24)
    plt.close()


class Agent:
    """define a general Agent class, which contains the common properties and methods of a reinforcement learning
    agent.
    """

    def __init__(self, env, alpha = 1., layer_depth=512, layer_number=1, mem_size=100000, batch_size=64, target_update=20,
                 epsilon=0.01, epsilon_decay=0.995, discount=0.99):

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

        self.batch_size = batch_size
        self.target_update = target_update
        self.alpha = alpha

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_final = epsilon
        self.discount = discount

        self.mem_size = mem_size
        self.mem_count = 0
        self.replay_memory = np.zeros((mem_size, self.state_size * 2 + 2), dtype=np.float32)
        self.priority = np.ones(mem_size, dtype=np.float32)  # priority for experience replay

        self.policy_network = QN(self.state_size, self.action_size, layer_depth, layer_number)
        self.target_network = QN(self.state_size, self.action_size, layer_depth, layer_number)
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
        # set sample pool
        pool_size = min(self.mem_size, self.mem_count)
        memory = self.replay_memory[:pool_size, :]
        p = self.priority[:pool_size]

        # calculate sampling probability
        p = p**self.alpha
        p = p/p.sum()

        # prioritised sample
        idx = np.random.choice(pool_size, batch_size, replace=False, p=p)
        batch = memory[idx]
        p = p[idx]

        # return sample
        return batch, idx, p

    # train the agent
    def train(self, episodes=1000, save_every=20):

        # record training history
        hist = []

        for episode in range(episodes):

            # switch on/off of enabling epsilon decay
            if self.epsilon_decay > 0. and np.abs(self.epsilon_decay) < 1.:
                self.epsilon = max(self.epsilon_final, self.epsilon_decay ** episode)

            # initialise each episode
            state = self.env.reset()
            done = False
            score = 0
            t = 0
            frames = []

            # switch on/off record
            record = episode % save_every == 0
            # record the first frame
            if record:
                frames.append(self.env.render(mode="rgb_array"))

            while not done:
                # choose epsilon greedy action
                action = self.action_epsilon_greedy(state)

                old_state = state

                # march one step and receive feedback
                state, reward, done, _ = self.env.step(action)

                # update records and
                score += reward
                t += 1
                self.remember(old_state, state, action, reward)

                if t == 1000:
                    done = True

                # record frame
                if record:
                    frames.append(self.env.render(mode="rgb_array"))

                # QN batch training
                if self.mem_count > self.batch_size:
                    self.batch_training(train_type="double DQN")

            hist.append([episode, t, score, self.epsilon])
            print("%d %d %d %4f" % (episode, t, score, self.epsilon))

            # intermediate savings
            if record:
                # pass
                _thread.start_new_thread(self.record, (episode, hist, frames))

    def record(self, episode, hist, frames):
        self.policy_network.model.save("models/model_training_%d.h5" % episode)
        np.savetxt("csv/hist.csv", hist, delimiter=",")
        save_frames_as_gif(frames, episode)

    def batch_training(self, train_type="double DQN"):
        batch, batch_idx, p = self.batch_sample(self.batch_size)

        # unpack batch samples
        old_states = batch[:, :self.state_size]
        states = batch[:, self.state_size: 2 * self.state_size]
        actions = batch[:, -2].astype(int)
        rewards = batch[:, -1].astype(int)

        # define indices for later use
        index = np.arange(self.batch_size)
        terminal_states = (rewards == -100)

        # predict Q(s, a) using Q nets
        Q_values = self.policy_network.predict(old_states)

        # calculate updates based on selections
        if train_type == "DQN":
            Q_prime = self.policy_network.predict(states)
            Q_update = Q_prime.max(axis=1)

        elif train_type == "natural DQN":
            Q_prime = self.target_network.predict(states)
            Q_update = Q_prime.max(axis=1)

        else:
            Q_next = self.policy_network.predict(states)
            best_actions_next = Q_next.argmax(axis=1)
            Q_prime = self.target_network.predict(states)
            Q_update = Q_prime[index, best_actions_next]

        # clear target next Q values if it is terminal states
        Q_update[terminal_states] = 0

        # update Q for training
        Q_values[index, actions] = rewards + self.discount * Q_update

        # exploit the symmetric property of the problem
        old_states = np.concatenate((old_states, -old_states), axis=0)
        Q_values = np.concatenate((Q_values, Q_values[:, ::-1]), axis=0)
        p = np.concatenate((p/2, p/2), axis=0)

        # sample weights
        w = (1/(2*self.batch_size)*1/p)
        w = (w/w.max())**(1 - 0.5**((self.mem_count-self.batch_size)/10))

        # train the policy network
        self.policy_network.model.fit(old_states, Q_values, sample_weight=w, verbose=0)

        # update priority
        self.priority[batch_idx] = np.abs(self.policy_network.predict(old_states[:self.batch_size, :])
                                          - Q_values[:self.batch_size])[index, actions] + 0.1

        if self.mem_count % self.target_update == 0:
            self.target_network.set_weights(self.policy_network)

    def load_model(self, model):
        self.policy_network.model = load_model(model)
        self.target_network.set_weights(self.policy_network)
