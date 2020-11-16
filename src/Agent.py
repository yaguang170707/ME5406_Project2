import os
import numpy as np
from QN import QN
import matplotlib as mpl
from matplotlib import animation
import matplotlib.pyplot as plt
import _thread
from tensorflow.keras.models import load_model
from problem_parameters import *

# deal with matplotlib thread warning by using use a non-interactive backend
mpl.use('Agg')


# a helper function to store env.render() to gif, code downloaded from and modified base on
# https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
def save_frames_as_gif(frames, episode, path='./gif/'):
    _ = plt.figure(figsize=(6, 5), dpi=100)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    plt.margins(0.)
    plt.tight_layout(pad=0., )
    plt.annotate("Episode %d" % episode, (15, 40), fontsize=25)

    def animate(i):
        patch.set_data(frames[i])

    t = len(frames)
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=t, interval=50)
    filename = 'Episode_%d_%d.gif' % (episode, t - 1)
    anim.save(path + filename, fps=24)
    plt.close()


# simple helper function to compute the sum of a power series
def q_max(discount, length):
    a = np.ones(length) + discount
    b = np.arange(length)
    y = np.power(a, b)
    return y.sum()


class Agent:
    """define a general Agent class, which contains the common properties and methods of a reinforcement learning
    agent.
    """

    def __init__(self,
                 env,
                 alpha=0.,
                 layer_depth=128,
                 layer_number=1,
                 mem_size=1000000,
                 batch_size=256,
                 target_update=500,
                 epsilon=0.001,
                 epsilon_decay=0.99,
                 max_episode=1000,
                 discount=0.99):

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
        self.max_episode = max_episode
        self.fail_percent = 0.05

        self.mem_size = mem_size
        self.mem_count = 0
        self.replay_memory = np.zeros((mem_size, self.state_size * 2 + 3), dtype=np.float32)

        self.Q_max = q_max(self.discount, self.max_episode)
        self.priority = np.ones(mem_size, dtype=np.float32)  # *self.Q_max  # priority for experience replay

        self.policy_network = QN(self.state_size, self.action_size, layer_depth, layer_number)
        self.target_network = QN(self.state_size, self.action_size, layer_depth, layer_number)
        self.target_network.set_weights(self.policy_network)

    def action_epsilon_greedy(self, state):
        """given a state, select a epsilon_greedy action"""

        # get Q(s,a)
        Q_values = self.policy_network.predict(state)

        # get best action
        ch = Q_values.argmax()

        # evenly distributed epsilon
        weights = np.ones(self.action_size) * self.epsilon / self.action_size

        # add the remaining weight on the greedy action
        weights[ch] += 1 - self.epsilon

        # choose action
        action = np.random.choice(self.action_size, p=weights)

        # check if the chosen one is the best
        is_best = action == ch

        # return the action and the boolean
        return action, is_best

    def remember(self, state, new_state, action, reward, done):

        if self.mem_count < self.mem_size:
            index = self.mem_count

        else:
            # index = self.mem_count % self.mem_size
            index = np.random.choice(self.mem_size)

        # print(self.mem_count, index)
        self.replay_memory[index, :] = *state, *new_state, action, reward, done
        self.priority[index] = self.priority[:min(self.mem_count + 1, self.mem_size)].max()
        self.mem_count += 1

    # def batch_sample_per(self, episode, percent=0.2):
    #     # set sample pool
    #     pool_size = min(self.mem_size, self.mem_count)
    #     memory = self.replay_memory[:pool_size, :]
    #
    #     p = np.ones(pool_size) / pool_size
    #
    #     if self.alpha != 0.:
    #         p = self.priority[:pool_size]
    #         temp = p.argsort()
    #         p[temp] = np.arange(pool_size) + 1
    #
    #         # calculate sampling probability
    #         p = p ** self.alpha
    #         p = p / p.sum()
    #
    #     # prioritised sample and make sure the sample contains certain amount of failure experience
    #     temp = np.arange(pool_size)
    #
    #     fail_batch_size = min(episode, int(percent * self.batch_size))
    #     fail_idx = temp[memory[:, -1].astype('bool')]
    #     fail_memory = memory[fail_idx]
    #     fail_p = p[fail_idx]
    #     fails_number = len(fail_memory)
    #
    #     f_idx = np.random.choice(fails_number, fail_batch_size, replace=False, p=fail_p / fail_p.sum())
    #     batch_fail = fail_memory[f_idx]
    #     p_fail = fail_p[f_idx]/fail_p.sum() * fail_batch_size / self.batch_size
    #     idx_fail = fail_idx[f_idx]
    #
    #     other_batch_size = self.batch_size - fail_batch_size
    #     other_idx = np.delete(temp, idx_fail)
    #     other_memory = memory[other_idx]
    #     other_p = p[other_idx]
    #     other_number = len(other_memory)
    #
    #     o_idx = np.random.choice(other_number, other_batch_size, replace=False, p=other_p / other_p.sum())
    #     batch_other = other_memory[o_idx]
    #     p_other = other_p[o_idx]/other_p.sum() * (1 - fail_batch_size / self.batch_size)
    #     idx_other = other_idx[o_idx]
    #
    #     batch = np.concatenate((batch_fail, batch_other))
    #     p = np.concatenate((p_fail, p_other))
    #     idx = np.concatenate((idx_fail, idx_other))
    #
    #     fails = batch[:, -1].astype('bool')
    #     print(len(batch[fails]))
    #     print(p)
    #
    #     # failure_samples = 0
    #     # i = 0
    #     # while failure_samples < min(episode, int(percent * self.batch_size)):
    #     #     idx = np.random.choice(pool_size, self.batch_size, replace=False)
    #     #     temp = memory[idx, -1].astype('bool')
    #     #     failure_samples = len(temp[temp])
    #     #     i += 1
    #     #
    #     # print(failure_samples, i)
    #
    #     # batch = memory[idx]
    #     # p = p[idx]
    #
    #     # return sample
    #     return batch, idx, p

    def batch_sample(self, episode):
        # set sample pool
        pool_size = min(self.mem_size, self.mem_count)
        memory = self.replay_memory[:pool_size, :]

        # ensure that at least a certain percentage samples a failed experience
        temp = np.arange(pool_size)

        if self.mem_count < self.mem_size:
            recent_idx = temp[-2:]
        else:
            recent_idx = np.array([], dtype=int)

        fail_batch_size = min(episode, int(self.fail_percent * self.batch_size))
        fail_idx = temp[memory[:, -1].astype('bool')]
        f_idx = np.random.choice(fail_idx, fail_batch_size, replace=False)

        # sample the rest batch
        other_idx = np.delete(temp, f_idx)
        other_idx = np.delete(other_idx, recent_idx)
        o_idx = np.random.choice(other_idx, self.batch_size-fail_batch_size-len(recent_idx), replace=False)

        idx = np.concatenate((recent_idx, f_idx, o_idx))
        batch = memory[idx]

        fails = batch[:, -1].astype('bool')
        print(len(batch), len(batch[fails]), len(batch[fails])/len(batch))

        # return sample
        return batch

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
                action, is_best = self.action_epsilon_greedy(state)
                # action, is_best = self.action_boltzmann(state, episode)
                # _thread.start_new_thread(print, (action, is_best))

                old_state = state

                # march one step and receive feedback
                state, reward, done, _ = self.env.step(action)

                # if done:
                #     print(self.policy_network.predict(state))

                # update records and
                score += reward
                t += 1
                self.remember(old_state, state, action, reward, done)
                self.remember(-old_state, -state, np.abs(self.action_size - action - 1), reward, done)

                # terminate episode if the agent survives. modify after the memory record so that only termination with
                # failure will be labeled
                if t == self.max_episode and not done:
                    done = True
                    self.fail_percent = 0.5
                    self.target_update = 50000#int(min(self.target_update+500, 50000))
                    self.target_network.set_weights(self.policy_network)

                # else:
                #     self.fail_percent = 0.05

                # record frame
                if record:
                    frames.append(self.env.render(mode="rgb_array"))

                # QN batch training
                if self.mem_count >= self.batch_size:
                    self.batch_training(episode, train_type="double DQN")

            hist.append([episode, t, score, self.epsilon, is_best])
            print("%d %d %d %4f %r" % (episode, t, score, self.epsilon, is_best))

            # intermediate savings
            if record:
                # pass
                _thread.start_new_thread(self.record, (episode, hist, frames))

    def record(self, episode, hist, frames):
        self.policy_network.model.save("models/model_training_%d.h5" % episode)
        np.savetxt("csv/hist.csv", hist, delimiter=",")
        save_frames_as_gif(frames, episode)

    def batch_training(self, episode, train_type="double DQN"):
        # batch, batch_idx, p = self.batch_sample_per(episode)
        batch = self.batch_sample(episode)

        # unpack batch samples
        old_states = batch[:, :self.state_size]
        states = batch[:, self.state_size: 2 * self.state_size]
        actions = batch[:, -3].astype(int)
        rewards = batch[:, -2].astype(int)
        terminal_states = batch[:, -1].astype('bool')

        # print(terminal_states)

        # define indices for later use
        index = np.arange(self.batch_size)

        # predict Q(s, a) using Q nets
        q_values = self.policy_network.predict(old_states)
        prediction = q_values.copy()

        # calculate updates based on selections
        if train_type == "DQN":
            q_prime = self.policy_network.predict(states)
            q_update = q_prime.max(axis=1)

        elif train_type == "natural DQN":
            q_prime = self.target_network.predict(states)
            q_update = q_prime.max(axis=1)

        else:
            q_next = self.policy_network.predict(states)
            best_next_actions = q_next.argmax(axis=1)
            q_prime = self.target_network.predict(states)
            q_update = q_prime[index, best_next_actions]

        # clear target next Q values if it is terminal states
        q_update[terminal_states] = 0
        # print(index)

        # update Q for training
        q_values[index, actions] = rewards + self.discount * q_update
        # q_values = q_values.clip(0, self.Q_max)

        # exploit the symmetric property of the problem
        # old_states = np.concatenate((old_states, -old_states), axis=0)
        # q_values = np.concatenate((q_values, q_values[:, ::-1]), axis=0)
        # p = np.concatenate((p / 2, p / 2), axis=0)

        # sample weights
        # beta = min(0.4 * 1.001 ** episode, 1.)
        # w = (self.batch_size * p) ** (-beta)
        # w = w / w.max()
        # w = np.ones(self.batch_size)

        # train the policy network
        self.policy_network.model.fit(old_states, q_values, verbose=1)  # sample_weight=w , batch_size=self.batch_size
        # callbacks=[self.policy_network.tensorboard_callback])#sample_weight=w,

        # update priority
        prediction = self.policy_network.predict(old_states)
        delta = np.abs(prediction - q_values)[index, actions]
        # self.priority[batch_idx] = delta
        _thread.start_new_thread(print, (prediction.max(), prediction.min(), delta.max(), delta.min(), delta.mean()))
        # _thread.start_new_thread(print, (np.sort(w),))

        if self.mem_count % self.target_update == 0:
            self.target_network.set_weights(self.policy_network)

    def load_model(self, model):
        self.policy_network.model = load_model(model)
        self.target_network.set_weights(self.policy_network)