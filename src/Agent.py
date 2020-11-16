import os
import numpy as np
from QN import QN
import matplotlib as mpl
from matplotlib import animation
import matplotlib.pyplot as plt
import _thread
from tensorflow.keras.models import load_model
from problem_parameters import *
from agent_parameters import *

# deal with matplotlib thread warning by using use a non-interactive backend
mpl.use('Agg')


def save_frames_as_gif(frames, episode, path):
    """
    a helper function to store env.render() to gif, code downloaded from and modified base on
    https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
    """
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


class Agent:
    """define a general Agent class, which contains the common properties and methods of a DQN reinforcement learning
    agent.
    """

    def __init__(self,
                 env,
                 name='test',
                 layer_depth=LAYER_DEPTH,
                 layer_number=LAYER_NUMBERS,
                 mem_size=MEM_SIZE,
                 batch_size=BATCH_SIZE,
                 target_update=TARGET_UPDATE,
                 epsilon_init=EPSILON_INIT,
                 epsilon_decay=EPSILON_DECAY,
                 epsilon_final=EPSILON_FINAL,
                 max_length=MAX_LENGTH,
                 discount=DISCOUNT,
                 alpha=ALPHA,
                 fail_percent_init=FAIL_PERCENT_INIT,
                 fail_percent_final=FAIL_PERCENT_FINAL):

        # define class name and create directories
        self.name = name
        self.make_directories()

        # environment parameters
        self.env = env
        self.state_size = int(self.env.observation_space.shape[0])
        self.action_size = int(self.env.action_space.n)

        # RL agent parameters
        self.epsilon_init = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_final = epsilon_final
        self.epsilon = self.epsilon_init
        self.discount = discount

        # initialise replay memory and its counter
        self.mem_size = mem_size
        self.replay_memory = np.zeros((self.mem_size, self.state_size * 2 + 3), dtype=np.float32)
        self.mem_count = 0

        # initialise priority array
        self.priority = np.ones(self.mem_size, dtype=np.float32)

        # construct policy and target networks and initialise their parameters
        self.policy_network = QN(self.state_size, self.action_size, layer_depth, layer_number)
        self.target_network = QN(self.state_size, self.action_size, layer_depth, layer_number)
        self.target_network.set_weights(self.policy_network)
        self.batch_size = batch_size
        self.target_update = target_update
        self.alpha = alpha
        self.max_length = max_length
        self.fail_percent_init = fail_percent_init
        self.fail_percent_final = fail_percent_final
        self.fail_percent = self.fail_percent_init

    def make_directories(self):
        """
        create directories for storing data
        """
        if not os.path.exists(self.name):
            os.makedirs(self.name)

        dirs = ['training', 'testing']
        sub_dirs = ['models', 'csv', 'GIFs']

        for d in dirs:
            temp = "%s/%s" % (self.name, d)
            if not os.path.exists(temp):
                os.makedirs(temp)

            for sd in sub_dirs:
                temp = "%s/%s/%s" % (self.name, d, sd)
                if not os.path.exists(temp):
                    os.makedirs(temp)

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

    def action_greedy(self, state):
        """given a state, select the greedy action"""

        # get Q(s,a)
        Q_values = self.policy_network.predict(state)

        # get best action
        action = Q_values.argmax()

        # return the action
        return action

    def remember(self, state, new_state, action, reward, done):
        """
        store the information into replay memory
        """

        # set up the index for store the memory
        if self.mem_count < self.mem_size:
            index = self.mem_count
        else:
            index = np.random.choice(self.mem_size)  # eliminate a random memory if the replay memory is full

        self.replay_memory[index, :] = *state, *new_state, action, reward, done

        # initialising the priority with the existing max
        self.priority[index] = self.priority[:min(self.mem_count + 1, self.mem_size)].max()
        self.mem_count += 1

    def batch_sample_per(self, episode):
        """
        prioritised replay memory sampling
        """
        # set sample pool
        pool_size = min(self.mem_size, self.mem_count)
        memory = self.replay_memory[:pool_size, :]

        p = np.ones(pool_size) / pool_size

        if self.alpha != 0.:
            p = self.priority[:pool_size]
            temp = p.argsort()
            p[temp] = np.arange(pool_size) + 1

            # calculate sampling probability
            p = p ** self.alpha
            p = p / p.sum()

        idx = np.random.choice(pool_size, self.batch_size, replace=False, p=p)
        batch = memory[idx]
        p = p[idx]

        # return sample
        return batch, idx, p

    def batch_sample(self, episode):
        """
        replay memory sampling, a certain percentage of failed experiences is ensured in the batch
        """
        # set sample pool
        pool_size = min(self.mem_size, self.mem_count)
        memory = self.replay_memory[:pool_size, :]

        # ensure that at least a certain percentage samples a failed experience
        temp = np.arange(pool_size)

        # ensure that recent 2 memories a selected
        if self.mem_count < self.mem_size:
            recent_idx = temp[-2:]
        else:
            recent_idx = np.array([], dtype=int)

        # sample failed experiences
        fail_batch_size = min(episode, int(self.fail_percent * self.batch_size))
        fail_idx = temp[memory[:, -1].astype('bool')]
        f_idx = np.random.choice(fail_idx, fail_batch_size, replace=False)

        # sample the rest batch
        other_idx = np.delete(temp, f_idx)
        other_idx = np.delete(other_idx, recent_idx)
        o_idx = np.random.choice(other_idx, self.batch_size-fail_batch_size-len(recent_idx), replace=False)

        # combine the samples
        idx = np.concatenate((recent_idx, f_idx, o_idx))
        batch = memory[idx]

        # fails = batch[:, -1].astype('bool')
        # print(len(batch), len(batch[fails]), len(batch[fails])/len(batch))

        # return sample
        return batch

    def train(self, episodes=1000, save_every=20, test_every=100, test_size=1000):
        """
        train the agent
        """

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

                old_state = state

                # march one step and receive feedback
                state, reward, done, _ = self.env.step(action)

                # update records
                score += reward
                t += 1
                self.remember(old_state, state, action, reward, done)
                self.remember(-old_state, -state, np.abs(self.action_size - action - 1), reward, done)

                # terminate episode if the agent survives. modify after the memory record so that only termination with
                # failure will be labeled
                if t == self.max_length and not done:
                    done = True
                    self.fail_percent = 0.5
                    self.target_update = 50000  # int(min(self.target_update+500, 50000))
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
                _thread.start_new_thread(self.record, (episode, hist, frames, "%s/%s" % (self.name, "training")))

    def test(self, name, episodes, save_every=20):
        pass

    def record(self, episode, hist, frames, DIR):
        self.policy_network.model.save("%s/models/model_training_%d.h5" % (episode, DIR))
        np.savetxt("%s/csv/hist.csv" % DIR, hist, delimiter=",")
        save_frames_as_gif(frames, episode, "%s/GIFs/" % DIR)

    def batch_training(self, episode, train_type="double DQN"):
        # batch, batch_idx, p = self.batch_sample_per(episode)
        batch = self.batch_sample(episode)

        # unpack batch samples
        old_states = batch[:, :self.state_size]
        states = batch[:, self.state_size: 2 * self.state_size]
        actions = batch[:, -3].astype(int)
        rewards = batch[:, -2].astype(int)
        terminal_states = batch[:, -1].astype('bool')

        # define indices for later use
        index = np.arange(self.batch_size)

        # predict Q(s, a) using Q nets
        q_values = self.policy_network.predict(old_states)
        # prediction = q_values.copy() when using PER

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

        # update Q for training
        q_values[index, actions] = rewards + self.discount * q_update

        # set sample weights when using PER
        # beta = min(0.4 * 1.001 ** episode, 1.)
        # w = (self.batch_size * p) ** (-beta)
        # w = w / w.max()
        # w = np.ones(self.batch_size)

        # train the policy network
        self.policy_network.model.fit(old_states, q_values, verbose=1)  # sample_weight=w when using PER

        # update priority using PER
        # prediction = self.policy_network.predict(old_states)
        # delta = np.abs(prediction - q_values)[index, actions]
        # self.priority[batch_idx] = delta
        # _thread.start_new_thread(print, (prediction.max(), prediction.min(), delta.max(), delta.min(), delta.mean()))
        # _thread.start_new_thread(print, (np.sort(w),))

        if self.mem_count % self.target_update == 0:
            self.target_network.set_weights(self.policy_network)

    def load_model(self, model):
        self.policy_network.model = load_model(model)
        self.target_network.set_weights(self.policy_network)
