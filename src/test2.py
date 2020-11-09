import gym
from doubleInvertedPendulum import doubleInvertedPendulum
import matplotlib.pyplot as plt
import numpy as np
from Agent import Agent
import sys

batch_size = 128


env_vis =[]

env = doubleInvertedPendulum()

agent = Agent(env, epsilon=0.1, discount=0.999)
# agent.policy_network.model.load_weights('model')
# agent.target_network.model.load_weights('model')


#
# sys.stdout = open("logs.txt", mode="w")
for i_episode in range(10000):
    state = agent.env.reset()
    # agent.epsilon = max(0.999**i_episode, 0.01)

    # print(agent.policy_network.evaluate_single_state(state))
#
    done = False

    score = 0

    if i_episode % 10 == 0:
        agent.policy_network.model.save("model.h5")
#
    while not done:
#         # plt.imshow(env.render(mode='rgb_array'))
#         if i_episode%10 == 0:
            # agent.env.render()
#
        action = agent.action_epsilon_greedy(state)

        old_state = state

        state, reward, done, _ = agent.env.step(action)
        score += reward
#
        agent.remember(old_state, state, action, reward)
#
        if agent.mem_count > batch_size:
            batch = agent.batch_sample(batch_size)
#
            old_states = batch[:, :6]
            states = batch[:, 6:12]
            actions = batch[:, -2].astype(int)
            rewards = batch[:, -1].astype(int)

            terminal_states = (rewards == -100)
#
            Q_values = agent.policy_network.predict(old_states)

            next_Q_values = agent.policy_network.predict(states)

            next_actions = next_Q_values.argmax(axis=1)

            new_Q_values = agent.target_network.predict(states)

            new_Q_values[terminal_states] = 0

            Q_update = new_Q_values[range(batch_size), next_actions]

            Q_values[range(batch_size), actions] = rewards + agent.discount*Q_update

            agent.policy_network.model.fit(old_states, Q_values, verbose=0)

            if agent.mem_count % 10:
                agent.target_network.set_weights(agent.policy_network)

    print("%d, %d, %.4f" % (i_episode, score, agent.epsilon))





agent.env.close()

#         print("%.2f, %.2f, %.2f, %.2f, %.2f, %.2f"%(x, x_dot, alpha*180/np.pi, alpha_dot*180/np.pi, beta*180/np.pi, beta_dot*180/np.pi))
#
# env.close()