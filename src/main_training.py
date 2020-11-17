import gym
from DoubleInvertedPendulum import DoubleInvertedPendulum
from Agent import Agent

if __name__ == "__main__":

    # create an environment
    env = DoubleInvertedPendulum()
    # env = gym.make("CartPole-v0")

    # create an agent
    agent = Agent(env, '1_DQN')

    # train the agent
    agent.train(episodes=1000, save_every=20, testing=False)

    # closing
    agent.env.close()
