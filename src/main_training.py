import gym
from DoubleInvertedPendulum import DoubleInvertedPendulum
from Agent import Agent

if __name__ == "__main__":

    # create an environment
    env = DoubleInvertedPendulum()
    # env = gym.make("CartPole-v0")

    # create an agent
    agent = Agent(env, 'training')

    # train the agent
    agent.train(episodes=2000, save_every=20, testing=False)

    # closing
    agent.env.close()
