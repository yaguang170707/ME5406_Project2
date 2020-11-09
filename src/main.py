import gym
from doubleInvertedPendulum import doubleInvertedPendulum
from Agent import Agent

if __name__ == "__main__":

    # env = doubleInvertedPendulum()
    env = gym.make("CartPole-v0")

    agent = Agent(env)

    agent.train(episodes = 1000, save_every = 20)

    agent.env.close()
