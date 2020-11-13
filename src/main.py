import gym
from doubleInvertedPendulum import doubleInvertedPendulum
from Agent_PER import Agent

if __name__ == "__main__":

    env = doubleInvertedPendulum()
    # env = gym.make("CartPole-v0")

    agent = Agent(env)
    
    # agent = agent.load_model("model_training_500.h5")

    agent.train(episodes = 2000, save_every = 10)

    agent.env.close()
