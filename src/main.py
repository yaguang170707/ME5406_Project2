import gym
from DoubleInvertedPendulum import DoubleInvertedPendulum
from Agent import Agent

if __name__ == "__main__":

    env = DoubleInvertedPendulum()
    # env = gym.make("CartPole-v0")

    agent = Agent(env)
    
    # agent = agent.load_model("model_training_500.h5")

    agent.train(episodes=20000, save_every=10)

    agent.env.close()
