import gym
from DoubleInvertedPendulum import DoubleInvertedPendulum
from Agent import Agent

if __name__ == "__main__":

    # create an environment
    env = DoubleInvertedPendulum()

    # create an agent
    agent = Agent(env, 'testing')

    # load the trained model
    agent.load_model("/home/yaguang/PycharmProjects/ME5406_Project2/src/test/training/models/model_training_260.h5")

    # test with certain episodes
    agent.test("final_test", episodes=1000, gif_write=True, save_every=100)

    # closing
    agent.env.close()
