import gym
from DoubleInvertedPendulum import DoubleInvertedPendulum
from Agent import Agent

if __name__ == "__main__":

    # create an environment
    env = DoubleInvertedPendulum()

    # create an agent
    agent = Agent(env, 'testing')

    name = "DQN"
    num = 650

    # load the trained model
    agent.load_model("/home/yaguang/PycharmProjects/ME5406_Project2/src/%s/training/models/model_training_%d_500.h5"
                     % (name, num))

    # test with certain episodes
    agent.test("final_test", episodes=100, gif_write=False, save_every=10)

    # closing
    agent.env.close()
