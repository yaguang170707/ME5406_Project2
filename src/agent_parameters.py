# This is the input file for setting up the parameters of the DQN agent.

# number of perceptron in each hidden layer
LAYER_DEPTH = 128

# number of hidden layers
LAYER_NUMBERS = 1

# size of replay memory
MEM_SIZE = 1000000

# Size of minibatch for DQN training
BATCH_SIZE = 256

# training interval for updating target network
TARGET_UPDATE = 500

# initial value for exploration parameter
EPSILON_INIT = 1.0

# epsilon decay rate
EPSILON_DECAY = 0.99

# final value for exploration parameter
EPSILON_FINAL = 0.001

# max length of an episode
MAX_LENGTH = 1000

# reward discount rate
DISCOUNT = 0.99

# exponent of the priorities as in the prioritised experience replay
ALPHA = 0.

# initial portion of failed experience kept in the training minibatch
FAIL_PERCENT_INIT = 0.05

# final portion of failed experience kept in the training minibatch
FAIL_PERCENT_FINAL = 0.5
