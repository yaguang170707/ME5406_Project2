import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import copy

layer_size = 256


def constructNN(input_size, output_size):
    model = keras.Sequential()

    model.add(layers.Dense(layer_size, input_shape=(input_size, ), activation='relu'))
    model.add(layers.Dense(layer_size, activation='relu'))
    # model.add(layers.Dense(layer_size, activation='relu'))
    model.add(layers.Dense(output_size))

    model.compile(optimizer='adam',
                  loss="mean_squared_error",
                  metrics=['accuracy'])

    # print(model.summary())

    return model


class DQN():
    def __init__(self, input_size, output_size):
        self.model = constructNN(input_size, output_size)

    def predict(self, state):
        if len(state.shape) == 1:
            state = (np.expand_dims(state, 0))
        Q_values = self.model.predict(state)
        return Q_values

    def set_weights(self, new_model):
        self.model.set_weights(new_model.model.get_weights())
















