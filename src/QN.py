import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import copy


def constructNN(input_size, output_size, layer_depth, layer_number):
    model = keras.Sequential()

    model.add(layers.Dense(layer_depth, input_shape=(input_size, ), activation='relu'))
    for i in range(layer_number):
        model.add(layers.Dense(layer_depth, activation='relu'))

    model.add(layers.Dense(output_size))

    model.compile(optimizer=keras.optimizers.Adam(lr=0.01),
                  loss="mean_squared_error",
                  metrics=['accuracy'])

    # print(model.summary())

    return model


class QN():
    def __init__(self, input_size, output_size, layer_depth, layer_number):
        self.model = constructNN(input_size, output_size, layer_depth, layer_number)

    def predict(self, state):
        if len(state.shape) == 1:
            state = (np.expand_dims(state, 0))
        Q_values = self.model.predict(state)
        return Q_values

    def set_weights(self, new_model):
        self.model.set_weights(new_model.model.get_weights())
















