import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import datetime


def constructNN(input_size, output_size, layer_depth, layer_number):
    model = keras.Sequential()

    model.add(layers.Dense(input_size, input_shape=(input_size, ), activation='relu'))
    for _ in range(layer_number):
        model.add(layers.Dense(layer_depth, activation='relu'))

    def custom_activation(x):
        return tf.keras.activations.relu(x)-100

    model.add(layers.Dense(output_size))#, activation=custom_activation)) #

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="mean_squared_error",
                  metrics=['accuracy'])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # print(model.summary())

    return model, tensorboard_callback


class QN:
    def __init__(self, input_size, output_size, layer_depth, layer_number):
        self.model, self.tensorboard_callback = constructNN(input_size, output_size, layer_depth, layer_number)

    def predict(self, state):
        if len(state.shape) == 1:
            state = (np.expand_dims(state, 0))
        data = self.model(state)
        return data.numpy()

    def set_weights(self, new_model):
        self.model.set_weights(new_model.model.get_weights())
















