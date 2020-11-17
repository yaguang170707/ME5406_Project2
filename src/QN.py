import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def construct_nn(input_size, output_size, layer_depth, layer_number):
    """
    construct a NN based on the given parameters
    """

    # initialise the model
    model = keras.Sequential()

    # add input layer
    model.add(layers.Dense(input_size, input_shape=(input_size, ), activation='relu'))

    # add hidden layers
    for _ in range(layer_number):
        model.add(layers.Dense(layer_depth, activation='relu'))

    # add output layers
    model.add(layers.Dense(output_size))

    # compile the model
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="mean_squared_error",
                  metrics=['accuracy'])

    # print(model.summary())

    return model


class QN:
    """
    wrapper class for tensorflow keras sequential model
    """
    def __init__(self, input_size, output_size, layer_depth, layer_number):
        # construct the NN
        self.model = construct_nn(input_size, output_size, layer_depth, layer_number)

    def predict(self, state):
        """
        using the model for q_value evaluation with arbitrary sized input
        """
        if len(state.shape) == 1:
            state = (np.expand_dims(state, 0))
        data = self.model(state)
        return data.numpy()

    def set_weights(self, new_model):
        """
        set the weights of the model from the input model
        """
        self.model.set_weights(new_model.model.get_weights())
















