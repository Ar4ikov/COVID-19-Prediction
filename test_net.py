# | Created by Ar4ikov
# | Время: 24.04.2020 - 18:25

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.activations import softmax, tanh, relu
from keras.optimizers import RMSprop
from keras.metrics import accuracy


def convolution_network(input_shape):
    input_layer = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), activation=relu, padding="same")(input_layer)
    x = MaxPool2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation=relu, padding="same")(x)
    x = Conv2D(128, (3, 3), activation=relu, padding="same")(x)
    x = MaxPool2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation=relu, padding="same")(x)
    x = Conv2D(256, (3, 3), activation=relu, padding="same")(x)
    x = Conv2D(256, (3, 3), activation=relu, padding="same")(x)
    x = MaxPool2D((2, 2))(x)

    x = Conv2D(512, (3, 3), activation=relu, padding="same")(x)
    x = Conv2D(512, (3, 3), activation=relu, padding="same")(x)
    x = Conv2D(512, (3, 3), activation=relu, padding="same")(x)
    x = Conv2D(512, (3, 3), activation=relu, padding="same")(x)
    x = MaxPool2D((2, 2))(x)

    x = Conv2D(512, (3, 3), activation=relu, padding="same")(x)
    x = Conv2D(512, (3, 3), activation=relu, padding="same")(x)
    x = Conv2D(512, (3, 3), activation=relu, padding="same")(x)
    x = Conv2D(512, (3, 3), activation=relu, padding="same")(x)
    x = MaxPool2D((2, 2))(x)

    x = Flatten()(x)

    dense_1 = Dense(2048, activation=relu)(x)
    dropout_1 = Dropout(.4)(dense_1)

    dense_2 = Dense(1024, activation=relu)(dropout_1)
    dropout_2 = Dropout(.4)(dense_2)

    output_layer = Dense(100, activation=softmax)(dropout_2)

    model = Model(input_layer, output_layer, name="CIFAR100")
    model.compile(optimizer=RMSprop(), loss="categorical_crossentropy", metrics=[accuracy])

    return model
