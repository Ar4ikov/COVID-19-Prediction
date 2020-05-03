# | Created by Ar4ikov
# | Время: 24.04.2020 - 11:06

from keras.models import Model, load_model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Concatenate, Flatten, Reshape, Dropout, Lambda
from keras.optimizers import RMSprop
from keras.activations import sigmoid, softmax, tanh, relu
import keras.backend as K
from numpy.random import choice
from numpy import mean


class SiameseNet:
    def __init__(self, name=None):
        if name is None:
            name = "Siamese_Net"

        self.name = name

    @staticmethod
    def convolution_block(input_layer, filter_size=(3, 3), nb_filters=256):
        conv2d_layer = Conv2D(nb_filters, filter_size, activation=relu)(input_layer)
        pool2d_layer = MaxPool2D((2, 2))(conv2d_layer)

        return pool2d_layer

    @staticmethod
    def euclidean_distance(vectors):
        x, y = vectors
        square_sum = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(square_sum, K.epsilon()))

    @staticmethod
    def euclidean_output_shape(shapes):
        shape1, shape2 = shapes
        return shape1[0], 1

    @staticmethod
    def accuracy(y_true, y_pred):
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

    @staticmethod
    def contrastive_loss(y_true, y_pred):
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    def generate_branch(self, input_shape, conv_blocks=3, branch_name=None):
        if branch_name is None:
            branch_name = choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

        input_layer = Input(shape=input_shape)

        layers = [input_layer, ]

        for i in range(conv_blocks):
            layers.append(self.convolution_block(layers[-1]))

        flatten = Flatten()(layers[-1])

        dense_1 = Dense(1024, activation=softmax)(flatten)
        dropout_1 = Dropout(0.4)(dense_1)

        output_layer = Dense(512, activation=softmax)(dropout_1)

        branch = Model(input_layer, output_layer, name=self.name + f"_branch_{branch_name}")

        return branch

    def generate_model(self, input_shape):
        input_A = Input(shape=input_shape)
        input_B = Input(shape=input_shape)

        base_network = self.generate_branch(input_shape)

        branch_A = base_network(input_A)
        branch_B = base_network(input_B)

        distance = Lambda(self.euclidean_distance, output_shape=self.euclidean_output_shape)([branch_A, branch_B])

        model = Model([input_A, input_B], distance, name=self.name)
        model.compile(optimizer=RMSprop(), metrics=[self.accuracy], loss=self.contrastive_loss)

        return model


model_cls = SiameseNet()
model = model_cls.generate_model(input_shape=(32, 32, 3))

model.summary()
