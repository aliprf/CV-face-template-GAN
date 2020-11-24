from confguration import Config

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Conv2DTranspose, BatchNormalization, \
    Activation, GlobalAveragePooling2D, DepthwiseConv2D, Dropout, ReLU, Concatenate, Input, GlobalMaxPool2D, LeakyReLU


class NetworkModels:
    def get_generator_model(self):
        cnf = Config()

        model = tf.keras.Sequential()
        model.add(Dense(cnf.num_of_landmarks, use_bias=False, input_shape=(cnf.noise_input_size,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(.3))
        model.add(Dense(cnf.num_of_landmarks))
        model.summary()

        model_json = model.to_json()
        with open("./model_arch/Gen_model.json", "w") as json_file:
            json_file.write(model_json)
        return model

    def get_discriminator_model(self):
        cnf = Config()
        initializer = tf.random_normal_initializer(0., 0.02)

        inputs = tf.keras.Input(shape=(cnf.num_of_landmarks,))
        x = Dense(cnf.num_of_landmarks, kernel_initializer=initializer, use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(128, kernel_initializer=initializer, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(128, kernel_initializer=initializer, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(256, kernel_initializer=initializer, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(256, kernel_initializer=initializer, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(512, kernel_initializer=initializer, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(.3)(x)
        outputs = Dense(1, activation='sigmoid', kernel_initializer=initializer, use_bias=False)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="disc_model")
        model.summary()

        model_json = model.to_json()
        with open("./model_arch/Disc_model.json", "w") as json_file:
            json_file.write(model_json)
        return model