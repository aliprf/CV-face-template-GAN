from confguration import Config

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Add,MaxPooling2D, Conv2D, Flatten, Conv2DTranspose, BatchNormalization, \
    Activation, GlobalAveragePooling2D, DepthwiseConv2D, Dropout, ReLU, Concatenate, Input, GlobalMaxPool2D, LeakyReLU


class NetworkModels:
    def get_generator_model(self):
        cnf = Config()

        inputs = tf.keras.Input(shape=(cnf.noise_input_size,))
        x = Dense(cnf.noise_input_size)(inputs)
        x = BatchNormalization()(x)
        x_1 = LeakyReLU()(x)
        x = Dropout(.3)(x_1)

        x = Dense(cnf.noise_input_size)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Add()([x, x_1])
        x = Dropout(.3)(x)

        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x_1 = LeakyReLU()(x)
        x = Dropout(.3)(x_1)

        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Add()([x, x_1])
        x = Dropout(.3)(x)

        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x_2 = LeakyReLU()(x)
        x = Dropout(.3)(x_2)
        x = Dense(256)(x)

        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Add()([x, x_2])
        x = Dropout(.3)(x)

        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x_3 = LeakyReLU()(x)
        x = Dropout(.3)(x_3)

        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Add()([x, x_3])
        x = Dropout(.3)(x)

        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(.5)(x)
        outputs = Dense(cnf.num_of_landmarks, activation='linear')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="gen_model")
        model.summary()

        model_json = model.to_json()
        with open("./model_arch/Gen_model.json", "w") as json_file:
            json_file.write(model_json)
        return model

    def get_discriminator_model(self):
        cnf = Config()

        inputs = tf.keras.Input(shape=(cnf.num_of_landmarks,))
        x = Dense(cnf.num_of_landmarks)(inputs)
        x = BatchNormalization()(x)
        x_1 = LeakyReLU()(x)
        x = Dropout(.3)(x_1)

        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(.3)(x)

        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x_2 = LeakyReLU()(x)
        x = Dropout(.3)(x_2)

        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Add()([x, x_2])
        x = Dropout(.3)(x)

        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x_3 = LeakyReLU()(x)
        x = Dropout(.3)(x_3)

        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Add()([x, x_3])
        x = Dropout(.3)(x)

        # outputs = Dense(1, activation='sigmoid')(x)
        outputs = Dense(1)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="disc_model")
        model.summary()

        model_json = model.to_json()
        with open("./model_arch/Disc_model.json", "w") as json_file:
            json_file.write(model_json)
        return model