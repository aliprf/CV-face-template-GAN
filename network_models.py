from confguration import Config

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Add,MaxPooling2D, Conv2D, Flatten,\
    Conv2DTranspose, BatchNormalization, \
    Activation, GlobalAveragePooling2D, DepthwiseConv2D,\
    Dropout, ReLU, Concatenate, Input, GlobalMaxPool2D, LeakyReLU


class NetworkModels:

    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    # def _get_generator_model(self):
    #     cnf = Config()
    #     inputs = tf.keras.layers.Input(shape=[cnf.net_image_input_size, cnf.net_image_input_size, 3])
    #     down_stack = [
    #         self.downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
    #         self.downsample(128, 4),  # (bs, 64, 64, 128)
    #         self.downsample(256, 4),  # (bs, 32, 32, 256)
    #         self.downsample(512, 4),  # (bs, 16, 16, 512)
    #         self.downsample(512, 4),  # (bs, 8, 8, 512)
    #         self.downsample(512, 4),  # (bs, 4, 4, 512)
    #         self.downsample(512, 4),  # (bs, 2, 2, 512)
    #         self.downsample(512, 4),  # (bs, 1, 1, 512)
    #     ]
    #     up_stack = [
    #         self.upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
    #         self.upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
    #         self.upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
    #         self.upsample(512, 4),  # (bs, 16, 16, 1024)
    #         self.upsample(256, 4),  # (bs, 32, 32, 512)
    #         self.upsample(128, 4),  # (bs, 64, 64, 256)
    #         self.upsample(64, 4),  # (bs, 128, 128, 128)
    #     ]
    #
    #     initializer = tf.random_normal_initializer(0., 0.02)
    #     last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
    #                                            strides=2,
    #                                            padding='same',
    #                                            kernel_initializer=initializer,
    #                                            activation='tanh')  # (bs, 256, 256, 3)
    #
    #     x = inputs
    #     # Downsampling through the model
    #     skips = []
    #     for down in down_stack:
    #         x = down(x)
    #         skips.append(x)
    #     skips = reversed(skips[:-1])
    #     # Upsampling and establishing the skip connections
    #     for up, skip in zip(up_stack, skips):
    #         x = up(x)
    #         x = tf.keras.layers.Concatenate()([x, skip])
    #     x = last(x)
    #     return tf.keras.Model(inputs=inputs, outputs=x)

    # def get_discriminator_model(self):
    #     initializer = tf.random_normal_initializer(0., 0.02)
    #
    #     inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    #     tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    #
    #     x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)
    #
    #     down1 = self.downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    #     down2 = self.downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    #     down3 = self.downsample(256, 4)(down2)  # (bs, 32, 32, 256)
    #
    #     zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    #     conv = tf.keras.layers.Conv2D(512, 4, strides=1,
    #                                   kernel_initializer=initializer,
    #                                   use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)
    #
    #     batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    #
    #     leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    #
    #     zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
    #
    #     last = tf.keras.layers.Conv2D(1, 4, strides=1,
    #                                   kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)
    #
    #     return tf.keras.Model(inputs=[inp, tar], outputs=last)

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
        x = Dropout(.5)(x)

        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x_2 = LeakyReLU()(x)
        x = Dropout(.5)(x_2)

        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Add()([x, x_2])
        x = Dropout(.5)(x)

        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x_3 = LeakyReLU()(x)
        x = Dropout(.5)(x_3)

        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Add()([x, x_3])
        x = Dropout(.5)(x)

        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x_3 = LeakyReLU()(x)
        x = Dropout(.5)(x_3)

        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Add()([x, x_3])
        x = Dropout(.5)(x)

        # outputs = Dense(cnf.flatten_img_size, activation='linear')(x)
        outputs = Dense(cnf.flatten_img_size, activation='tanh')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="gen_model")
        model.summary()

        model_json = model.to_json()
        with open("./model_arch/Gen_model.json", "w") as json_file:
            json_file.write(model_json)
        return model

    def get_discriminator_model(self):
        cnf = Config()

        inputs = tf.keras.Input(shape=(cnf.flatten_img_size,))
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
        x = Dropout(.5)(x)

        # outputs = Dense(1, activation='softmax')(x)
        # outputs = Dense(1, activation='sigmoid')(x)
        outputs = Dense(1)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="disc_model")
        model.summary()

        model_json = model.to_json()
        with open("./model_arch/Disc_model.json", "w") as json_file:
            json_file.write(model_json)
        return model