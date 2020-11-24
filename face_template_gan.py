from custom_losses import CustomLosses
from network_models import NetworkModels
from data_helper import DataHelper
from confguration import Config

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.callbacks import CSVLogger
from datetime import datetime
from sklearn.utils import shuffle
import os
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray
import os.path
import csv
from skimage.io import imread
import pickle


class FaceTemplateGAN:

    def train(self):
        """"""

        cnf = Config()
        '''create loss obj'''
        c_loss = CustomLosses()

        '''create summary writer'''
        summary_writer = tf.summary.create_file_writer(
            "./train_logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

        '''making models'''
        net_model = NetworkModels()
        model_gen = net_model.get_generator_model()
        model_disc = net_model.get_discriminator_model()

        '''optimizer'''
        opti_gen = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, decay=1e-7)
        opti_disc = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, decay=1e-7)

        '''create sample generator'''
        dhp = DataHelper()
        x_train_filenames = dhp.create_generators()

        '''create train configuration'''
        step_per_epoch = len(x_train_filenames) // cnf.batch_size
        # step_per_epoch = 5

        test_sample = tf.random.normal([9, cnf.noise_input_size])

        '''start train process'''
        for epoch in range(cnf.epochs):
            for batch_index in range(step_per_epoch):
                '''load annotation and images'''
                real_data = dhp.get_batch_sample(batch_index=batch_index, x_train_filenames=x_train_filenames)
                self.train_step(epoch=epoch, step=batch_index, real_data=real_data, model_gen=model_gen,
                                model_disc=model_disc, opti_gen=opti_gen, opti_disc=opti_disc, cnf=cnf, c_loss=c_loss)

            '''save sample images:'''
            self.save_sample_images(model=model_gen, epoch=epoch, test_input=test_sample, dhp=dhp)
            '''save weights'''
            if (epoch + 1) % 1000 == 0:
                model_gen.save_weights('./models/model_gen' + str(epoch) + '_.h5')
                model_disc.save_weights('./models/model_disc' + str(epoch) + '_.h5')
        '''save last weights'''
        model_gen.save_weights('./models/model_gen_LAST.h5')
        model_disc.save_weights('./models/model_disc_LAST.h5')

    def train_step(self, epoch, step, real_data, model_gen, model_disc, opti_gen, opti_disc, cnf, c_loss):
        """the train step"""

        '''creating noises'''
        noise = tf.random.normal([cnf.batch_size, cnf.noise_input_size])
        '''creating tape'''
        with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_disc:
            '''generate data'''
            generated_data = model_gen(noise)
            '''discriminate'''
            real_output = model_disc(real_data, training=True)
            fake_output = model_disc(generated_data, training=True)
            '''calculate lossws'''
            loss_gen = c_loss.generator_loss(fake_output=fake_output)
            real_loss, fake_loss, loss_disc = c_loss.discriminator_loss(real_output=real_output, fake_output=fake_output)

        '''calculate gradient'''
        grad_gen = tape_gen.gradient(loss_gen, model_gen.trainable_variables)
        grad_disc = tape_disc.gradient(loss_disc, model_disc.trainable_variables)

        '''apply gradients to optimizers'''
        opti_gen.apply_gradients(zip(grad_gen, model_gen.trainable_variables))
        opti_disc.apply_gradients(zip(grad_disc, model_disc.trainable_variables))

        '''create output report:'''
        tf.print("->EPOCH: ", str(epoch), "->STEP: ", str(step), 'Loss_gen:', loss_gen, 'Loss_disc:', loss_disc)

    def save_sample_images(self, model, epoch, test_input, dhp):
        predicted_data = model(test_input, training=False)

        fig = plt.figure(figsize=(15, 15))

        for i in range(predicted_data.shape[0]):
            plt.subplot(3, 3, i + 1)
            landmarks_x, landmarks_y = dhp.create_landmarks(predicted_data[i])
            plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#fddb3a', s=15)
            plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#000000', s=10)
            plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#fddb3a', s=5)

        plt.savefig('./sample_output/image_at_epoch_{:04d}.png'.format(epoch))
        # plt.show()