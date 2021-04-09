from custom_losses import CustomLosses
from network_models import NetworkModels
from data_helper import DataHelper
from confguration import Config

import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


class FaceTemplateGAN:
    def test(self):
        cnf = Config()

        '''making models'''
        net_model = NetworkModels()
        model_gen = net_model.get_generator_model()

        model_disc = net_model.get_discriminator_model()
        model_disc.load_weights('./models/last_we_model_disc_.h5')
        # model_disc = tf.keras.models.load_model('./models/model_disc1999_.h5')
        '''noise'''
        test_sample = tf.random.normal([9, cnf.num_of_landmarks])
        out_fake = model_disc(test_sample)
        print('------------')

    def train(self, train_gen, train_disc):
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
        model_gen.trainable = train_gen
        # model_gen.load_weights('./models/last_we_model_gen39_.h5')

        model_disc = net_model.get_discriminator_model()
        model_disc.trainable = train_disc
        # model_disc.load_weights('./models/last_we_model_disc_.h5')

        '''optimizer'''
        opti_gen = tf.keras.optimizers.Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, decay=1e-5)
        opti_disc = tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.5, beta_2=0.999, decay=1e-6)

        '''create sample generator'''
        dhp = DataHelper()
        x_train_filenames = dhp.create_generators()

        '''create train configuration'''
        step_per_epoch = len(x_train_filenames) // cnf.batch_size
        # step_per_epoch = 5

        test_sample = tf.random.normal([9, cnf.noise_input_size])

        '''start train process'''
        train_gen = False
        train_disc = True
        model_gen.trainable = train_gen
        model_disc.trainable = train_disc

        for epoch in range(cnf.epochs):
            # self.save_sample_images(model=model_gen, epoch=epoch, test_input=test_sample, dhp=dhp)
            if (epoch + 1) % 10 == 0:
                train_gen = not train_gen
                train_disc = not train_disc

                model_gen.trainable = train_gen
                model_disc.trainable = train_disc

            print('=================================')
            print(' Generator is :' + str(train_gen))
            print(' Discriminator is :' + str(train_disc))
            print('=================================')

            for batch_index in range(step_per_epoch):
                '''creating noises'''
                noise = tf.random.normal([cnf.batch_size, cnf.noise_input_size])
                '''load annotation and images'''
                real_data = dhp.get_batch_sample(batch_index=batch_index, x_train_filenames=x_train_filenames)
                ''''''
                self.train_step(epoch=epoch, step=batch_index, real_data=real_data, model_gen=model_gen,
                                model_disc=model_disc, opti_gen=opti_gen, opti_disc=opti_disc, cnf=cnf, c_loss=c_loss,
                                noise=noise)

            '''save sample images:'''
            if (epoch + 1) % 20 == 0:
                self.save_sample_images(model=model_gen, epoch=epoch, test_input=test_sample, dhp=dhp)
            '''save weights'''
            if (epoch + 1) % 1000 == 0:
                model_gen.save('./models/model_gen' + str(epoch) + '_.h5')
                model_disc.save('./models/model_disc' + str(epoch) + '_.h5')
                model_gen.save_weights('./models/we_model_gen' + str(epoch) + '_.h5')
                model_disc.save_weights('./models/we_model_disc' + str(epoch) + '_.h5')
        '''save last weights'''
        model_gen.save('./models/model_gen_LAST.h5')
        model_disc.save('./models/model_disc_LAST.h5')

    # @tf.function
    def train_step(self, epoch, step, real_data, model_gen, model_disc, opti_gen, opti_disc, cnf, c_loss,noise):
        """the train step"""

        '''creating tape'''
        with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_disc:
            '''generate data'''
            generated_data = model_gen(noise, training=True)
            '''discriminate'''
            real_output = model_disc(real_data, training=True)
            fake_output = model_disc(generated_data, training=True)
            '''calculate losses'''
            loss_gen = c_loss.generator_loss(fake_output=fake_output)
            real_loss, fake_loss, loss_disc = c_loss.discriminator_loss(real_output=real_output, fake_output=fake_output)

            '''calculate gradient'''
            grad_gen = tape_gen.gradient(loss_gen, model_gen.trainable_variables)
            grad_disc = tape_disc.gradient(loss_disc, model_disc.trainable_variables)

            '''apply gradients to optimizers'''

            opti_gen.apply_gradients(zip(grad_gen, model_gen.trainable_variables))
            opti_disc.apply_gradients(zip(grad_disc, model_disc.trainable_variables))

        '''create output report:'''
        tf.print("->EPOCH: ", str(epoch), "->STEP: ", str(step), 'Loss_gen:', loss_gen, 'Loss_disc:', loss_disc,
                 'real_loss:', real_loss, 'fake_loss:', fake_loss)

    # def save_sample_images(self, model, epoch, test_input, dhp):
    #     predicted_data = model(test_input, training=False)
    #
    #     fig = plt.figure(figsize=(15, 15))
    #
    #     for i in range(predicted_data.shape[0]):
    #         plt.subplot(3, 3, i + 1)
    #         hm_img = np.array(predicted_data[i])
    #         # hm_img = np.array(predicted_data[i]).reshape(28,28)
    #         plt.imshow(hm_img)
    #
    #     plt.savefig('./sample_output/image_at_epoch_{:04d}.png'.format(epoch))
    #
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