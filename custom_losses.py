
import numpy as np
import tensorflow as tf


class CustomLosses:

    def discriminator_loss(self, real_output, fake_output, epoch):
        """"""
        '''create loss obj'''
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        '''calculate losses'''
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        '''create total los'''
        # if epoch < 30:
        #     total_loss = real_loss
        # else:
        #     total_loss = real_loss + fake_loss
        total_loss = real_loss + fake_loss
        return real_loss, fake_loss, total_loss

    def generator_loss(self, fake_output):
        """we consider the generated results as FAKE and update the network accordingly"""

        '''create loss obj'''
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        return loss