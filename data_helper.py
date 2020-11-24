from confguration import Config

import numpy as np
from numpy import save, load, asarray
import os
import tensorflow as tf


class DataHelper:

    def create_generators(self):
        cnf = Config()
        fn_prefix = './file_names/'
        x_trains_path = fn_prefix + 'annotations_fns.npy'

        train_filenames = self.create_annotation_name(annotation_path=cnf.annotation_path)
        save(x_trains_path, train_filenames)

        return train_filenames

    def create_annotation_name(self, annotation_path):
        filenames = []
        for file in os.listdir(annotation_path):
            if file.endswith(".npy"):
                filenames.append(str(file))
        return np.array(filenames)

    def get_batch_sample(self, batch_index, x_train_filenames):
        cnf = Config()
        pn_tr_path = cnf.annotation_path

        batch_x = x_train_filenames[batch_index * cnf.batch_size:(batch_index + 1) * cnf.batch_size]
        pn_batch = np.array([self._load_and_normalize(pn_tr_path + file_name) for file_name in batch_x])
        pn_batch = tf.cast(pn_batch, tf.float32)
        return pn_batch

    def _load_and_normalize(self, point_path):
        cnf = Config()
        annotation = load(point_path)

        """for training we dont normalize COFW"""

        '''normalize landmarks based on hyperface method'''
        width = cnf.image_input_size
        height = cnf.image_input_size
        x_center = width / 2
        y_center = height / 2
        annotation_norm = []
        for p in range(0, len(annotation), 2):
            annotation_norm.append((x_center - annotation[p]) / width)
            annotation_norm.append((y_center - annotation[p + 1]) / height)
        return annotation_norm

    def create_landmarks(self, normal_lnd):
        cnf = Config()
        normal_lnd = np.array(normal_lnd)
        # landmarks_splited = _landmarks.split(';')
        landmark_arr_x = []
        landmark_arr_y = []

        for j in range(0, len(normal_lnd), 2):
            x = cnf.image_input_size//2 - float(normal_lnd[j]) * cnf.image_input_size
            y = cnf.image_input_size//2 - float(normal_lnd[j + 1]) * cnf.image_input_size

            landmark_arr_x.append(x)
            landmark_arr_y.append(y)

        return landmark_arr_x, landmark_arr_y
