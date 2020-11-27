from confguration import Config

import numpy as np
from numpy import save, load, asarray
import os
import tensorflow as tf
import matplotlib.pyplot as plt


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
        pn_batch = np.array([self._load_and_create_hm(pn_tr_path + file_name) for file_name in batch_x])
        # pn_batch = np.array([self._load_and_normalize(pn_tr_path + file_name) for file_name in batch_x])
        pn_batch = tf.cast(pn_batch, tf.float32)
        return pn_batch

    def _load_and_create_hm(self, point_path):
        annotation = load(point_path)
        hm = self.generate_hm(landmarks=annotation, width=28, height=28, s=1)
        # '''test print'''
        # plt.figure()
        # plt.imshow(hm)
        # plt.savefig('hm_tes')
        hm = np.array(hm).reshape([hm.shape[0]*hm.shape[1]])
        return hm

    def _load_and_normalize(self, point_path):
        cnf = Config()
        annotation = load(point_path)

        '''test print '''
        # self.test_image_print(img_name='1', img=np.zeros([224,224,3]), landmarks=annotation)
        '''normalize landmarks based on hyperface method'''
        width = 1  # cnf.image_input_size
        height = 1  # cnf.image_input_size
        x_center = width / 2
        y_center = height / 2
        annotation_norm = []
        for p in range(0, len(annotation), 2):
            annotation_norm.append((x_center - annotation[p]) / width)
            annotation_norm.append((y_center - annotation[p + 1]) / height)

        '''denormalize for test'''
        # landmarks_x, landmarks_y = self.create_landmarks(annotation_norm)
        # plt.figure()
        # plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#000000', s=15)
        # plt.savefig('11.png')
        ''''''

        return annotation_norm

    def create_landmarks(self, normal_lnd):
        cnf = Config()
        normal_lnd = np.array(normal_lnd)
        # landmarks_splited = _landmarks.split(';')
        landmark_arr_x = []
        landmark_arr_y = []

        for j in range(0, len(normal_lnd), 2):
            x = cnf.image_input_size // 2 - float(normal_lnd[j])  # * cnf.image_input_size
            y = cnf.image_input_size // 2 - float(normal_lnd[j + 1])  # * cnf.image_input_size

            landmark_arr_x.append(x)
            landmark_arr_y.append(y)

        return landmark_arr_x, landmark_arr_y

    def __gaussian_k(self, x0, y0, sigma, width, height):
        """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
        """
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        gaus = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        gaus[gaus <= 0.01] = 0
        return gaus

    def generate_hm(self, height, width, landmarks, s=1.5):
        Nlandmarks = len(landmarks)
        hm = np.zeros((height, width, Nlandmarks // 2), dtype=np.float32)

        j = 0
        for i in range(0, Nlandmarks, 2):
            x = landmarks[i]
            y = landmarks[i + 1]

            x = int(x // 8)
            y = int(y // 8)

            hm[:, :, j] = self.__gaussian_k(x, y, s, height, width)
            j += 1
        '''fuze all layers'''
        hm = np.sum(hm, axis=-1)
        return hm

    def test_image_print(self, img_name, img, landmarks):
        plt.figure()
        plt.imshow(img)

        landmarks_x = []
        landmarks_y = []
        for i in range(0, len(landmarks), 2):
            landmarks_x.append(landmarks[i])
            landmarks_y.append(landmarks[i + 1])

        for i in range(len(landmarks_x)):
            plt.annotate(str(i), (landmarks_x[i], landmarks_y[i]), fontsize=6, color='red')

        plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#000000', s=15)
        plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#fddb3a', s=8)
        plt.savefig(img_name + '.png')
        # plt.show()
        plt.clf()
