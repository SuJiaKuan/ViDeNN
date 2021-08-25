import math
import os
import random

import cv2
import numpy as np

from utilis import get_imagenames
from utilis import load_pickle


class TrainLoader(object):

    def __init__(self, data_dir, batch_size, weight_min=1, weight_max=20000):
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._weight_min = weight_min
        self._weight_max = weight_max

        self._dir_noisy = os.path.join(self._data_dir, 'before')
        self._dir_clean = os.path.join(self._data_dir, 'after')

        # Load filenames for both noisy and clean images (patches).
        self._filenames = load_pickle(
            os.path.join(self._data_dir, 'filenames.pickle'),
        )

        # Load image pair difference values.
        self._diff_values = np.load(
            os.path.join(self._data_dir, 'diff_values.npy'),
        )
        # Define the weights for sampling.
        self._weights = np.clip(
            math.e ** self._diff_values,
            self._weight_min,
            self._weight_max,
        )

    def __len__(self):
        return math.ceil(len(self._filenames) / self._batch_size)

    def __iter__(self):
        self._iter_idx = 0

        return self

    def __next__(self):
        if self._iter_idx < len(self):
            # Weighted random sample.
            filenames = random.choices(
                self._filenames,
                weights=self._weights,
                k=self._batch_size,
            )

            # Load a batch of data, including noisy and clean images.
            data_noisy, data_clean = self._load_data_pair(filenames)

            self._iter_idx += 1

            return data_noisy, data_clean
        else:
            raise StopIteration

    def _load_data_pair(self, filenames):
        data_noisy = []
        data_clean = []
        for filename in filenames:
            filename_noisy = os.path.join(self._dir_noisy, filename)
            filename_clean = os.path.join(self._dir_clean, filename)
            data_noisy.append(cv2.imread(filename_noisy))
            data_clean.append(cv2.imread(filename_clean))

        return np.array(data_noisy), np.array(data_clean)


def load_test_imagenames(data_dir):
    filenames_noisy = get_imagenames(os.path.join(data_dir, 'before'))
    filenames_clean = get_imagenames(os.path.join(data_dir, 'after'))

    return filenames_noisy, filenames_clean
