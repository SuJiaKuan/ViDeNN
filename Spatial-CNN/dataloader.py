import math
import os
import random

import cv2
import numpy as np

from utilis import get_imagenames


class TrainLoader(object):

    def __init__(self, data_dir, batch_size):
        self._data_dir = data_dir
        self._batch_size = batch_size

        # Collect noisy images (patches).
        self._filenames_noisy = get_imagenames(
            os.path.join(self._data_dir, 'before'),
        )
        # Collect clean images (patches).
        self._filenames_clean = get_imagenames(
            os.path.join(self._data_dir, 'after'),
        )

        assert len(self._filenames_noisy) == len(self._filenames_clean), \
            'Number of patches are not equal for noisy and clean pair'

    def __len__(self):
        return math.ceil(len(self._filenames_noisy) / self._batch_size)

    def __iter__(self):
        self._iter_idx = 0

        # Take pairs of noisy and clean images, and make a random shuffle.
        filename_pairs = list(zip(
            self._filenames_noisy,
            self._filenames_clean,
        ))
        random.shuffle(filename_pairs)

        # If the last batch size is not enough, pick some samples randomly and
        # pad it.
        if len(filename_pairs) % self._batch_size:
            num_pad = \
                self._batch_size - (len(filename_pairs) % self._batch_size)
            filename_pairs += random.sample(filename_pairs, num_pad)

        self._filename_pairs = filename_pairs

        return self

    def __next__(self):
        if self._iter_idx < len(self):
            # Load a batch of data, including noisy and clean images.
            start_idx = self._batch_size * self._iter_idx
            end_idx = self._batch_size * (self._iter_idx + 1)
            data_noisy, data_clean = self._load_data_pair(
                self._filename_pairs[start_idx:end_idx],
            )

            self._iter_idx += 1

            return data_noisy, data_clean
        else:
            raise StopIteration

    def _load_data_pair(self, filename_pairs):
        data_noisy = []
        data_clean = []
        for filename_noisy, filename_clean in filename_pairs:
            data_noisy.append(cv2.imread(filename_noisy))
            data_clean.append(cv2.imread(filename_clean))

        return np.array(data_noisy), np.array(data_clean)


def load_test_imagenames(data_dir):
    filenames_noisy = get_imagenames(os.path.join(data_dir, 'before'))
    filenames_clean = get_imagenames(os.path.join(data_dir, 'after'))

    return filenames_noisy, filenames_clean
