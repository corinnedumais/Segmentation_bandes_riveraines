"""
    File name: dataset.py
    Author: Corinne Dumais
    Last modified: May 2022
"""

import numpy as np
from tensorflow import keras


class RiverBandsDataset(keras.utils.Sequence):
    """
    Helper class to iterate over the dataset to feed batches of data to the model.
    """

    def __init__(self, batch_size, data_img_size, target_img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.data_img_size = data_img_size
        self.target_img_size = target_img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.data_img_size + (3,), dtype="float32")

        for j, path in enumerate(batch_input_img_paths):
            img = np.load(path)
            x[j] = img

        y = np.zeros((self.batch_size,) + self.target_img_size + (7,), dtype="float32")

        for j, path in enumerate(batch_target_img_paths):
            arr = np.load(path)
            y[j] = arr

        return x, y
