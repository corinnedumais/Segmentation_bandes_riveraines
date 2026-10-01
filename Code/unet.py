"""
    File name: unet.py
    Author: Corinne Dumais
    Last modified: May 2022
"""

from __future__ import division

import logging
import os
from contextlib import redirect_stdout

import numpy as np
import tensorflow as tf

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, \
    AveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.regularizers import l2
import keras.backend as K

# To avoid the display of certain warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


class UNetRiverBands:
    """
    Class representing a U-Net neural network to segment riverbands
    """

    def __init__(self, input_size, output_classes, kernel_size, channels, reg, batchnorm, drop):
        """
        Parameters:
            input_size (tuple): Size of the input data (height, width, channels)
            output_classes (int): Number of classes to segment
            kernel_size (tuple): Size of the convolutional kernels (height, width)
            channels (int): Number of channels for the first convolutional layer.
        """
        # Network parameters
        self.input_size = input_size
        self.output_classes = output_classes
        self.kernel_size = kernel_size
        self.channels = channels
        self.batchnorm = batchnorm

        # Parameters for the convolutional layers
        self.init = 'he_normal'
        self.act = 'relu'
        self.reg = l2(reg)
        self.drop = drop

        # Set optimizer and loss function
        self.optimizer = Adam(learning_rate=1e-4)  # 1e-5 for batch size 32 and 128
        self.loss = DiceLoss

    def Conv2D_block(self, input_tensor, channels, kernel_size, kernel_initializer, padding='same'):
        """Function to add 2 convolutional layers"""
        x = Conv2D(channels, kernel_size=kernel_size, kernel_initializer=kernel_initializer, kernel_regularizer=self.reg, padding=padding)(input_tensor)
        if self.batchnorm:
            x = BatchNormalization()(x)
        x = Activation(self.act)(x)

        x = Conv2D(filters=channels, kernel_size=kernel_size, kernel_initializer=kernel_initializer, kernel_regularizer=self.reg, padding=padding)(x)
        if self.batchnorm:
            x = BatchNormalization()(x)
        x = Activation(self.act)(x)

        return x

    def build(self):
        """
        Method to build and compile the U-Net neural network.
        Returns:
            Model: A U-Net neural network model.
        """

        # Block 1 (input)
        inputs = Input(self.input_size)
        conv1 = self.Conv2D_block(inputs, self.channels, kernel_size=self.kernel_size, kernel_initializer=self.init)

        # Block 2
        pool1 = MaxPooling2D()(conv1)
        pool1 = Dropout(self.drop)(pool1)
        conv2 = self.Conv2D_block(pool1, 2*self.channels, kernel_size=self.kernel_size, kernel_initializer=self.init)

        # Block 3
        pool2 = MaxPooling2D()(conv2)
        pool2 = Dropout(self.drop)(pool2)
        conv3 = self.Conv2D_block(pool2, 4*self.channels, kernel_size=self.kernel_size, kernel_initializer=self.init)

        # Block 4
        pool3 = MaxPooling2D()(conv3)
        pool3 = Dropout(self.drop)(pool3)
        conv4 = self.Conv2D_block(pool3, 8*self.channels, kernel_size=self.kernel_size, kernel_initializer=self.init)

        # Block 5 (bottleneck)
        pool4 = MaxPooling2D()(conv4)
        pool4 = Dropout(self.drop)(pool4)
        conv5 = self.Conv2D_block(pool4, 16*self.channels, kernel_size=self.kernel_size, kernel_initializer=self.init)

        # Block 6
        up1 = UpSampling2D()(conv5)
        merge6 = concatenate([conv4, up1], axis=-1)
        conv6 = self.Conv2D_block(merge6, 8*self.channels, kernel_size=self.kernel_size, kernel_initializer=self.init)

        # Block 7
        up2 = UpSampling2D()(conv6)
        merge7 = concatenate([conv3, up2], axis=-1)

        conv7 = self.Conv2D_block(merge7, 4*self.channels, kernel_size=self.kernel_size, kernel_initializer=self.init)

        # Block 8
        up3 = UpSampling2D()(conv7)
        merge8 = concatenate([conv2, up3], axis=-1)
        conv8 = self.Conv2D_block(merge8, 2*self.channels, kernel_size=self.kernel_size, kernel_initializer=self.init)

        # Block 9 (output)
        up4 = UpSampling2D()(conv8)
        merge9 = concatenate([conv1, up4], axis=-1)

        conv9 = self.Conv2D_block(merge9, self.channels, kernel_size=self.kernel_size, kernel_initializer=self.init)
        conv9 = Conv2D(self.output_classes, kernel_size=(1, 1), activation='softmax', padding='same')(conv9)

        model = Model(inputs=inputs, outputs=conv9)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[Dice, 'accuracy'])

        return model

    def save_summary(self):
        """
        Method to save in a text file the summary of the model.
        """
        with open('modelsummary.txt', 'w') as f:
            with redirect_stdout(f):
                self.build().summary(line_length=145)


def categorical_crossentropy_mask(targets, inputs):
    mask = K.cast(K.not_equal(targets, 0), K.floatx())
    # mask = K.expand_dims(mask, axis=-1)  # Add an extra dimension to mask
    targets = targets * mask  # Set targets to zero where mask is zero
    inputs = inputs * mask  # Set inputs to zero where mask is zero
    crossentropy = K.categorical_crossentropy(targets, inputs)
    masked_crossentropy = crossentropy * mask  # Set crossentropy to zero where mask is zero
    return K.mean(masked_crossentropy)


def Dice(targets, inputs):
    mask = K.cast(K.not_equal(targets, 0), K.floatx())
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    mask = K.flatten(mask)
    targets = targets * mask  # Set targets to zero where mask is zero
    inputs = inputs * mask  # Set inputs to zero where mask is zero
    intersection = K.sum(K.abs(targets * inputs), axis=-1)
    denominator = K.sum(K.square(targets), -1) + K.sum(K.square(inputs), -1)
    dice = (2. * intersection) / (denominator + 1e-8)  # Add small constant to avoid division by zero
    return dice


def DiceLoss(targets, inputs):
    return 1 - Dice(targets, inputs)


weights = K.variable([0, 0.53030188202602, 1.5208194282807577, 0.7024109637803685, 70.90362917714697, 0.19163198851712188, 14.154278746144467])
def weighted_categorical_crossentropy(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss


def combined(targets, inputs):
    return DiceLoss(targets, inputs) + weighted_categorical_crossentropy(targets, inputs)
