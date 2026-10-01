"""
    File name: unet_attention.py
    Author: Corinne Dumais
    Last modified: May 2022
"""

from keras.layers import multiply
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D, BatchNormalization, \
    Activation
from keras.optimizers import Adam
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K

from Bandes_riveraines.unet import DiceLoss, Dice


def attention_gate(input_1, input_2, n_filters):
    g = Conv2D(n_filters, kernel_size=1, strides=1, padding='same')(input_1)
    x = Conv2D(n_filters, kernel_size=1, strides=1, padding='same')(input_2)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    x = multiply([x, g])
    return x


def UNetRiverBandsAttention(input_shape, num_classes, channels, l2_reg):
    reg = l2(l2_reg)

    inputs = Input(shape=input_shape)
    conv1 = Conv2D(channels, kernel_size=3, padding='same', kernel_regularizer=reg)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(channels, kernel_size=3, padding='same', kernel_regularizer=reg)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2*channels, kernel_size=3, padding='same', kernel_regularizer=reg)(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(2*channels, kernel_size=3, padding='same', kernel_regularizer=reg)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(4*channels, kernel_size=3, padding='same', kernel_regularizer=reg)(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(4*channels, kernel_size=3, padding='same', kernel_regularizer=reg)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(8*channels, kernel_size=3, padding='same', kernel_regularizer=reg)(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(8*channels, kernel_size=3, padding='same', kernel_regularizer=reg)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)

    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = attention_gate(up5, conv3, 4*channels)
    up5 = concatenate([up5, conv3], axis=3)

    conv5 = Conv2D(4*channels, kernel_size=3, padding='same', kernel_regularizer=reg)(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(4*channels, kernel_size=3, padding='same', kernel_regularizer=reg)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = attention_gate(up6, conv2, 2*channels)
    up6 = concatenate([up6, conv2], axis=3)

    conv6 = Conv2D(2*channels, kernel_size=3, padding='same', kernel_regularizer=reg)(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(2*channels, kernel_size=3, padding='same', kernel_regularizer=reg)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = attention_gate(up7, conv1, channels)
    up7 = concatenate([up7, conv1], axis=3)

    conv7 = Conv2D(channels, kernel_size=3, padding='same', kernel_regularizer=reg)(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(channels, kernel_size=3, padding='same', kernel_regularizer=reg)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    outputs = Conv2D(num_classes, kernel_size=1, activation='softmax')(conv7)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(1e-4), loss=DiceLoss, metrics=[Dice, 'accuracy'])

    return model


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


