import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers as lr
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model

from dataset_connector import load_dataset


def create_model(save_plot=False):
    INPUT_IMAGE_SHAPE = 50, 60, 3
    INPUT_AUDIO_SHAPE = 43, 20  # (time, rows, channels)
    OUTPUT_SHAPE = 25, 50, 60, 3

    ACT = 'elu'

    im_inputs = lr.Input(INPUT_IMAGE_SHAPE, name='image')
    au_inputs = lr.Input(INPUT_AUDIO_SHAPE, name='audio')

    x = lr.Conv2D(16, (3, 3), padding='same')(im_inputs)
    x = lr.BatchNormalization()(x)
    x = lr.Activation(ACT)(x)

    x = lr.Conv2D(32, (3, 3), padding='same', strides=2)(x)
    x = lr.BatchNormalization()(x)
    x = lr.Activation(ACT)(x)

    x = lr.Conv2D(64, (3, 3), padding='same', strides=2)(x)
    x = lr.BatchNormalization()(x)
    x = lr.Activation(ACT)(x)

    x = lr.Conv2D(128, (3, 3), padding='same', strides=2)(x)
    x = lr.BatchNormalization()(x)
    x = lr.Activation(ACT)(x)

    inner_shape = [1] + x.shape[1:]

    x = lr.Flatten()(x)
    flatten_len = x.shape[1]
    z = lr.LSTM(128)(au_inputs[:-1])

    x = lr.Dense(3200)(x)
    x = lr.Concatenate(-1)([x, z])

    x = lr.BatchNormalization()(x)
    x = lr.Activation(ACT)(x)
    x = lr.Dense(flatten_len)(x)
    x = lr.BatchNormalization()(x)
    x = lr.Activation(ACT)(x)

    x = lr.Reshape(inner_shape)(x)

    z = lr.LSTM(inner_shape[1] * inner_shape[2])(au_inputs[:-1])
    z = lr.Reshape(inner_shape[:3] + [1])(z)
    x = lr.Concatenate(-1)([x, z])

    x = lr.Conv3DTranspose(64, 3, strides=(3, 2, 2), padding='same')(x)
    x = lr.BatchNormalization()(x)
    x = lr.Activation(ACT)(x)

    x = lr.Conv3DTranspose(32, 3, strides=(3, 2, 2), padding='same')(x)
    x = lr.BatchNormalization()(x)
    x = lr.Activation(ACT)(x)

    x = lr.Conv3DTranspose(16, 3, strides=(3, 2, 2), padding='same')(x)
    x = lr.BatchNormalization()(x)
    x = lr.Activation(ACT)(x)

    x = lr.Conv3D(8, (3, 5, 3), padding='valid')(x)
    x = lr.BatchNormalization()(x)
    x = lr.Activation(ACT)(x)

    x = lr.Conv3D(3, (1, 3, 3), padding='valid')(x)
    x = lr.BatchNormalization()(x)
    outputs = lr.Activation('sigmoid')(x)

    model = tf.keras.Model(
        inputs={'image': im_inputs, 'audio': au_inputs},
        outputs=outputs
    )

    if save_plot:
        plot_model(model, dpi=70, show_shapes=True)

    return model