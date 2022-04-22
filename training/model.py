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


def create_model(save_plot=False):
    INPUT_IMAGE_SHAPE = 50, 60, 3
    INPUT_AUDIO_SHAPE = 43, 20  # (time, rows)
    OUTUT_SHAPE = 24, 50, 60, 3
    ACT = 'elu'

    im_inputs = lr.Input(INPUT_IMAGE_SHAPE, name='image')
    au_inputs = lr.Input(INPUT_AUDIO_SHAPE, name='audio')

    x = lr.Conv2D(16, (3, 3), padding='same')(im_inputs)
    x = lr.BatchNormalization()(x)
    inner_link = lr.Activation(ACT)(x)

    x = lr.Conv2D(32, (3, 3), padding='same', strides=2)(inner_link)
    x = lr.BatchNormalization()(x)
    x = lr.Activation(ACT)(x)

    x = lr.Conv2D(64, (3, 3), padding='same')(x)
    x = lr.BatchNormalization()(x)
    x = lr.Activation(ACT)(x)

    z = lr.Reshape(au_inputs.shape[1:] + [1, 1])(au_inputs)
    x = lr.Reshape([1] + x.shape[1:])(x)

    x = lr.UpSampling3D((z.shape[1] + 5, 1, 1))(x)
    z = lr.UpSampling3D((1, 1, x.shape[3] - 5))(z)

    z = lr.Conv3DTranspose(1, 5, padding='valid')(z)
    z = lr.Conv3DTranspose(1, 2, padding='valid')(z)

    x = lr.Concatenate(axis=-1)([x, z])

    x = lr.TimeDistributed(lr.Conv2DTranspose(64, 3, strides=(2, 2), padding='same'))(x)
    x = lr.BatchNormalization()(x)
    x = lr.Activation(ACT)(x)
    x = lr.SpatialDropout3D(0.2)(x)

    x = lr.ConvLSTM2D(64, 3, padding='same', return_sequences=True)(x)
    x = lr.AveragePooling3D((2, 1, 1))(x)

    inner_link = lr.Reshape([1] + inner_link.shape[1:])(inner_link)
    inner_link = lr.UpSampling3D((24, 1, 1))(inner_link)

    x = lr.Concatenate(axis=-1)([x, inner_link])

    x = lr.TimeDistributed(lr.Conv2D(32, 3, padding='same'))(x)
    x = lr.BatchNormalization()(x)
    x = lr.Activation(ACT)(x)

    x = lr.ConvLSTM2D(32, 3, padding='same', return_sequences=True)(x)
    outputs = lr.ConvLSTM2D(3, 3, padding='same', return_sequences=True,
                            activation='sigmoid')(x)

    model = tf.keras.Model(inputs={'image': im_inputs,
                                   'audio': au_inputs},
                           outputs=outputs)

    if save_plot:
        plot_model(model, dpi=70, show_shapes=True)

    return model
