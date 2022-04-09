import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt

from dataset_connector import load_dataset


def create_model():
    input_frame = keras.Input(shape=(50, 60, 3), name='frame')
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_frame)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Reshape(target_shape=(768000,))(x)

    input_mfcc = keras.Input(shape=(20, 43, 1), name='mfccs')
    y = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(input_mfcc)
    y = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(y)
    y = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(y)
    y = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(y)
    y = layers.Reshape(target_shape=(220160,))(y)

    x = layers.concatenate([x, y])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(216000, activation='softmax')(x)
    x = layers.Reshape((24, 50, 60, 3))(x)

    outputs = x

    return keras.Model(inputs=[input_frame, input_mfcc], outputs=outputs)