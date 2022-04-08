import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
from PIL import Image
import os
np.random.seed(123)


def load_dataset(source, for_train=0.8):
    frames_dir = f"{source}/frames"
    mfccs_dir = f"{source}/mfccs"

    files = [file for file in os.listdir(mfccs_dir) if check_mfcc_file(file)]

    datasetX = np.zeros(shape=(len(files), 9860), dtype=float)  # 25 - число кадров, 9000 = 50 * 60 * 3 - кадр
    datasetY = np.zeros(shape=(len(files), 216000), dtype=float)  # 24 * 9000
    for idx, file in enumerate(sorted(files)):
        filename = file.split('.')[0]

        mfccs = load_mfccs(f"{mfccs_dir}/{file}")
        frames = load_frames(f"{frames_dir}/{filename}")

        mfccs = mfccs.reshape(860)  # 9860 = 20*43 + 9000, 20 количество mfcc, 43 - количество mfcc на одну секунду видео

        X, Y = np.concatenate((mfccs, frames[0])), frames[1:].reshape(216000)

        datasetX[idx] = X
        datasetY[idx] = Y

    train_size = int(len(files) * for_train)

    trainX, testX = datasetX[:train_size], datasetX[train_size:]
    trainY, testY = datasetY[:train_size], datasetY[train_size:]

    return trainX, trainY, testX, testY


def load_mfccs(file):
    return np.loadtxt(file)


def load_frames(source_dir):
    files = [file for file in os.listdir(source_dir) if check_frame_file(file)]

    all_frames = np.zeros(shape=(25, 9000))
    for idx, file in enumerate(files):

        img = Image.open(f"{source_dir}/{file}")
        image = np.array(img)
        image = image.reshape(9000)

        all_frames[idx] = image

    return all_frames


def save_image(image, name="tmp.jpg"):
    ax = plt.subplot(3, 3, 1)
    plt.imshow(image.astype("uint8"))
    plt.axis("off")
    plt.savefig(name)


def read_image(source, image_shape=(50, 60)):
    image = tf.io.read_file(source)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.reshape(image, (1, image_shape[0] * image_shape[1], 3))  # (1, 3000, 3)
    image = tf.cast(image, tf.float32)
    image = tf.cast(image / 255., tf.float32)
    return image


def check_mfcc_file(file):
    elements = file.split('.')
    return len(elements) == 2 and elements[-1] == 'csv'


def check_frame_file(file):
    elements = file.split('.')
    return len(elements) == 2 and elements[-1] == 'jpg'


load_dataset("../dataset")