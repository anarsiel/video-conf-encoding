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

    frames = np.zeros(shape=(len(files), 24, 50, 60, 3), dtype=float)  # 25 - число кадров, 9000 = 50 * 60 * 3 - кадр
    mfccs = np.zeros(shape=(len(files), 20, 43), dtype=float)  # 24 * 9000
    input_frames = np.zeros(shape=(len(files), 50, 60, 3), dtype=float)
    for idx, file in enumerate(sorted(files)):
        filename = file.split('.')[0]

        file_mfccs = load_mfccs(f"{mfccs_dir}/{file}")
        file_frames = load_frames(f"{frames_dir}/{filename}")

        input_frames[idx], frames[idx] = file_frames[0], file_frames[1:]
        mfccs[idx] = file_mfccs

    train_size = int(len(files) * for_train)

    trainMfccs, testMfccs = mfccs[:train_size], mfccs[train_size:]
    trainFrame, testFrame = input_frames[:train_size], input_frames[train_size:]
    trainY, testY = frames[:train_size], frames[train_size:]

    return trainMfccs, testMfccs, trainFrame, testFrame, trainY, testY


def load_mfccs(file):
    mfcc = np.loadtxt(file)

    mean = np.mean(mfcc)
    std = np.std(mfcc)

    mfcc = (mfcc - mean) / std
    return mfcc


def load_frames(source_dir):
    files = [file for file in os.listdir(source_dir) if check_frame_file(file)]

    all_frames = np.zeros(shape=(25, 50, 60, 3))
    for idx, file in enumerate(files):
        image = Image.open(f"{source_dir}/{file}")
        image = np.array(image)
        image = image / 255.

        all_frames[idx] = image

    return all_frames


def check_mfcc_file(file):
    elements = file.split('.')
    return len(elements) == 2 and elements[-1] == 'csv'


def check_frame_file(file):
    elements = file.split('.')
    return len(elements) == 2 and elements[-1] == 'jpg'
