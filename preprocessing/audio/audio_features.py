import os

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import warnings


def get_mfccs(source):
    x, sr = librosa.load(source)
    mfccs = librosa.feature.mfcc(x, sr=sr)

    if mfccs.shape != (20, 129):
        raise Exception

    one_third_len = mfccs.shape[1] // 3

    first, second, third = \
        mfccs[:, 0*one_third_len:1*one_third_len], \
        mfccs[:, 1*one_third_len:2*one_third_len], \
        mfccs[:, 2*one_third_len:3*one_third_len]

    return first, second, third


def get_mfcc_for_predict(source):
    x, sr = librosa.load(source)
    mfccs = librosa.feature.mfcc(x, sr=sr)

    return mfccs[:, :43]


def get_mfccs_as_plot(source):
    x, sr = librosa.load(source)
    mfccs = librosa.feature.mfcc(x, sr=sr)  # [:, :65]
    plt.figure(figsize=(15, 7))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.show()
