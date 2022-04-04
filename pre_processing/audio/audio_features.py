import os

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import warnings


def save_mfccs(source, dest_dir):
    x, sr = librosa.load(source)
    mfccs = librosa.feature.mfcc(x, sr=sr)

    filename = source.split('/')[-1].split('.')[0]
    np.savetxt(f"{dest_dir}/{filename}.csv", mfccs, delimiter=",")


def get_mfccs_as_plot(source):
    x, sr = librosa.load(source)
    mfccs = librosa.feature.mfcc(x, sr=sr)  # [:, :65]
    plt.figure(figsize=(15, 7))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.show()
