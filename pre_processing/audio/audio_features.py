import os

import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def get_mfccs_as_files(source, frames_count, dest_dir="mfccs"):
    x, sr = librosa.load(source)
    mfccs = librosa.feature.mfcc(x, sr=sr)

    chunks = get_chunks(mfccs, frames_count)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for idx, chunk in enumerate(chunks):
        np.savetxt(f"{dest_dir}/{idx}{idx+1}.csv", chunk, delimiter=",")
    return None


def get_chunks(mfccs, count):
    step = mfccs.shape[1] / count
    step_int = round(step)
    return [mfccs[:, int(i * step):int(i * step) + step_int] for i in range(0, count - 2)]


def get_mfccs_as_plot(source):
    x, sr = librosa.load(source)
    mfccs = librosa.feature.mfcc(x, sr=sr)  # [:, :65]
    plt.figure(figsize=(15, 7))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.show()
