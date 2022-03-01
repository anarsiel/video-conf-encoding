import librosa.display
import matplotlib.pyplot as plt


def get_mfccs(source):
    x, sr = librosa.load(source)
    return librosa.feature.mfcc(x, sr=sr)


def get_mfccs_as_plot(source):
    x, sr = librosa.load(source)
    mfccs = librosa.feature.mfcc(x, sr=sr)
    plt.figure(figsize=(15, 7))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.show()
