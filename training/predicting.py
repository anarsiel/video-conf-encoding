import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

from dataset_connector import load_dataset
from model import create_model
from visualizing import save_image


model = create_model()

model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mean_squared_error']
)
model.load_weights('weights.h5')

trainMfccs, testMfccs, trainFrame, testFrame, trainY, testY = load_dataset("../dataset_small")
Y = model.predict({'frame': testFrame[1:2], 'mfccs': testMfccs[1:2]})
Y = Y[0][0]
Y = Y * 255.

save_image(Y, "a.jpg")