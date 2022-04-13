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
Y = model.predict({'image': testFrame[0:1], 'audio': testMfccs[0:1]})

for idx, y in enumerate(Y[0]):
    image = y * 255.
    save_image(image, f"out/{idx}.jpg")
