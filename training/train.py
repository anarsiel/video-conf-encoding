import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from keras import backend as K

from dataset_connector import load_dataset
from model import create_model
import numpy as np


model = create_model()

adam = tf.keras.optimizers.Adam(0.1)
model.compile(
    loss='mean_squared_error',
    optimizer=adam,
    metrics=['accuracy']
)

trainMfccs, testMfccs, trainFrame, testFrame, trainY, testY = load_dataset("../dataset_s1")

checkpoint_filepath = 'tmp'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    verbose=1,
    save_weights_only=True,
    save_freq=50
)

old = np.array(model.layers[-2].weights)
print("------TRAINING------")
model.fit(
    {'image': trainFrame, 'audio': trainMfccs},
    trainY,
    epochs=3,
    batch_size=4,
    # callbacks=[model_checkpoint_callback]
)
nnew = np.array(model.layers[-2].weights)

print(nnew - old)

model.save_weights('weights.h5')

print("-------TESTING------")
score = model.evaluate({'image': testFrame, 'audio': testMfccs}, testY, verbose=0)
print(score)
