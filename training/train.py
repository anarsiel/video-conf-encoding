import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from keras import backend as K

# from dataset_connector import load_dataset
from custom_dataset import get_datasets
from model import create_model
import numpy as np

VIDEO_DIR = '../dataset_small/frames'
AUDIO_DIR = '../dataset_small/mfccs'

train_dataset, test_dataset, MU_AU, STD_AU = get_datasets(VIDEO_DIR, AUDIO_DIR)

print(f"MU_AU: {MU_AU}\nSTD_AU: {STD_AU}\n")

model = create_model(save_plot=True)

mse = tf.keras.losses.MeanSquaredError(
    name='mean_squared_error'
)

adam = tf.keras.optimizers.Adam(0.1)
model.compile(
    loss=mse,
    optimizer=adam,
    metrics=['mean_squared_error']
)

checkpoint_filepath = 'best_model.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="mean_squared_error",
    mode="min",
    save_best_only=True,
    verbose=1,
    save_weights_only=True,
)

print("------TRAINING------")
model.fit(train_dataset,
          validation_data=test_dataset,
          epochs=2,
          batch_size=4,
          callbacks=[model_checkpoint_callback],
          verbose=1
          )
model.save_weights('weights.h5')

print("-------TESTING------")
score = model.evaluate(test_dataset, verbose=0)
print(score)

# model = create_model()
# model.compile(
#     loss=mse,
#     optimizer=adam,
#     metrics=['mean_squared_error']
# )
# model.load_weights(checkpoint_filepath)
# print("-------TESTING------")
# score = model.evaluate({'image': testFrame, 'audio': testMfccs}, testY, verbose=0)
# print(score)
