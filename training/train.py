import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

from dataset_connector import load_dataset
from model import create_model

trainMfccs, testMfccs, trainFrame, testFrame, trainY, testY = load_dataset("../dataset")

# print(trainMfccs.shape)
# print(testMfccs.shape, end='\n\n')
#
# print(trainFrame.shape)
# print(testFrame.shape, end='\n\n')
#
# print(trainY.shape)
# print(testY.shape, end='\n\n')


model = create_model()
keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)

model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['accuracy']
)

checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    verbose=1,
    save_weights_only=True,
    save_freq=3
)

print("------TRAINING------")
model.fit(
    {'frame': trainFrame, 'mfccs': trainMfccs},
    trainY,
    epochs=10,
    batch_size=32,
    callbacks=[model_checkpoint_callback]
)

model.save_weights('path_to_my_model.h5')

print("-------TESTING------")
score = model.evaluate({'frame': testFrame, 'mfccs': testMfccs}, testY, verbose=0)
print(score)
