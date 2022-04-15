import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from PIL import Image

from dataset_connector import load_dataset
from model import create_model
from visualizing import save_image
from preprocessing.preprocess import create_dataset

# a, b = create_dataset(f"../predict", "../dataset_predict")
# print(a, b)

model = create_model()

model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mean_squared_error']
)
model.load_weights('weights.h5')


trainMfccs, testMfccs, trainFrame, testFrame, trainY, testY = load_dataset("../dataset_predict")
Y = model.predict({'image': testFrame[0:1], 'audio': testMfccs[0:1]})

input_frame = Image.open("face2.jpg")
for idx, y in enumerate(Y[0]):
    image = Image.fromarray(np.uint8(y * 255.))

    back_im = input_frame.copy()
    back_im.paste(image, (130, 198))
    back_im.save(f"out-old2/{idx}.jpg")
    # save_image(im, f"out-old/{idx}.jpg")


def images2video():
    image_folder = 'images'
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()