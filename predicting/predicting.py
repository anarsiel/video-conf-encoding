import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from PIL import Image

from training.model import create_model
from visualizing import save_image
from preprocessing.preprocess import create_dataset


def predict(frame, mfcc, model=None, model_source=None, only_weights=True):
    # TODO feature extraction

    if model is None:
        if only_weights:
            model = create_model()

            model.compile(
                loss='mean_squared_error',
                optimizer='adam',
                metrics=['mean_squared_error']
            )

            model.load_weights(model_source)
        else:
            model.load_model

    Y = model.predict({'image': [frame], 'audio': [mfcc]})

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