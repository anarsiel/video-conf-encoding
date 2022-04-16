import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from PIL import Image
import dlib
import cv2

from training.visualizing import save_image
from preprocessing.preprocess import create_dataset
from preprocessing.audio.audio_features import get_mfccs_for_precict
from preprocessing.video.landmark_detectors.common import apply_points_to_image, get_landmarks_as_points, __shape_predictor_5
from training.model import create_model
from training import dataset_connector


def predict(frame_source, audio_source, model=None, model_source=None, only_weights=True):
    # TODO feature extraction
    mfcc = get_mfccs_for_precict(audio_source)

    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor(__shape_predictor_5)

    frame = cv2.imread(frame_source)
    points, input_frame = get_mouth(frame, face_detector, landmark_detector)

    cv2.imwrite(f"a.jpg", input_frame)

    image = apply_points_to_image(frame, points)
    cv2.imwrite(f"b.jpg", image)

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
            model.load_model(model_source)

    input_frame, mfcc = prepare(input_frame, mfcc)
    Y = model.predict({'image': input_frame, 'audio': mfcc})[0]

    for idx, y in enumerate(Y):
        image = Image.fromarray(np.uint8(y * 255.))
        image.save(f"predicting/images/{idx}.jpg")

        # back_im = input_frame.copy()
        # back_im.paste(image, (130, 198))
        # back_im.save(f"images/{idx}.jpg")


def get_mouth(frame, face_detector, landmark_detector):
    points = get_landmarks_as_points(frame, face_detector, landmark_detector)

    x, y = points[-1][0], points[-1][1]
    image = frame[y:y + 50, x - 30:x + 30]

    return points, image


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


def prepare(input_frame, mfcc):
    input_frame = input_frame / 255.
    input_frame = input_frame.reshape((1, 50, 60, 3))

    mean = np.mean(mfcc)
    std = np.std(mfcc)
    mfcc = (mfcc - mean) / std
    mfcc.transpose()

    mfcc = mfcc.reshape((1, 43, 20))
    return input_frame, mfcc
