import tensorflow as tf
import numpy as np
import cv2
import dlib
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from PIL import Image

from dataset_connector import load_dataset
from model import create_model
from visualizing import save_image
from preprocessing.preprocess import create_dataset
from preprocessing.video.landmark_detectors.common import apply_points_to_image, get_landmarks_as_points, __shape_predictor_5

model = create_model()
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mean_squared_error']
)
model.load_weights('TimeDistributedModel/TimeDistributedModel')


def get_mouth(frame, face_detector, landmark_detector):
    points = get_landmarks_as_points(frame, face_detector, landmark_detector)

    x, y = points[-1][0], points[-1][1]
    image = frame[y:y + 50, x - 30:x + 30]

    return points, image


def load_mfccs(file, MU_AU, STD_AU):
    mfcc = np.loadtxt(file)

    # mean = np.mean(mfcc)
    # std = np.std(mfcc)
    mfcc = mfcc.transpose()
    mfcc = (mfcc - MU_AU) / STD_AU

    return mfcc


frame = cv2.imread("../image_400.jpg")

MU_AU = np.loadtxt("MU_AU.csv")
STD_AU = np.loadtxt("STD_AU.csv")
input_audio = load_mfccs("unnormalized_mfcc.csv", MU_AU, STD_AU)

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor(__shape_predictor_5)
points, input_frame = get_mouth(frame, face_detector, landmark_detector)
input_frame = input_frame / 255.
cv2.imwrite('color_img.jpg', input_frame * 255)

input_frame = np.expand_dims(input_frame, axis=0)
input_audio = np.expand_dims(input_audio, axis=0)
Y = model.predict({'image': input_frame, 'audio': input_audio})
# image = Image.fromarray(np.uint8(testFrame[0] * 255))
nose = points[2]
for idx, image in enumerate(Y[0]):
    # image2 = Image.fromarray(np.uint8(image * 255.))
    # image2.save(f"out/{idx}.jpg")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    overlay = np.copy(frame)
    # overlay[198:248, 130:190] = image * 255.
    overlay[nose[1]:nose[1] + 50, nose[0]-30:nose[0] + 30] = np.uint8(image * 255.)
    cv2.imwrite(f"out/{idx}.jpg", overlay)

    # back_im = input_frame.copy()
    # back_im.paste(image, (130, 198))
    # back_im.save(f"out/{idx}.jpg")
    # save_image(im, f"out-old/{idx}.jpg")


# def images2video():
#     image_folder = 'images'
#     video_name = 'video.avi'
#
#     images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
#     frame = cv2.imread(os.path.join(image_folder, images[0]))
#     height, width, layers = frame.shape
#
#     video = cv2.VideoWriter(video_name, 0, 1, (width, height))
#
#     for image in images:
#         video.write(cv2.imread(os.path.join(image_folder, image)))
#
#     cv2.destroyAllWindows()
#     video.release()