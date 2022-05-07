# from predicting.predicting import predict
from PIL import Image
from preprocessing.preprocess import create_dataset
from predicting.generator import Generator
import numpy as np
import cv2
import os

# s, bf = create_dataset(f"dataset_small", "alksd")

# image = Image.open('predicting/face2.jpeg')
# image.thumbnail((360, 360))
# image.save('image_400.jpg')
#
# # predict("predicting/face.jpg", "predicting/University_ITMO.mp3", model_source="predicting/weights-2.h5", only_weights=True)
# predict(
#     "predicting/face.jpg",
#     "predicting/University_ITMO.mp3",
#     model_source="predicting/TimeDistributedModel/TimeDistributedModel",
#     only_weights=True
# )


def change_res(source):
    os.system(f'ffmpeg -y -i {source} -vf scale=280:-1 r_{source}')


# change_res("aa.mp4")

generator = Generator(
    weights_path="predicting/current_best.hdf5",
    shape_predictor_path="preprocessing/video/landmark_detectors/shape_predictors/shape_predictor_5_face_landmarks.dat",
    mu_au_path="predicting/MU_AU.csv",
    std_au_path="predicting/STD_AU.csv"
)

# generator.generate_video("resources/videos/sgwp8n.mpg", 5)
generator.generate_video("gg.mp4", 5)

# for idx, frame in enumerate(frames):
#     # image = Image.fromarray(np.uint8(frame))
#     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # image.save(f"out/{idx}.jpg")
#
#     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     cv2.imwrite(f"out/{idx}.jpg", frame)