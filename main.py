from PIL import Image
from preprocessing.preprocess import create_dataset
from predicting.generator import Generator
import numpy as np
import cv2
import os


def change_res(source):
    os.system(f'ffmpeg -y -i {source} -vf scale=280:-1 r_{source}')


generator = Generator(
    weights_path="predicting/current_best_v02.hdf5",
    shape_predictor_path="preprocessing/video/landmark_detectors/shape_predictors/shape_predictor_5_face_landmarks.dat",
    mu_au_path="predicting/MU_AU.csv",
    std_au_path="predicting/STD_AU.csv"
)

generator.generate_video("gg.mp4", 5)
# generator.generate_video("sgwp8n.mpg", 5)
