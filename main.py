from PIL import Image
from preprocessing.preprocess import create_dataset
from predicting.generator import Generator
import numpy as np
import cv2
import os


def check_video_file(file):
    video_ext = ['mpg', 'mp4']
    elements = file.split('.')
    return elements[-1] in video_ext


generator = Generator(
    weights_path="predicting/weights.hdf5",
    shape_predictor_path="preprocessing/video/landmark_detectors/shape_predictors/shape_predictor_5_face_landmarks.dat",
    mu_au_path="predicting/MU_AU.csv",
    std_au_path="predicting/STD_AU.csv"
)

source_dir = "resources/test"
dest_dir = "generated"

files = [file for file in os.listdir(source_dir) if check_video_file(file)]
for file in files:
    generator.generate_video(f"{source_dir}/{file}", dest_dir)
    print(f"preprocessed: {file}")
