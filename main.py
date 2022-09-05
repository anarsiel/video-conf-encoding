from PIL import Image
from preprocessing.preprocess import create_dataset
from predicting.generator import Generator
import numpy as np
import cv2
import os


def check_video_file(file):
    video_ext = ['yuv']  # ['mpg', 'mp4', 'yuv']
    elements = file.split('.')
    return elements[-1] in video_ext


generator = Generator(
    weights_path="predicting/weights.hdf5",
    shape_predictor_path="preprocessing/video/landmark_detectors/shape_predictors/shape_predictor_5_face_landmarks.dat",
    mu_au_path="predicting/MU_AU.csv",
    std_au_path="predicting/STD_AU.csv"
)

video_source_dir = "test-video"
dest_dir = "test-x"


# audio_file = "test-audio/sasha.wav"
files = [file for file in os.listdir(video_source_dir) if check_video_file(file)]
for file in files:
    filename = file.split('.')[0]
    generator.generate_video(
        video_source=f"{video_source_dir}/{file}",
        audio_source=f'test-audio/12345.mov',
        dest_path=dest_dir,
        t=8
    )

    print(f'preprocessed: {file}')
