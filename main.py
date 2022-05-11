from PIL import Image
from preprocessing.preprocess import create_dataset
from predicting.generator import Generator
import numpy as np
import cv2
import os

def check_video_file(file):
    video_ext = ['mpg', 'mp4', 'yuv']
    elements = file.split('.')
    return elements[-1] in video_ext


generator = Generator(
    weights_path="predicting/weights.hdf5",
    shape_predictor_path="preprocessing/video/landmark_detectors/shape_predictors/shape_predictor_5_face_landmarks.dat",
    mu_au_path="predicting/MU_AU.csv",
    std_au_path="predicting/STD_AU.csv"
)

video_source_dir = "test-video"
audio_source_dir = "test-audio"
dest_dir = "test-x"

# source_dir = "resources/test"
# dest_dir = "results"

# files = [file for file in os.listdir(source_dir) if check_video_file(file)]
# for file in files:
#     generator.generate_video(f"{source_dir}/{file}", dest_dir, save_mp4=True)
#     print(f"preprocessed: {file}")

file = "12345"
generator.generate_video(
    video_source=f"{video_source_dir}/{file}.yuv",
    audio_source=f"{audio_source_dir}/{file}.mp4",
    dest_path=dest_dir,
)
