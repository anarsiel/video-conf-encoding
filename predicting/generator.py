import copy
import math

import numpy as np
import dlib
import cv2
import librosa
from training.model import create_model

import os

from preprocessing.video.landmark_detectors.common import apply_points_to_image, get_landmarks_as_points, \
    __shape_predictor_5


class Generator:
    def __init__(self, weights_path="", shape_predictor_path="", mu_au_path="MU_AU.csv", std_au_path="STD_AU.csv"):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_detector = dlib.shape_predictor(shape_predictor_path)

        self.MU = 0.
        self.STD = 255.
        self.MU_AU = np.loadtxt(mu_au_path)
        self.STD_AU = np.loadtxt(std_au_path)

        self.model = self.__load_model(weights_path)

    @staticmethod
    def __load_model(weights_path):
        model = create_model()
        model.compile(
            loss='mean_squared_error',
            optimizer='adam',
            metrics=['mean_squared_error']
        )
        model.load_weights(weights_path)

        return model

    def __get_mouth(self, frame):
        points = get_landmarks_as_points(frame, self.face_detector, self.landmark_detector)

        x, y = points[-1][0], points[-1][1]
        image = frame[y:y + 50, x - 30:x + 30]

        return points, image

    def __fragmentate(self, source):
        vidcap = cv2.VideoCapture(source)
        success, image = vidcap.read()
        success = True

        frames = []
        original_frames = []
        noses = []
        while success:
            original_frames.append(image)

            points, image = self.__get_mouth(image)
            image = image.astype('float64')
            image -= self.MU
            image /= self.STD
            frames.append(image)

            noses.append(points[2])

            success, image = vidcap.read()

        return np.array(frames), np.array(original_frames), noses

    def __get_mfccs(self, source):
        x, sr = librosa.load(source)
        mfccs = librosa.feature.mfcc(x, sr=sr)

        mfccs = mfccs.transpose()
        mfccs -= self.MU_AU
        mfccs /= self.STD_AU
        return mfccs

    def __generate(self, frame, mfccs, t):
        frame = np.expand_dims(frame, axis=0)
        mfccs = np.expand_dims(mfccs, axis=0)

        Y = self.model.predict({'image': frame, 'audio': mfccs})

        Y *= self.STD
        Y += self.MU
        return Y[0][:t]

    def __put_mouth(self, frame, output_frames, nose):
        overlayed_frames = np.empty((output_frames.shape[0],) + frame.shape)
        for idx, output_frame in enumerate(output_frames):
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

            overlay = np.copy(frame)
            overlay[nose[1]:nose[1] + 50, nose[0] - 30:nose[0] + 30] = output_frame
            overlayed_frames[idx] = overlay
        return overlayed_frames

    def frames_to_video(self, frames, filename_extension):
        filename, extension = filename_extension
        filename_g = f"{filename}_g"

        height, width, layers = frames[0].shape
        video = cv2.VideoWriter(f'tmp.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height), True)

        for frame in frames:
            video.write(np.uint8(frame))

        cv2.destroyAllWindows()
        video.release()

        os.system(f'ffmpeg -y -i {filename}.{extension} {filename}.mp3')
        os.system(f'ffmpeg -y -i tmp.mp4 -i {filename}.mp3 -c:v copy -c:a aac {filename_g}.mp4')

    def generate_video(self, path_to_video, t):
        filename_extension = path_to_video.split("/")[-1].split('.')

        frames, original_frames, noses = self.__fragmentate(path_to_video)
        mfccs = self.__get_mfccs(path_to_video)

        mfccs_len = len(mfccs)
        last = np.expand_dims(mfccs[mfccs_len - 1], axis=0)
        for i in range(100):
            mfccs = np.append(mfccs, copy.deepcopy(last), axis=0)

        gen_frames = np.empty(original_frames.shape)
        for i in range(0, len(frames), t):
            print(i)
            input_frame = copy.deepcopy(frames[i])

            mfccs_starting_index = math.floor(43.0 * i / 25)
            input_mfccs = copy.deepcopy(mfccs[mfccs_starting_index:mfccs_starting_index + 43])

            output_frames = self.__generate(input_frame, input_mfccs, t)
            output_frames = self.__put_mouth(original_frames[i], output_frames, noses[i])

            gen_frames[i] = copy.deepcopy(original_frames[i])
            for j in range(i + 1, min(i + t, len(frames))):
                gen_frames[j] = output_frames[j - i - 1]
            # gen_frames = np.r_[gen_frames, np.array([input_frame]), output_frames]

        self.frames_to_video(gen_frames, filename_extension)
