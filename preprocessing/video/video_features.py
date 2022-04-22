import os
import cv2
import dlib
import numpy as np
import shutil

from preprocessing.video.landmark_detectors.common import get_landmarks_as_points, apply_points_to_image, \
    __shape_predictor_68, __shape_predictor_5


def save_frames(source, dest_dir, cut=True, apply_landmarks=False):
    path_for_fragmentation_dir = 'fragmentated'
    fragmentate(source, path_for_fragmentation_dir)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor(__shape_predictor_5)

    files = os.listdir(path_for_fragmentation_dir)
    cnt_files = len(files)
    third_size = cnt_files // 3
    for idx, f in enumerate(sorted(files)):
        image = cv2.imread(f"{path_for_fragmentation_dir}/{f}")

        third_number = get_third_number(idx, cnt_files)
        if third_number == 2:
            break

        points = get_landmarks_as_points(image, face_detector, landmark_detector)

        path_for_image = f'{dest_dir}/{source.split("/")[-1].split(".")[0]}_0{third_number + 1}'
        if not os.path.exists(path_for_image):
            os.makedirs(path_for_image)

        if apply_landmarks:
            image = apply_points_to_image(image, points)

        if cut:
            x, y = points[0][0], points[0][1]
            image = image[y:y + 50, x - 30:x + 30]

        cv2.imwrite(f"{path_for_image}/{idx - third_number * third_size}.jpg", image)

    shutil.rmtree(path_for_fragmentation_dir)
    frames_count = len(files)
    return frames_count


def fragmentate(source, dest_dir):
    vidcap = cv2.VideoCapture(source)
    success, image = vidcap.read()
    count = 0
    success = True

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    while success:
        cv2.imwrite(f"{dest_dir}/%d.jpg" % count, image)  # save frame as JPEG file

        count += 1
        success, image = vidcap.read()

    if count != 75:
        raise Exception


def get_third_number(x, cnt):
    if x * 3 < cnt:
        return 0
    else:
        if x * 3 < 2 * cnt:
            return 1
        else:
            return 2