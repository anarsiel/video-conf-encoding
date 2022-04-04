import os
import cv2
import dlib
import numpy as np
import shutil

from pre_processing.video.landmark_detectors.common import get_landmarks_as_points, apply_points_to_image, \
    __shape_predictor_68, __shape_predictor_5


def save_frames(source, dest_dir, cut=True, apply_landmarks=False, save_landmarks=False):
    path_for_fragmentation_dir = 'fragmentated'
    fragmentate(source, path_for_fragmentation_dir)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor(__shape_predictor_5)

    files = os.listdir(path_for_fragmentation_dir)
    for f in files:
        image = cv2.imread(f"{path_for_fragmentation_dir}/{f}")

        if image is None:
            continue

        points = find_landmarks_on_img(image, f, f'{dest_dir}/{source.split(".")[0].split("/")[-1]}', face_detector, landmark_detector, cut, apply_landmarks)

        if save_landmarks:
            np.savetxt(f"{dest_dir}/{f.split('.')[0]}.csv", points, delimiter=",")

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

        if cv2.waitKey(10) == 27:
            break

        count += 1
        success, image = vidcap.read()


def find_landmarks_on_img(image, image_name, dest_dir, face_detector, landmark_detector, cut, apply_landmarks):
    points = get_landmarks_as_points(image, face_detector, landmark_detector)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if apply_landmarks:
        image = apply_points_to_image(image, points)

    if cut:
        x, y = points[0][0], points[0][1]
        image = image[y:y + 50, x - 30:x + 30]

    cv2.imwrite(f"{dest_dir}/{image_name}", image)
    return points

# def merge(source):
#     os.system(f"ffmpeg -r 24 -i {source}/frame%01d.jpg -vcodec mpeg4 -y movie.mp4")
