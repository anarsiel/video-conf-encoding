import os
import cv2
import dlib
import numpy as np

from pre_processing.video.landmark_detectors.common import get_landmarks_as_points, apply_points_to_image, \
    __shape_predictor_68


def find_landmarks_on_video(source, dest_dir='landmarks', apply_and_save=False):
    fragmentate(source, 'dir')

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor(__shape_predictor_68)

    files = os.listdir('dir')
    for f in files:
        points = find_landmarks_on_img(f, face_detector, landmark_detector, apply_and_save=apply_and_save)

        np.savetxt(f"{dest_dir}/{f.split('.')[0]}.csv", points, delimiter=",")

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
        cv2.imwrite(f"{dest_dir}/frame%d.jpg" % count, image)  # save frame as JPEG file

        if cv2.waitKey(10) == 27:
            break

        count += 1
        success, image = vidcap.read()


def find_landmarks_on_img(img, face_detector, landmark_detector, apply_and_save=False):
    image = cv2.imread(f"dir/{img}")

    points = get_landmarks_as_points(image, face_detector, landmark_detector)

    if apply_and_save:
        tmp_dir = "tmp"
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        image = apply_points_to_image(image, points)
        cv2.imwrite(f"{tmp_dir}/{img}", image)
    return points

# def merge(source):
#     os.system(f"ffmpeg -r 24 -i {source}/frame%01d.jpg -vcodec mpeg4 -y movie.mp4")
