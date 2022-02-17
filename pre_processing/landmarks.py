import cv2
import os
from landmark_detectors.common import get_landmarks_as_points, apply_points_to_image
from landmark_detectors.dlib5 import __get_mouth_landmarks5
from landmark_detectors.dlib68 import Landmarks


def find_landmarks_on_video(source, dest_dir):
    # fragmentate(source, dest_dir)

    files = os.listdir('dir')[:10]
    for f in files:
        find_landmarks_on_img(f)


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


def find_landmarks_on_img(img):
    image = cv2.imread(f"dir/{img}")

    points = get_landmarks_as_points(image, "dlib68", Landmarks.MOUTH)
    image = apply_points_to_image(image, points)
    cv2.imwrite(f"dir2/{img}", image)


# def merge(source):
#     os.system(f"ffmpeg -r 24 -i {source}/frame%01d.jpg -vcodec mpeg4 -y movie.mp4")
