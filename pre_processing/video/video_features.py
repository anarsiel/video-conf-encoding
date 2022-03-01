import os
import cv2
import dlib

from pre_processing.video.landmark_detectors.common import get_landmarks_as_points, apply_points_to_image, \
    __shape_predictor_68


def find_landmarks_on_video(source, dest_dir):
    fragmentate(source, 'dir')

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor(__shape_predictor_68)

    files = os.listdir('dir')
    for f in files:
        points = find_landmarks_on_img(f, face_detector, landmark_detector)

        f = open(f"{dest_dir}/{f.split('.')[0]}.txt", "w")
        for point in points:
            f.write(f"{point[0]} {point[1]}\n")
        f.close()


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


def find_landmarks_on_img(img, face_detector, landmark_detector):
    image = cv2.imread(f"dir/{img}")

    points = get_landmarks_as_points(image, face_detector, landmark_detector)
    image = apply_points_to_image(image, points)
    cv2.imwrite(f"dir2/{img}", image)
    return points

# def merge(source):
#     os.system(f"ffmpeg -r 24 -i {source}/frame%01d.jpg -vcodec mpeg4 -y movie.mp4")
