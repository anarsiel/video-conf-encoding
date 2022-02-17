import cv2

# from landmark_detectors.common import get_landmarks
# from landmark_detectors.dlib68 import Landmarks
#
# image = cv2.imread("resources/images/sonya.jpg")
#
# image = get_landmarks(image, "dlib5")
# image = get_landmarks(image, "dlib68", Landmarks.MOUTH)
#
# cv2.imwrite("a.jpg", image)

from pre_processing.landmarks import find_landmarks_on_video

find_landmarks_on_video("resources/videos/two-people-cleaning.mp4", "dir")
