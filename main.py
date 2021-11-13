import cv2

from landmark_detectors.common import get_landmarks
from landmark_detectors.dlib68 import Landmarks

image = cv2.imread("faces/sonya.jpg")

image = get_landmarks(image, "dlib5")
image = get_landmarks(image, "dlib68", Landmarks.MOUTH)

cv2.imshow("Smile Detection", image)
cv2.waitKey(0)
