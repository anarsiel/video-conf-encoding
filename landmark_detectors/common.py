import cv2
import dlib

from landmark_detectors.dlib5 import __get_mouth_landmarks5, __shape_predictor_5
from landmark_detectors.dlib68 import __get_mouth_landmarks68, Landmarks, __shape_predictor_68, landmarks_indexes, \
    dlib5_indexes


def get_landmarks_as_points(image, predictor_type):
    if predictor_type == "dlib5":
        all_faces_landmarks = __face_landmarks(image, __shape_predictor_5)
        return get_only_wanted_points(all_faces_landmarks)
        # return __get_mouth_landmarks5(image, all_faces_landmarks)
    elif predictor_type == "dlib68":
        all_faces_landmarks = __face_landmarks(image, __shape_predictor_68)
        return get_only_wanted_points(all_faces_landmarks)
        # return __get_mouth_landmarks68(image, all_faces_landmarks, landmark_wanted)


def __face_landmarks(image, shape_predictor):
    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor(shape_predictor)

    face_rects = face_detector(image, 0)

    all_landmarks = []
    for face_rect in face_rects:
        new_rect = dlib.rectangle(
            int(face_rect.left()),
            int(face_rect.top()),
            int(face_rect.right()),
            int(face_rect.bottom())
        )

        landmarks = landmark_detector(image, new_rect)
        all_landmarks.append(landmarks)

    return all_landmarks


def get_only_wanted_points(all_faces_landmarks):
    starting_index = landmarks_indexes[Landmarks.MOUTH][0]
    ending_index = landmarks_indexes[Landmarks.MOUTH][1]

    points = []
    for face_landmarks in all_faces_landmarks:
        for idx, p in enumerate(face_landmarks.parts()):
            if starting_index <= idx < ending_index or idx in dlib5_indexes:
                points.append((p.x, p.y))

    return points


def apply_points_to_image(image, points):
    for p in points:
        cv2.circle(image, (int(p[0]), int(p[1])), 2, (255, 0, 0), -1)

    return image
