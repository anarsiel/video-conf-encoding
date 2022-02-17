from enum import Enum

import cv2

__shape_predictor_68 = "landmark_detectors/shape_predictors/shape_predictor_68_face_landmarks.dat"


class Landmarks(Enum):
    MOUTH = 1
    RIGHT_EYEBROW = 2
    LEFT_EYEBROW = 3
    RIGHT_EYE = 4
    LEFT_EYE = 5
    NOSE = 6
    JAW = 7


landmarks_indexes = {
    Landmarks.JAW: (0, 17),
    Landmarks.RIGHT_EYEBROW: (17, 22),
    Landmarks.LEFT_EYEBROW: (22, 27),
    Landmarks.NOSE: (27, 35),
    Landmarks.RIGHT_EYE: (36, 42),
    Landmarks.LEFT_EYE: (42, 48),
    Landmarks.MOUTH: (48, 68)
}

dlib5_indexes = [36, 39, 42, 45, 33]


def __get_mouth_landmarks68(image, all_faces_landmarks, wanted_landmark: Landmarks = Landmarks.MOUTH):
    starting_index = landmarks_indexes[wanted_landmark][0]
    ending_index = landmarks_indexes[wanted_landmark][1]

    for face_landmarks in all_faces_landmarks:
        # points = [(p.x, p.y) for p in face_landmarks.parts()[starting_index:ending_index]]

        points = []
        for idx, p in enumerate(face_landmarks.parts()):
            if starting_index <= idx < ending_index or idx in dlib5_indexes:
                points.append((p.x, p.y))

        for p in points:
            cv2.circle(image, (int(p[0]), int(p[1])), 2, (255, 0, 0), -1)

    return image
