from enum import Enum


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
