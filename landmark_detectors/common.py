import dlib

from landmark_detectors.dlib5 import __get_mouth_landmarks5, __shape_predictor_5
from landmark_detectors.dlib68 import __get_mouth_landmarks68, Landmarks, __shape_predictor_68


def get_landmarks(image, predictor_type, landmark_wanted=Landmarks.MOUTH):
    if predictor_type == "dlib5":
        all_faces_landmarks = __face_landmarks(image, __shape_predictor_5)
        return __get_mouth_landmarks5(image, all_faces_landmarks)
    elif predictor_type == "dlib68":
        all_faces_landmarks = __face_landmarks(image, __shape_predictor_68)
        return __get_mouth_landmarks68(image, all_faces_landmarks, landmark_wanted)


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
