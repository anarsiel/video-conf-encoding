import cv2

__shape_predictor_5 = "shape_predictor_5_face_landmarks.dat"


def __get_mouth_landmarks5(image, all_faces_landmarks):
    for face_landmarks in all_faces_landmarks:
        points = [(p.x, p.y) for p in face_landmarks.parts()]
        for p in points:
            cv2.circle(image, (int(p[0]), int(p[1])), 2, (255, 0, 0), -1)

    return image
