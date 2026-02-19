# visualizer.py

import cv2
import numpy as np


COLOR_LIST = [
    (0, 255, 0),     # Vert
    (255, 0, 0),     # Bleu
    (0, 0, 255),     # Rouge
    (255, 255, 0),   # Cyan
]


def draw_person(frame, person_id, keypoints, confidence_threshold=0.5):
    h, w = frame.shape[:2]
    color = COLOR_LIST[person_id % len(COLOR_LIST)]

    pixel_points = []

    for kp in keypoints:
        x = int(kp[0] * w)
        y = int(kp[1] * h)
        pixel_points.append((x, y))

    # Dessiner les points
    for (x, y) in pixel_points:
        cv2.circle(frame, (x, y), 4, color, -1)

    # Centre des hanches pour positionner l'ID
    left_hip = keypoints[23]
    right_hip = keypoints[24]

    center_x = int(((left_hip[0] + right_hip[0]) / 2) * w)
    center_y = int(((left_hip[1] + right_hip[1]) / 2) * h)

    cv2.putText(
        frame,
        f"ID {person_id}",
        (center_x, center_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    return frame
