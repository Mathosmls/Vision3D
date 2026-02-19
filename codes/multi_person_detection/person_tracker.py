# person_tracker.py (version robuste)

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


class FixedPerson:
    def __init__(self, person_id):
        self.id = person_id
        self.keypoints = None
        self.color_hist = None
        self.visible = False
        self.missed_frames = 0


class ColorAwareTracker:
    def __init__(self, max_missed=30):
        self.tracks = [FixedPerson(0), FixedPerson(1)]
        self.max_missed = max_missed

        self.alpha = 0.35  # poids spatial
        self.beta = 0.55   # poids couleur

    def _compute_center(self, keypoints):
        left_hip = keypoints[23]
        right_hip = keypoints[24]
        return ((left_hip + right_hip) / 2.0)[:2]

    def _extract_torso_histogram(self, frame, keypoints):
        h, w = frame.shape[:2]

        pts = [keypoints[11], keypoints[12], keypoints[23], keypoints[24]]
        pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in pts])

        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)

        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        return hist

    def update(self, frame, detections):
        if len(detections) == 0:
            for t in self.tracks:
                t.missed_frames += 1
                if t.missed_frames > self.max_missed:
                    t.visible = False
            return [(t.id, t.keypoints) for t in self.tracks]

        cost_matrix = np.zeros((2, len(detections)))

        det_centers = [self._compute_center(d) for d in detections]
        det_hists = [self._extract_torso_histogram(frame, d) for d in detections]

        for i, track in enumerate(self.tracks):
            for j in range(len(detections)):

                spatial_cost = 0
                color_cost = 0

                if track.keypoints is not None:
                    track_center = self._compute_center(track.keypoints)
                    spatial_cost = np.linalg.norm(track_center - det_centers[j])

                if track.color_hist is not None and det_hists[j] is not None:
                    color_cost = cv2.compareHist(
                        track.color_hist,
                        det_hists[j],
                        cv2.HISTCMP_BHATTACHARYYA
                    )

                cost_matrix[i, j] = (
                    self.alpha * spatial_cost +
                    self.beta * color_cost
                )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            self.tracks[r].keypoints = detections[c]
            self.tracks[r].color_hist = det_hists[c]
            self.tracks[r].visible = True
            self.tracks[r].missed_frames = 0

        return [(t.id, t.keypoints) for t in self.tracks if t.keypoints is not None]
