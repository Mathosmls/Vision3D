# smoother.py

from kalman_filter import KalmanFilter2D


class MultiPersonSmoother:
    def __init__(self, num_persons=2, num_keypoints=33, fps=30):
        self.num_persons = num_persons
        self.num_keypoints = num_keypoints

        self.filters = {
            person_id: [
                KalmanFilter2D(dt=1/fps)
                for _ in range(num_keypoints)
            ]
            for person_id in range(num_persons)
        }

    def smooth(self, tracked_persons):
        """
        tracked_persons: [(id, keypoints)]
        keypoints en coordonnées normalisées
        """

        smoothed = []

        for person_id, keypoints in tracked_persons:
            new_keypoints = []

            for i, kp in enumerate(keypoints):
                kf = self.filters[person_id][i]

                kf.predict()
                smoothed_xy = kf.update(kp[:2])

                new_keypoints.append([
                    smoothed_xy[0],
                    smoothed_xy[1],
                    kp[2]  # garder z brut pour l'instant
                ])

            smoothed.append((person_id, new_keypoints))

        return smoothed
