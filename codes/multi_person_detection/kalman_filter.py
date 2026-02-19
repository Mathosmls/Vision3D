# kalman_filter.py

import numpy as np


class KalmanFilter2D:
    def __init__(self, dt=1/30):

        self.dt = dt

        # Etat : [x, y, vx, vy]
        self.x = np.zeros((4, 1))

        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        self.P = np.eye(4) * 500
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 5

        self.initialized = False

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        z = np.array(z).reshape(2, 1)

        if not self.initialized:
            self.x[0:2] = z
            self.initialized = True
            return self.x[0:2].flatten()

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

        return self.x[0:2].flatten()
