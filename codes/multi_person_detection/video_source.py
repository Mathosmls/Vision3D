# video_source.py

import cv2
import time


class VideoSource:
    def __init__(self, source=0):
        """
        source = 0 → webcam
        source = "video.mp4" → fichier vidéo
        """
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la source vidéo")

        self.is_webcam = isinstance(source, int)

        if not self.is_webcam:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_index = 0
        else:
            self.start_time = time.time()

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        if self.is_webcam:
            timestamp_ms = int((time.time() - self.start_time) * 1000)
        else:
            timestamp_ms = int((self.frame_index / self.fps) * 1000)
            self.frame_index += 1

        return frame, timestamp_ms

    def release(self):
        self.cap.release()
