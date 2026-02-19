# pose_detector.py

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from config import MODEL_PATH, NUM_POSES, CONFIDENCE_THRESHOLD


class MultiPoseDetector:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=NUM_POSES,
            min_pose_detection_confidence=CONFIDENCE_THRESHOLD,
            min_pose_presence_confidence=CONFIDENCE_THRESHOLD,
            min_tracking_confidence=CONFIDENCE_THRESHOLD,
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame, timestamp_ms):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        poses = []

        for pose_landmarks in result.pose_landmarks:
            keypoints = []
            for landmark in pose_landmarks:
                keypoints.append([landmark.x, landmark.y, landmark.z])
            poses.append(np.array(keypoints))

        return poses
