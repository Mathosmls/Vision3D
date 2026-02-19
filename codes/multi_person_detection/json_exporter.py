# json_exporter.py

import json
from datetime import datetime


class JSONExporter:
    def __init__(self):
        self.frames = []

    def add_frame(self, frame_id, persons, frame_width, frame_height):
        """
        persons: liste de (id, keypoints normalisés)
        """
        frame_data = {
            "frame_id": frame_id,
            "persons": []
        }

        for person_id, keypoints in persons:
            pixel_keypoints = []

            for kp in keypoints:
                pixel_keypoints.append({
                    "x": float(kp[0] * frame_width),
                    "y": float(kp[1] * frame_height),
                    "z": float(kp[2])
                })

            frame_data["persons"].append({
                "id": person_id,
                "keypoints": pixel_keypoints
            })

        self.frames.append(frame_data)

    def save(self, output_path, fps):
        output = {
            "metadata": {
                "fps": fps,
                "created_at": datetime.now().isoformat(),
                "format": "multi_person_pose_v1"
            },
            "total_frames": len(self.frames),
            "frames": self.frames
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"JSON sauvegardé : {output_path}")
