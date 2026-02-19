# main_multi_pose.py

import cv2
from pose_detector import MultiPoseDetector
from video_source import VideoSource
from config import FRAME_WIDTH, FRAME_HEIGHT
from person_tracker import ColorAwareTracker
from visualizer import draw_person
from json_exporter import JSONExporter
from smoother import MultiPersonSmoother

def main(source=0):  # 0 = webcam, sinon chemin vidéo
    video = VideoSource(source)

    detector = MultiPoseDetector()
    tracker = ColorAwareTracker()
    exporter = JSONExporter()
    frame_id = 0
    smoother = MultiPersonSmoother(num_persons=2, num_keypoints=33, fps=30)
    while True:
        frame, timestamp_ms = video.read()
        if frame is None:
            break
        
        poses = detector.detect(frame, timestamp_ms)

        tracked_persons = tracker.update(frame, poses)

        smoothed_persons = smoother.smooth(tracked_persons)

        for person_id, keypoints in tracked_persons:
            frame = draw_person(frame, person_id, keypoints)
        print(f"Nombre de poses détectées: {len(poses)}")
        print(f"Nombre de personnes suivies: {[p[0] for p in tracked_persons]}")

        cv2.imshow("Multi Pose Detection", frame)

        h, w = frame.shape[:2]

        exporter.add_frame(
            frame_id,
            smoothed_persons,
            w,
            h
        )

        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    exporter.save("output.json", fps=30)
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # main(0)  # webcam
    main("example2.mp4")  # vidéo
