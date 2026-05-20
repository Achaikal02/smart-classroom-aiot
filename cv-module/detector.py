# detector.py

import mediapipe as mp
import config

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Satu instance, pakai dengan context manager agar thread-safe
_pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=config.POSE_MODEL_COMPLEXITY,
    enable_segmentation=False,
    min_detection_confidence=config.POSE_DETECTION_CONFIDENCE,
    min_tracking_confidence=config.POSE_TRACKING_CONFIDENCE
)

def detect_pose(frame_bgr):
    """Proses satu frame BGR, return pose landmarks."""
    frame_rgb = frame_bgr[:, :, ::-1]
    return _pose.process(frame_rgb)


def analyze_pose(results):
    """Analisis landmarks, return dict angkat_tangan & menghadap_depan."""
    if not results.pose_landmarks:
        return {"angkat_tangan": 0, "menghadap_depan": 0}

    lm = results.pose_landmarks.landmark

    # --- Angkat tangan ---
    right_up = lm[16].y < lm[12].y - config.HAND_RAISE_THRESHOLD
    left_up  = lm[15].y < lm[11].y - config.HAND_RAISE_THRESHOLD
    angkat = 1 if (right_up or left_up) else 0

    # --- Menghadap depan ---
    center_x = (lm[11].x + lm[12].x) / 2
    hadap = 1 if abs(lm[0].x - center_x) < config.FACE_CENTER_THRESHOLD else 0

    return {"angkat_tangan": angkat, "menghadap_depan": hadap}


def draw_skeleton(frame, results):
    """Gambar skeleton mediapipe ke frame (in-place)."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )