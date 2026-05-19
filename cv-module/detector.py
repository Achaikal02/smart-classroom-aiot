import mediapipe as mp

# setup mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def detect_pose(frame):
    frame_rgb = frame[:, :, ::-1]  # BGR → RGB
    results = pose.process(frame_rgb)
    return results


def analyze_pose(results):
    if not results.pose_landmarks:
        return {
            "angkat_tangan": 0,
            "menghadap_depan": 0
        }

    landmarks = results.pose_landmarks.landmark

    # =========================
    # 🖐️ DETEKSI ANGKAT TANGAN (2 tangan)
    # =========================

    # tangan kanan
    right_hand = landmarks[16].y
    right_shoulder = landmarks[12].y

    # tangan kiri
    left_hand = landmarks[15].y
    left_shoulder = landmarks[11].y

    # threshold biar lebih akurat
    threshold = 0.05

    angkat_tangan = 1 if (
        right_hand < right_shoulder - threshold or
        left_hand < left_shoulder - threshold
    ) else 0

    # =========================
    # 👀 DETEKSI MENGHADAP DEPAN
    # =========================

    nose = landmarks[0].x
    left_shoulder_x = landmarks[11].x
    right_shoulder_x = landmarks[12].x

    center = (left_shoulder_x + right_shoulder_x) / 2

    menghadap_depan = 1 if abs(nose - center) < 0.1 else 0

    # =========================
    # RETURN DATA
    # =========================

    return {
        "angkat_tangan": angkat_tangan,
        "menghadap_depan": menghadap_depan
    }