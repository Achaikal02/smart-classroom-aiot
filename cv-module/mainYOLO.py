# mainYOLO.py

import cv2
import mediapipe as mp
from ultralytics import YOLO
from datetime import datetime
from detector import analyze_pose, draw_skeleton
from utils import calculate_engagement, smooth_engagement
import config

# =========================
# LOAD MODEL
# =========================
model = YOLO(config.YOLO_MODEL)
mp_pose = mp.solutions.pose

# =========================
# POSE INSTANCE GLOBAL
# =========================
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=0,
    min_detection_confidence=config.POSE_DETECTION_CONFIDENCE,
    min_tracking_confidence=config.POSE_TRACKING_CONFIDENCE
)

# =========================
# KAMERA
# =========================
cap = cv2.VideoCapture(config.CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

prev_engagement = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    total_siswa    = 0
    total_angkat   = 0
    total_hadap    = 0
    total_menunduk = 0
    boxes_seen     = []

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0 or float(box.conf[0]) < config.YOLO_CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if (x2 - x1) < config.MIN_BOX_WIDTH or (y2 - y1) < config.MIN_BOX_HEIGHT:
                continue

            # anti duplikat
            if any(abs(x1 - bx) < config.DUPLICATE_THRESHOLD and
                   abs(y1 - by) < config.DUPLICATE_THRESHOLD
                   for bx, by in boxes_seen):
                continue

            boxes_seen.append((x1, y1))
            total_siswa += 1

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # pose pakai instance global
            crop_rgb     = crop[:, :, ::-1]
            pose_results = pose.process(crop_rgb)

            data = analyze_pose(pose_results)
            total_angkat   += data["angkat_tangan"]
            total_hadap    += data["menghadap_depan"]
            total_menunduk += data["menunduk"]

            draw_skeleton(crop, pose_results)

            # warna box: merah kalau menunduk, hijau kalau normal
            box_color = (0, 0, 255) if data["menunduk"] else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame,
                        f"T:{data['angkat_tangan']} H:{data['menghadap_depan']} N:{data['menunduk']}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # =========================
    # HITUNG ENGAGEMENT
    # =========================
    raw_score  = calculate_engagement(total_siswa, total_angkat, total_hadap, total_menunduk)
    engagement = smooth_engagement(prev_engagement, raw_score)
    prev_engagement = engagement

    # =========================
    # OUTPUT JSON
    # =========================
    output = {
        "total_siswa":    total_siswa,
        "angkat_tangan":  total_angkat,
        "menghadap_depan": total_hadap,
        "menunduk":       total_menunduk,
        "engagement_score": engagement,
        "timestamp":      datetime.now().isoformat()
    }
    print(output)

    # =========================
    # HUD
    # =========================
    cv2.putText(frame, f"Siswa : {total_siswa}",    (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),   2)
    cv2.putText(frame, f"Tangan: {total_angkat}",   (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),   2)
    cv2.putText(frame, f"Hadap : {total_hadap}",    (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Nunduk: {total_menunduk}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),   2)
    cv2.putText(frame, f"Engage: {engagement:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("YOLO Smart Classroom", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# CLEANUP
# =========================
pose.close()
cap.release()
cv2.destroyAllWindows()