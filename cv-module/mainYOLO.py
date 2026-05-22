# mainYOLO.py
# IoU-based tracker + kirim data ke backend Flask

import cv2
import mediapipe as mp
from ultralytics import YOLO
from datetime import datetime
from detector import analyze_pose, draw_skeleton
from utils import calculate_engagement, smooth_engagement
import config
import time
import requests  # <-- tambahan untuk kirim ke backend

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

# =========================
# TRACKER IoU
# =========================
tracked_persons = []
next_id = 0

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(areaA + areaB - interArea)

def update_debounce(person, key_deb, key_state, detected):
    if detected:
        person[key_deb] = min(person[key_deb] + 1, config.DEBOUNCE_FRAMES)
    else:
        person[key_deb] = max(person[key_deb] - 1, 0)
    if person[key_deb] >= config.DEBOUNCE_FRAMES:
        person[key_state] = 1
    elif person[key_deb] == 0:
        person[key_state] = 0

# =========================
# FUNGSI KIRIM KE BACKEND
# =========================
def send_to_backend(data):
    """Kirim data JSON ke Flask backend. Non-blocking (tidak crash kalau backend mati)."""
    try:
        requests.post(
            config.BACKEND_URL,
            json=data,
            timeout=1  # timeout 1 detik agar tidak lag
        )
    except Exception:
        pass  # Abaikan error koneksi agar program tetap jalan

# =========================
# MAIN LOOP
# =========================
prev_engagement = 0.0
last_send       = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    # Kumpulkan deteksi frame ini
    detections = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0 or float(box.conf[0]) < config.YOLO_CONFIDENCE:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2-x1) < config.MIN_BOX_WIDTH or (y2-y1) < config.MIN_BOX_HEIGHT:
                continue
            detections.append((x1, y1, x2, y2))

    # Match deteksi ke tracker (IoU)
    matched_track_ids = set()
    matched_det_ids   = set()

    for di, det in enumerate(detections):
        best_iou   = 0.3
        best_track = -1
        for ti, track in enumerate(tracked_persons):
            if ti in matched_track_ids:
                continue
            score = iou(det, track["box"])
            if score > best_iou:
                best_iou   = score
                best_track = ti
        if best_track >= 0:
            tracked_persons[best_track]["box"]  = det
            tracked_persons[best_track]["miss"] = 0
            matched_track_ids.add(best_track)
            matched_det_ids.add(di)

    for di, det in enumerate(detections):
        if di not in matched_det_ids:
            tracked_persons.append({
                "box": det, "miss": 0, "id": next_id,
                "deb_angkat": 0, "deb_hadap": 0, "deb_nunduk": 0,
                "state_angkat": 0, "state_hadap": 0, "state_nunduk": 0
            })
            next_id += 1

    for ti in range(len(tracked_persons)):
        if ti not in matched_track_ids:
            tracked_persons[ti]["miss"] += 1

    tracked_persons = [t for t in tracked_persons if t["miss"] <= config.MAX_MISS_FRAMES]

    # Proses pose per track
    total_angkat   = 0
    total_hadap    = 0
    total_menunduk = 0

    for track in tracked_persons:
        x1, y1, x2, y2 = track["box"]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_rgb     = crop[:, :, ::-1]
        pose_results = pose.process(crop_rgb)
        data         = analyze_pose(pose_results)

        update_debounce(track, "deb_angkat", "state_angkat", data["angkat_tangan"])
        update_debounce(track, "deb_hadap",  "state_hadap",  data["menghadap_depan"])
        update_debounce(track, "deb_nunduk", "state_nunduk", data["menunduk"])

        total_angkat   += track["state_angkat"]
        total_hadap    += track["state_hadap"]
        total_menunduk += track["state_nunduk"]

        draw_skeleton(crop, pose_results)

        box_color = (0, 0, 255) if track["state_nunduk"] else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame,
                    f"ID:{track['id']} T:{track['state_angkat']} H:{track['state_hadap']} N:{track['state_nunduk']}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)

    total_siswa = len(tracked_persons)

    # Hitung engagement
    raw_score  = calculate_engagement(total_siswa, total_angkat, total_hadap, total_menunduk)
    engagement = smooth_engagement(prev_engagement, raw_score)
    prev_engagement = engagement

    # Output JSON ke terminal
    output = {
        "total_siswa":      total_siswa,
        "angkat_tangan":    total_angkat,
        "menghadap_depan":  total_hadap,
        "menunduk":         total_menunduk,
        "engagement_score": round(float(engagement), 2),
        "timestamp":        datetime.now().isoformat()
    }
    print(output)

    # =========================
    # KIRIM KE BACKEND (per interval)
    # =========================
    if time.time() - last_send >= config.SEND_INTERVAL:
        send_to_backend(output)
        last_send = time.time()

    # HUD overlay
    cv2.putText(frame, f"Siswa : {total_siswa}",    (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),   2)
    cv2.putText(frame, f"Tangan: {total_angkat}",   (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),   2)
    cv2.putText(frame, f"Hadap : {total_hadap}",    (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Nunduk: {total_menunduk}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),   2)
    cv2.putText(frame, f"Engage: {engagement:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("YOLO Smart Classroom", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
pose.close()
cap.release()
cv2.destroyAllWindows()