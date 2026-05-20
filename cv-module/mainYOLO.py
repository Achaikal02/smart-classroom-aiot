# mainYOLO.py

import cv2
import mediapipe as mp
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict, deque
from detector import analyze_pose, draw_skeleton
from utils import calculate_engagement, smooth_engagement
import config
import time

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
# TRACKER & DEBOUNCE
# =========================
siswa_tracker   = {}                  # key: grid, value: sisa frame toleransi
debounce_angkat = defaultdict(int)
debounce_hadap  = defaultdict(int)
debounce_nunduk = defaultdict(int)
state_angkat    = defaultdict(int)
state_hadap     = defaultdict(int)
state_nunduk    = defaultdict(int)

prev_engagement = 0.0
last_send       = time.time()

def update_debounce(counter, state, key, detected):
    """Update debounce counter dan state stabil."""
    if detected:
        counter[key] = min(counter[key] + 1, config.DEBOUNCE_FRAMES)
    else:
        counter[key] = max(counter[key] - 1, 0)

    if counter[key] >= config.DEBOUNCE_FRAMES:
        state[key] = 1
    elif counter[key] == 0:
        state[key] = 0

    return counter, state

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    total_angkat   = 0
    total_hadap    = 0
    total_menunduk = 0
    boxes_seen     = []
    active_keys    = set()

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

            # update tracker
            grid_key = (x1 // 50, y1 // 50)
            siswa_tracker[grid_key] = config.MAX_MISS_FRAMES
            active_keys.add(grid_key)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb     = crop[:, :, ::-1]
            pose_results = pose.process(crop_rgb)

            data = analyze_pose(pose_results)

            # debounce per siswa
            debounce_angkat, state_angkat = update_debounce(
                debounce_angkat, state_angkat, grid_key, data["angkat_tangan"])
            debounce_hadap, state_hadap = update_debounce(
                debounce_hadap, state_hadap, grid_key, data["menghadap_depan"])
            debounce_nunduk, state_nunduk = update_debounce(
                debounce_nunduk, state_nunduk, grid_key, data["menunduk"])

            total_angkat   += state_angkat[grid_key]
            total_hadap    += state_hadap[grid_key]
            total_menunduk += state_nunduk[grid_key]

            draw_skeleton(crop, pose_results)

            # warna box: merah = menunduk, hijau = normal
            box_color = (0, 0, 255) if state_nunduk[grid_key] else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame,
                        f"T:{state_angkat[grid_key]} H:{state_hadap[grid_key]} N:{state_nunduk[grid_key]}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # =========================
    # UPDATE TRACKER
    # kurangi counter siswa yang tidak terdeteksi frame ini
    # =========================
    to_delete = []
    for key in siswa_tracker:
        if key not in active_keys:
            siswa_tracker[key] -= 1
            if siswa_tracker[key] <= 0:
                to_delete.append(key)
    for k in to_delete:
        del siswa_tracker[k]
        # bersihkan state debounce siswa yang sudah hilang
        debounce_angkat.pop(k, None)
        debounce_hadap.pop(k, None)
        debounce_nunduk.pop(k, None)
        state_angkat.pop(k, None)
        state_hadap.pop(k, None)
        state_nunduk.pop(k, None)

    total_siswa = len(siswa_tracker)

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
        "total_siswa":     total_siswa,
        "angkat_tangan":   total_angkat,
        "menghadap_depan": total_hadap,
        "menunduk":        total_menunduk,
        "engagement_score": engagement,
        "timestamp":       datetime.now().isoformat()
    }
    print(output)

    # =========================
    # KIRIM KE BACKEND (per interval)
    # =========================
    # TODO: aktifkan setelah sender.py dibuat
    # if time.time() - last_send >= config.SEND_INTERVAL:
    #     from sender import send_data
    #     send_data(output)
    #     last_send = time.time()
    
    
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