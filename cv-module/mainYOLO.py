# mainYOLO.py
import cv2
import mediapipe as mp
from ultralytics import YOLO
from datetime import datetime
from detector import analyze_pose, draw_skeleton
from utils import calculate_engagement, smooth_engagement
import config
import time
import os
import numpy as np
import requests  # ← kirim data ke Flask

# =========================
# LOAD MODEL
# =========================
model   = YOLO(config.YOLO_MODEL)
mp_pose = mp.solutions.pose

# =========================
# POSE INSTANCE GLOBAL
# =========================
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=config.POSE_MODEL_COMPLEXITY,
    min_detection_confidence=config.POSE_DETECTION_CONFIDENCE,
    min_tracking_confidence=config.POSE_TRACKING_CONFIDENCE
)

# =========================
# KAMERA
# =========================
cap = cv2.VideoCapture(config.CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

# =========================
# TRACKER IoU
# =========================
tracked_persons = []
next_id = 0

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA);   interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0: return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(areaA + areaB - interArea)

def update_debounce(person, key_deb, key_state, detected):
    if detected:
        person[key_deb] = min(person[key_deb] + 1, config.DEBOUNCE_FRAMES)
    else:
        person[key_deb] = max(person[key_deb] - 1, 0)
    if   person[key_deb] >= config.DEBOUNCE_FRAMES: person[key_state] = 1
    elif person[key_deb] == 0:                       person[key_state] = 0

# =========================
# STUDENT FACE RECOGNITION
# =========================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
student_recognizer = None
student_label_names = {}
marked_attendance_today = set()
last_student_refresh = 0


def detect_face_gray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=config.FACE_MIN_SIZE)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = gray[y:y+h, x:x+w]
    return cv2.resize(face, (200, 200))


def normalize_face_image(img):
    if img is None or img.size == 0:
        return None
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    face = detect_face_gray(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    if face is not None:
        return face
    try:
        return cv2.resize(gray, (200, 200))
    except Exception:
        return None


def load_student_model():
    global student_recognizer, student_label_names, last_student_refresh
    try:
        res = requests.get(config.BACKEND_STUDENTS_URL, timeout=2)
        data = res.json()
        if not data.get("students"):
            print("[ATTENDANCE] Tidak ada data siswa terdaftar")
            student_recognizer = None
            return

        faces = []
        labels = []
        student_label_names = {}

        for student in data["students"]:
            foto = student.get("foto_path")
            if not foto or not os.path.exists(foto):
                continue
            img = cv2.imread(foto, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            face = normalize_face_image(img)
            if face is None:
                continue
            faces.append(face)
            labels.append(int(student["id"]))
            student_label_names[int(student["id"])] = student.get("nama", "Unknown")

        if faces and labels:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(np.array(faces), np.array(labels, dtype=np.int32))
            student_recognizer = recognizer
            last_student_refresh = time.time()
            print(f"[ATTENDANCE] Student face model loaded: {len(labels)} siswa")
        else:
            student_recognizer = None
            print("[ATTENDANCE] Tidak cukup foto siswa untuk model")
    except Exception as e:
        student_recognizer = None
        print(f"[ATTENDANCE] Gagal muat model siswa: {e}")


def sync_marked_attendance():
    global marked_attendance_today
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        res = requests.get(f"{config.BACKEND_STUDENTS_URL.replace('/students', '/attendance')}?tanggal={today}", timeout=2)
        data = res.json()
        marked_attendance_today = {int(r["id"]) for r in data.get("attendance", []) if r.get("status") == "hadir"}
    except Exception:
        marked_attendance_today = set()


def send_attendance(student_id):
    try:
        payload = {
            "student_id": int(student_id),
            "status": "hadir",
            "keterangan": "Auto hadir via kamera"
        }
        res = requests.post(config.BACKEND_ATTENDANCE_MARK_URL, json=payload, timeout=2)
        if res.ok:
            marked_attendance_today.add(int(student_id))
            print(f"[ATTENDANCE] Tercatat hadir: {student_label_names.get(student_id, student_id)}")
        else:
            print(f"[ATTENDANCE] Gagal kirim hadir {student_id}: {res.status_code} {res.text}")
    except Exception as e:
        print(f"[ATTENDANCE] Error kirim hadir {student_id}: {e}")


def recognize_student(crop, track):
    global student_recognizer, last_student_refresh
    if time.time() - last_student_refresh > config.STUDENT_MODEL_REFRESH_SEC:
        load_student_model()
    if student_recognizer is None:
        return None
    face = normalize_face_image(crop)
    if face is None:
        return None
    try:
        label, confidence = student_recognizer.predict(face)
    except Exception as e:
        print(f"[ATTENDANCE] Predict error: {e}")
        return None
    if confidence <= config.STUDENT_CONFIDENCE_THRESHOLD:
        track["student_id"] = int(label)
        track["student_name"] = student_label_names.get(label, "Unknown")
        if label not in marked_attendance_today:
            send_attendance(label)
        return int(label)
    return None

# =========================
# KIRIM KE FLASK
# =========================
def send_to_backend(data):
    try:
        requests.post(config.BACKEND_URL, json=data, timeout=1)
    except Exception:
        pass  # tidak crash kalau backend belum jalan

# =========================
# MAIN LOOP
# =========================
prev_engagement = 0.0
last_send       = time.time()

load_student_model()
sync_marked_attendance()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    # Kumpulkan deteksi
    detections = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0 or float(box.conf[0]) < config.YOLO_CONFIDENCE:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2-x1) < config.MIN_BOX_WIDTH or (y2-y1) < config.MIN_BOX_HEIGHT:
                continue
            detections.append((x1, y1, x2, y2))

    # Match IoU
    matched_track_ids = set()
    matched_det_ids   = set()
    for di, det in enumerate(detections):
        best_iou = 0.3; best_track = -1
        for ti, track in enumerate(tracked_persons):
            if ti in matched_track_ids: continue
            score = iou(det, track["box"])
            if score > best_iou: best_iou = score; best_track = ti
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
                "state_angkat": 0, "state_hadap": 0, "state_nunduk": 0,
                "student_id": None, "student_name": None, "last_attendance_sent": 0
            })
            next_id += 1

    for ti in range(len(tracked_persons)):
        if ti not in matched_track_ids:
            tracked_persons[ti]["miss"] += 1

    tracked_persons = [t for t in tracked_persons if t["miss"] <= config.MAX_MISS_FRAMES]

    # Proses pose
    total_angkat = 0; total_hadap = 0; total_menunduk = 0
    for track in tracked_persons:
        x1, y1, x2, y2 = track["box"]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: continue

        pose_results = pose.process(crop[:, :, ::-1])
        data         = analyze_pose(pose_results)

        update_debounce(track, "deb_angkat", "state_angkat", data["angkat_tangan"])
        update_debounce(track, "deb_hadap",  "state_hadap",  data["menghadap_depan"])
        update_debounce(track, "deb_nunduk", "state_nunduk", data["menunduk"])

        total_angkat   += track["state_angkat"]
        total_hadap    += track["state_hadap"]
        total_menunduk += track["state_nunduk"]

        student_id = recognize_student(crop, track)
        student_name = track.get("student_name") or ""
        display_label = student_name or f"ID:{track['id']}"

        draw_skeleton(crop, pose_results)

        box_color = (0, 0, 255) if track["state_nunduk"] else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, display_label, (x1, y1-28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)
        cv2.putText(frame,
            f"T:{track['state_angkat']} H:{track['state_hadap']} N:{track['state_nunduk']}",
            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

    total_siswa = len(tracked_persons)

    # Hitung engagement
    raw_score       = calculate_engagement(total_siswa, total_angkat, total_hadap, total_menunduk)
    engagement      = smooth_engagement(prev_engagement, raw_score)
    prev_engagement = engagement

    # Output terminal
    output = {
        "total_siswa":      total_siswa,
        "angkat_tangan":    total_angkat,
        "menghadap_depan":  total_hadap,
        "menunduk":         total_menunduk,
        "engagement_score": round(float(engagement), 2),
        "timestamp":        datetime.now().isoformat()
    }
    print(output)

    # Kirim ke Flask setiap SEND_INTERVAL detik
    if time.time() - last_send >= config.SEND_INTERVAL:
        send_to_backend(output)
        last_send = time.time()

    # HUD OpenCV
    cv2.putText(frame, f"Siswa : {total_siswa}",    (10,30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),   2)
    cv2.putText(frame, f"Tangan: {total_angkat}",   (10,60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),   2)
    cv2.putText(frame, f"Hadap : {total_hadap}",    (10,90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"Nunduk: {total_menunduk}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),   2)
    cv2.putText(frame, f"Engage: {engagement:.2f}", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("YOLO Smart Classroom", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

pose.close()
cap.release()
cv2.destroyAllWindows()