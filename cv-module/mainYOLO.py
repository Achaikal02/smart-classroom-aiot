import cv2
from ultralytics import YOLO
import mediapipe as mp
from utils import calculate_engagement
from datetime import datetime

# =========================
# LOAD MODEL
# =========================
model = YOLO("yolov8n.pt")

# =========================
# MEDIAPIPE SETUP (GLOBAL)
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

prev_engagement = 0

while True:
    ret, frame = cap.read()

    # 🔁 LOOP VIDEO (WAJIB DI SINI)
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    results = model(frame, verbose=False)

    total_siswa = 0
    total_angkat = 0
    total_hadap = 0

    boxes_seen = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # hanya person + filter confidence
            if cls != 0 or conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # filter ukuran kecil (noise)
            if (x2 - x1) < 100 or (y2 - y1) < 150:
                continue

            # =========================
            # ANTI DUPLICATE
            # =========================
            duplicate = False
            for bx in boxes_seen:
                if abs(x1 - bx[0]) < 50 and abs(y1 - bx[1]) < 50:
                    duplicate = True
                    break

            if duplicate:
                continue

            boxes_seen.append((x1, y1))
            total_siswa += 1

            # =========================
            # CROP PERSON
            # =========================
            person = frame[y1:y2, x1:x2]
            if person.size == 0:
                continue

            # =========================
            # POSE DETECTION
            # =========================
            rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb)

            angkat = 0
            hadap = 0

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark

                # =========================
                # ANGKAT TANGAN (2 tangan)
                # =========================
                right_hand = landmarks[16].y
                right_shoulder = landmarks[12].y

                left_hand = landmarks[15].y
                left_shoulder = landmarks[11].y

                if (
                    right_hand < right_shoulder - 0.05 or
                    left_hand < left_shoulder - 0.05
                ):
                    angkat = 1

                total_angkat += angkat

                # =========================
                # MENGHADAP DEPAN (FIXED)
                # =========================

                nose = landmarks[0].x
                left_s = landmarks[11].x
                right_s = landmarks[12].x

                shoulder_center = (left_s + right_s) / 2

                # toleransi lebih longgar untuk video kelas
                hadap = 1 if abs(nose - shoulder_center) < 0.15 else 0

                # draw skeleton
                mp_drawing.draw_landmarks(
                    person,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            # =========================
            # DRAW BOX + LABEL
            # =========================
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            cv2.putText(frame,
                        f"T:{angkat} H:{hadap}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,0),
                        2)

    # =========================
    # HITUNG ENGAGEMENT
    # =========================
    if total_siswa == 0:
        engagement = 0
    else:
        engagement = calculate_engagement(
            total_siswa,
            total_angkat,
            total_hadap
        )
        
    engagement = 0.7 * prev_engagement + 0.3 * engagement
    prev_engagement = engagement

    # =========================
    # OUTPUT JSON
    # =========================
    output = {
        "total_siswa": total_siswa,
        "angkat_tangan": total_angkat,
        "menghadap_depan": total_hadap,
        "engagement_score": round(engagement, 2),
        "timestamp": datetime.now().isoformat()
    }

    print(output)

    # =========================
    # DISPLAY TEXT
    # =========================
    cv2.putText(frame, f"Siswa: {total_siswa}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(frame, f"Tangan: {total_angkat}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.putText(frame, f"Hadap: {total_hadap}", (10,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.putText(frame, f"Engagement: {engagement:.2f}", (10,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("YOLO Smart Classroom FINAL", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()