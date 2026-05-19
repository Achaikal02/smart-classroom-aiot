import cv2
import mediapipe as mp
from detector import detect_pose, analyze_pose
from utils import calculate_engagement
from datetime import datetime

# setup drawing mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# buka kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # deteksi pose
    results = detect_pose(frame)
    data = analyze_pose(results)

    # sementara 1 orang
    total = 1

    # hitung engagement
    engagement = calculate_engagement(
        total,
        data["angkat_tangan"],
        data["menghadap_depan"]
    )

    # buat output JSON
    output = {
        "total_siswa": total,
        "angkat_tangan": data["angkat_tangan"],
        "menghadap_depan": data["menghadap_depan"],
        "engagement_score": engagement,
        "timestamp": datetime.now().isoformat()
    }

    # print ke terminal
    print(output)

    # =========================
    # 🎨 VISUALISASI (SKELETON)
    # =========================
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # =========================
    # 📊 TAMPILKAN INFO DI LAYAR
    # =========================
    cv2.putText(frame, f"Tangan: {data['angkat_tangan']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Hadap: {data['menghadap_depan']}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.putText(frame, f"Engagement: {engagement}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # tampilkan kamera
    cv2.imshow("CV Module", frame)

    # tekan ESC untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

# cleanup
cap.release()
cv2.destroyAllWindows()