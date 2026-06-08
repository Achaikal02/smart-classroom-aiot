# config.py

# =========================
# KAMERA
# =========================
CAMERA_INDEX  = 0
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480

# =========================
# TRACKER
# =========================
MAX_MISS_FRAMES      = 3
DEBOUNCE_FRAMES      = 3
HISTORY_SIZE         = 10
DUPLICATE_THRESHOLD  = 80

# =========================
# YOLO
# =========================
YOLO_MODEL       = "yolov8n.pt"
YOLO_CONFIDENCE  = 0.5
MIN_BOX_WIDTH    = 100
MIN_BOX_HEIGHT   = 150

# =========================
# MEDIAPIPE POSE
# =========================
POSE_DETECTION_CONFIDENCE = 0.5
POSE_TRACKING_CONFIDENCE  = 0.5
POSE_MODEL_COMPLEXITY     = 1

# =========================
# THRESHOLD DETEKSI
# =========================
HAND_RAISE_THRESHOLD  = 0.05
FACE_CENTER_THRESHOLD = 0.15
NUNDUK_THRESHOLD      = 0.1

# =========================
# ENGAGEMENT
# =========================
WEIGHT_TANGAN        = 0.6
WEIGHT_HADAP         = 0.4
ENGAGEMENT_SMOOTHING = 0.85

# =========================
# BACKEND FLASK (dashboard)
# =========================
BACKEND_URL                  = "http://127.0.0.1:5000/api/data"          # URL endpoint Flask lokal
BACKEND_STUDENTS_URL         = "http://127.0.0.1:5000/api/students"      # daftar siswa terdaftar
BACKEND_ATTENDANCE_MARK_URL  = "http://127.0.0.1:5000/api/attendance/mark"
SEND_INTERVAL                = 2                                       # detik antar pengiriman data
STUDENT_CONFIDENCE_THRESHOLD = 80
FACE_MIN_SIZE                = (60, 60)
STUDENT_MODEL_REFRESH_SEC    = 30
ATTENDANCE_COOLDOWN_SEC      = 5