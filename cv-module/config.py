# config.py

# =========================
# KAMERA
# =========================
CAMERA_INDEX  = 1
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
BACKEND_URL   = "http://127.0.0.1:5000/api/data"  # URL endpoint Flask lokal
SEND_INTERVAL = 2                                   # detik antar pengiriman data