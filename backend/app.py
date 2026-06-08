# backend/app.py
# Face recognition pakai OpenCV LBPH — tanpa TensorFlow/dlib/DeepFace
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from collections import deque
from datetime import datetime
import os, subprocess, sys, json, base64, re
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

# ════════════════════════════════════════════
#  PATH
# ════════════════════════════════════════════
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.join(BASE_DIR, "..")
DATASET_DIR  = os.path.join(ROOT_DIR, "dataset", "guru")
REPORT_DIR   = os.path.join(ROOT_DIR, "dataset", "laporan")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
YOLO_SCRIPT  = os.path.join(ROOT_DIR, "cv-module", "mainYOLO.py")
MODEL_FILE   = os.path.join(ROOT_DIR, "dataset", "guru_model.yml")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(REPORT_DIR,  exist_ok=True)

# ════════════════════════════════════════════
#  FACE DETECTOR (Haar Cascade — bawaan OpenCV)
# ════════════════════════════════════════════
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ════════════════════════════════════════════
#  LBPH RECOGNIZER
# ════════════════════════════════════════════
recognizer  = cv2.face.LBPHFaceRecognizer_create()
label_map   = {}   # { label_int: nama_str }
model_ready = False

def get_label(nama):
    """Kembalikan label int untuk nama guru, buat baru jika belum ada."""
    for lbl, n in label_map.items():
        if n.lower() == nama.lower():
            return lbl
    new_lbl = max(label_map.keys(), default=-1) + 1
    label_map[new_lbl] = nama
    return new_lbl

def load_or_train_model():
    """Load model dari file YML, atau train ulang dari foto di dataset/guru/."""
    global model_ready
    exts = ('.jpg', '.jpeg', '.png', '.webp')

    photos = [f for f in os.listdir(DATASET_DIR)
              if f.lower().endswith(exts) and not f.startswith('_')]

    if not photos:
        model_ready = False
        print("[INFO] Dataset kosong — model belum dibuat.")
        return

    faces_data = []
    labels_data = []

    for fname in photos:
        # Nama dari file: Faleza_Yassinia.jpg → "Faleza Yassinia"
        nama = os.path.splitext(fname)[0].replace('_', ' ').title()
        nama = re.sub(r'\s+\d+$', '', nama).strip()  # hapus suffix angka
        lbl  = get_label(nama)

        fpath = os.path.join(DATASET_DIR, fname)
        img   = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        detected = face_cascade.detectMultiScale(img, 1.1, 5, minSize=(60, 60))
        if len(detected) == 0:
            # Tidak ada wajah terdeteksi — pakai seluruh gambar resize
            img_resized = cv2.resize(img, (200, 200))
            faces_data.append(img_resized)
            labels_data.append(lbl)
        else:
            x, y, w, h = detected[0]
            face_crop = cv2.resize(img[y:y+h, x:x+w], (200, 200))
            faces_data.append(face_crop)
            labels_data.append(lbl)

    if faces_data:
        recognizer.train(faces_data, np.array(labels_data))
        recognizer.save(MODEL_FILE)
        # Simpan label map ke JSON
        lmap_path = MODEL_FILE.replace('.yml', '_labels.json')
        with open(lmap_path, 'w') as f:
            json.dump({str(k): v for k, v in label_map.items()}, f)
        model_ready = True
        print(f"[OK] Model LBPH dilatih: {len(faces_data)} foto, {len(set(labels_data))} guru")
    else:
        model_ready = False
        print("[WARNING] Tidak ada wajah valid di dataset.")

def load_model_from_file():
    """Muat model YML yang sudah ada."""
    global model_ready
    lmap_path = MODEL_FILE.replace('.yml', '_labels.json')
    if os.path.exists(MODEL_FILE) and os.path.exists(lmap_path):
        recognizer.read(MODEL_FILE)
        with open(lmap_path) as f:
            raw = json.load(f)
            label_map.update({int(k): v for k, v in raw.items()})
        model_ready = True
        print(f"[OK] Model LBPH dimuat: {label_map}")
    else:
        load_or_train_model()

load_model_from_file()

# ════════════════════════════════════════════
#  HELPER
# ════════════════════════════════════════════
def decode_bgr(b64str):
    arr = np.frombuffer(base64.b64decode(b64str), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def extract_face_gray(bgr_frame):
    """Deteksi wajah, kembalikan (crop_gray, x,y,w,h) atau None."""
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])  # ambil wajah terbesar
    crop = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
    return crop, (x, y, w, h)

# ════════════════════════════════════════════
#  STATE
# ════════════════════════════════════════════
MAX_HISTORY    = 50
latest_data    = {"total_siswa":0,"angkat_tangan":0,"menghadap_depan":0,
                  "menunduk":0,"engagement_score":0.0,"timestamp":datetime.now().isoformat()}
history        = deque(maxlen=MAX_HISTORY)
session_start  = datetime.now().isoformat()
logged_in_user = None
camera_running = False
camera_process = None

# ════════════════════════════════════════════
#  ENDPOINT: LOGIN
# ════════════════════════════════════════════
@app.route("/api/login", methods=["POST"])
def login_face():
    global logged_in_user

    b64 = (request.get_json() or {}).get("image")
    if not b64:
        return jsonify({"status":"error","message":"Tidak ada data gambar"}), 400

    exts   = ('.jpg','.jpeg','.png','.webp')
    photos = [f for f in os.listdir(DATASET_DIR)
              if f.lower().endswith(exts) and not f.startswith('_')]
    if not photos:
        return jsonify({"status":"no_dataset","face_detected":False})

    if not model_ready:
        return jsonify({"status":"error","message":"Model belum siap"}), 500

    try:
        frame = decode_bgr(b64)
        crop, bbox = extract_face_gray(frame)
        if crop is None:
            return jsonify({"status":"unknown","face_detected":False})

        label, confidence = recognizer.predict(crop)
        # LBPH: semakin kecil confidence semakin cocok (0 = sempurna)
        # Threshold 80 cukup ketat, naikkan ke 100 kalau terlalu strict
        THRESHOLD = 85
        print(f"[LOGIN] label={label}, confidence={confidence:.1f}")

        if confidence <= THRESHOLD and label in label_map:
            nama = label_map[label]
            logged_in_user = nama
            match_pct = round(max(0, (THRESHOLD - confidence) / THRESHOLD * 100), 1)
            return jsonify({"status":"ok","name":nama,"confidence":match_pct})
        else:
            return jsonify({"status":"unknown","face_detected":True,
                            "confidence":round(confidence,1)})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500


# ════════════════════════════════════════════
#  ENDPOINT: ENROLL
# ════════════════════════════════════════════
@app.route("/api/enroll", methods=["POST"])
def enroll_guru():
    global logged_in_user, model_ready

    data = request.get_json() or {}
    nama = data.get("nama","").strip()
    imgs = data.get("images", [])

    if not nama:
        return jsonify({"status":"error","message":"Nama tidak boleh kosong"}), 400
    if not imgs:
        return jsonify({"status":"error","message":"Tidak ada foto"}), 400

    safe_name   = nama.replace(' ','_')
    saved_count = 0

    for i, b64 in enumerate(imgs):
        try:
            frame = decode_bgr(b64)
            crop, bbox = extract_face_gray(frame)

            # Simpan frame asli (bukan crop) sebagai referensi foto
            fname = f"{safe_name}.jpg" if saved_count == 0 else f"{safe_name}_{saved_count}.jpg"
            fpath = os.path.join(DATASET_DIR, fname)
            c = 1
            while os.path.exists(fpath):
                fname = f"{safe_name}_{saved_count}_{c}.jpg"
                fpath = os.path.join(DATASET_DIR, fname)
                c += 1

            cv2.imwrite(fpath, frame)
            saved_count += 1
            print(f"[ENROLL] Tersimpan: {fname}" + (" (tanpa wajah terdeteksi)" if crop is None else ""))

        except Exception as e:
            print(f"[ENROLL] Frame {i} error: {e}")
            continue

    if saved_count == 0:
        return jsonify({"status":"error","message":"Gagal menyimpan foto"}), 400

    # Latih ulang model dengan data baru
    label_map.clear()
    load_or_train_model()

    logged_in_user = nama.title()
    print(f"[ENROLL] '{nama}' terdaftar dengan {saved_count} foto.")
    return jsonify({"status":"ok","name":nama.title(),"saved":saved_count})


# ════════════════════════════════════════════
#  ENDPOINT: DATA YOLO
# ════════════════════════════════════════════
@app.route("/api/data", methods=["POST"])
def receive_data():
    global latest_data
    data = request.get_json()
    if not data: return jsonify({"error":"No data"}), 400
    latest_data = data
    history.append({
        "time":  data.get("timestamp","")[-8:-3],
        "value": round(float(data.get("engagement_score",0)),1)
    })
    return jsonify({"status":"ok"}), 200

@app.route("/api/latest", methods=["GET"])
def get_latest():
    return jsonify({**latest_data,"history":list(history),
                    "session_start":session_start,
                    "camera_running":camera_running,
                    "logged_in_user":logged_in_user})

# ════════════════════════════════════════════
#  LAPORAN
# ════════════════════════════════════════════
@app.route("/api/save_report", methods=["POST"])
def save_report():
    data = request.get_json() or {}
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "guru":              data.get("guru", logged_in_user or "—"),
        "tanggal":           datetime.now().strftime("%d %B %Y"),
        "waktu":             datetime.now().strftime("%H:%M:%S"),
        "durasi_menit":      data.get("durasi", 0),
        "avg_engagement":    data.get("avg_engagement", 0),
        "avg_angkat_tangan": data.get("avg_angkat", 0),
        "avg_menunduk":      data.get("avg_nunduk", 0),
        "total_data":        data.get("total_data", 0),
        "verdict":           data.get("verdict", "—"),
        "history":           data.get("history", []),
    }
    fname = f"laporan_{ts}.json"
    with open(os.path.join(REPORT_DIR, fname), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Laporan disimpan: {fname}")
    return jsonify({"status":"ok","filename":fname})

@app.route("/api/download_report/<filename>", methods=["GET"])
def download_report(filename):
    fpath = os.path.join(REPORT_DIR, filename)
    if not os.path.exists(fpath):
        return jsonify({"error":"File tidak ditemukan"}), 404
    return send_file(fpath, as_attachment=True,
                     download_name=filename, mimetype="application/json")

@app.route("/api/reports", methods=["GET"])
def list_reports():
    files = sorted([f for f in os.listdir(REPORT_DIR)
                    if f.endswith(".json")], reverse=True)
    return jsonify({"files": files})

# ════════════════════════════════════════════
#  KONTROL KAMERA
# ════════════════════════════════════════════
@app.route("/api/control", methods=["POST"])
def control_camera():
    global camera_running, camera_process, latest_data, history, session_start
    action = (request.get_json() or {}).get("action","")

    if action == "start":
        if camera_running: return jsonify({"status":"already_running"}), 200
        latest_data = {"total_siswa":0,"angkat_tangan":0,"menghadap_depan":0,
                       "menunduk":0,"engagement_score":0.0,
                       "timestamp":datetime.now().isoformat()}
        history.clear()
        session_start = datetime.now().isoformat()
        script_path = os.path.abspath(YOLO_SCRIPT)
        if not os.path.exists(script_path):
            return jsonify({"error":f"mainYOLO.py tidak ditemukan: {script_path}"}), 500
        try:
            camera_process = subprocess.Popen(
                [sys.executable, script_path],
                cwd=os.path.dirname(script_path))
            camera_running = True
            return jsonify({"status":"started"}), 200
        except Exception as e:
            return jsonify({"error":str(e)}), 500

    elif action == "stop":
        if not camera_running: return jsonify({"status":"already_stopped"}), 200
        if camera_process and camera_process.poll() is None:
            camera_process.terminate()
            try: camera_process.wait(timeout=4)
            except subprocess.TimeoutExpired: camera_process.kill()
        camera_process = None; camera_running = False
        return jsonify({"status":"stopped"}), 200

    return jsonify({"error":"action harus 'start' atau 'stop'"}), 400

@app.route("/api/status", methods=["GET"])
def get_status():
    global camera_running, camera_process
    if camera_process and camera_process.poll() is not None:
        camera_running = False; camera_process = None
    return jsonify({"running": camera_running}), 200

# ════════════════════════════════════════════
#  FRONTEND
# ════════════════════════════════════════════
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "login.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

if __name__ == "__main__":
    guru_list = list(label_map.values()) or ['Kosong — gunakan self-enroll']
    print("=" * 55)
    print("  Smart Classroom Backend (OpenCV LBPH)")
    print(f"  Dataset guru : {DATASET_DIR}")
    print(f"  Laporan      : {REPORT_DIR}")
    print(f"  Guru terdaftar: {', '.join(guru_list)}")
    print("  Buka browser : http://127.0.0.1:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False)