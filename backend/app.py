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

def normalize_face(face):
    if face is None or getattr(face, 'size', 0) == 0:
        return None
    if len(face.shape) == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    try:
        face = cv2.resize(face, (200, 200))
    except Exception:
        return None
    return cv2.equalizeHist(face)


def extract_face_gray(bgr_frame):
    """Deteksi wajah, kembalikan (crop_gray, x,y,w,h) atau None."""
    if bgr_frame is None or getattr(bgr_frame, 'size', 0) == 0:
        return None, None
    try:
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None, None
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.05, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])  # ambil wajah terbesar
    crop = gray[y:y+h, x:x+w]
    return normalize_face(crop), (x, y, w, h)


def load_student_recognizer():
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT id, nama, foto_path FROM students WHERE foto_path IS NOT NULL")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    faces = []
    labels = []
    label_map = {}
    for row in rows:
        foto_path = row["foto_path"]
        if not foto_path or not os.path.exists(foto_path):
            continue
        img = cv2.imread(foto_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        face, _ = extract_face_gray(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        if face is None:
            face = normalize_face(img)
        if face is None:
            continue
        faces.append(face)
        labels.append(int(row["id"]))
        label_map[int(row["id"])] = row["nama"]

    if not faces:
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    return recognizer, label_map

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
# ════════════════════════════════════════════
#  DATABASE HELPER
# ════════════════════════════════════════════
import sqlite3

DB_PATH     = os.path.join(BASE_DIR, "database.db")
STUDENT_DIR = os.path.join(ROOT_DIR, "dataset", "siswa")
os.makedirs(STUDENT_DIR, exist_ok=True)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ════════════════════════════════════════════
#  ENDPOINT: ABSENSI SISWA
# ════════════════════════════════════════════

@app.route("/api/students/register", methods=["POST"])
def register_student():
    data  = request.get_json() or {}
    nama  = data.get("nama",  "").strip()
    kelas = data.get("kelas", "").strip()
    nisn  = data.get("nisn",  "").strip()
    foto  = data.get("foto",  "")
    if not nama or not kelas:
        return jsonify({"status": "error", "message": "Nama dan kelas wajib diisi"}), 400
    foto_path = None
    if foto:
        try:
            safe_name = nama.replace(" ", "_")
            fpath     = os.path.join(STUDENT_DIR, f"{safe_name}.jpg")
            frame     = decode_bgr(foto)
            cv2.imwrite(fpath, frame)
            foto_path = fpath
        except Exception as e:
            print(f"[REGISTER] Gagal simpan foto: {e}")
    conn = get_db()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO students (nama, kelas, nisn, foto_path) VALUES (?, ?, ?, ?)",
        (nama.title(), kelas.upper(), nisn, foto_path)
    )
    conn.commit()
    student_id = cur.lastrowid
    conn.close()
    return jsonify({"status": "ok", "id": student_id, "nama": nama.title()})


@app.route("/api/students", methods=["GET"])
def list_students():
    kelas = request.args.get("kelas", "").strip()
    conn  = get_db()
    cur   = conn.cursor()
    if kelas:
        cur.execute(
            "SELECT id, nama, kelas, nisn, foto_path, created_at FROM students WHERE kelas=? ORDER BY nama",
            (kelas.upper(),)
        )
    else:
        cur.execute(
            "SELECT id, nama, kelas, nisn, foto_path, created_at FROM students ORDER BY kelas, nama"
        )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify({"students": rows})


@app.route("/api/students/recognize", methods=["POST"])
def recognize_student():
    data = request.get_json() or {}
    image = data.get("image", "")
    if not image:
        return jsonify({"status": "error", "message": "Gambar tidak ditemukan"}), 400

    if image.startswith("data:"):
        parts = image.split(",", 1)
        if len(parts) == 2:
            image = parts[1]

    try:
        frame = decode_bgr(image)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Gagal decode gambar: {e}"}), 400

    if frame is None or getattr(frame, 'size', 0) == 0:
        return jsonify({"status": "error", "message": "Gambar tidak valid"}), 400

    crop, _ = extract_face_gray(frame)
    if crop is None:
        return jsonify({"status": "unknown", "message": "Wajah tidak terdeteksi"})

    recognizer, label_map = load_student_recognizer()
    if recognizer is None:
        return jsonify({"status": "unknown", "message": "Tidak ada foto siswa untuk pengenalan"})

    try:
        crop = normalize_face(crop)
        if crop is None:
            return jsonify({"status": "unknown", "message": "Wajah tidak valid"})
        label, confidence = recognizer.predict(crop)
        print(f"[RECOGNIZE] label={label}, confidence={confidence}")
    except Exception as e:
        return jsonify({"status": "error", "message": f"Pengenalan gagal: {e}"}), 500

    THRESHOLD = 80
    if confidence <= THRESHOLD and label in label_map:
        match_pct = round(max(0, (THRESHOLD - confidence) / THRESHOLD * 100), 1)
        return jsonify({
            "status": "ok",
            "student_id": int(label),
            "nama": label_map[label],
            "confidence": match_pct
        })
    return jsonify({"status": "unknown", "confidence": round(confidence, 1)})


@app.route("/api/attendance/mark", methods=["POST"])
def mark_attendance():
    data       = request.get_json() or {}
    student_id = data.get("student_id")
    status     = data.get("status",     "hadir")
    keterangan = data.get("keterangan", "")
    tanggal    = datetime.now().strftime("%Y-%m-%d")
    waktu      = datetime.now().strftime("%H:%M:%S")
    guru       = logged_in_user or "—"
    if not student_id:
        return jsonify({"status": "error", "message": "student_id wajib"}), 400
    conn = get_db()
    cur  = conn.cursor()
    cur.execute(
        "SELECT id, status FROM attendance WHERE student_id=? AND tanggal=?",
        (student_id, tanggal)
    )
    existing = cur.fetchone()
    if existing:
        cur.execute(
            "UPDATE attendance SET status=?, keterangan=?, waktu=? WHERE id=?",
            (status, keterangan, waktu, existing["id"])
        )
        conn.commit()
        conn.close()
        return jsonify({"status": "ok", "action": "updated", "tanggal": tanggal, "waktu": waktu})
    cur.execute(
        "INSERT INTO attendance (student_id, tanggal, waktu, status, keterangan, created_by) VALUES (?,?,?,?,?,?)",
        (student_id, tanggal, waktu, status, keterangan, guru)
    )
    conn.commit()
    conn.close()
    return jsonify({"status": "ok", "action": "created", "tanggal": tanggal, "waktu": waktu})


@app.route("/api/attendance/bulk", methods=["POST"])
def bulk_attendance():
    data    = request.get_json() or {}
    records = data.get("records", [])
    tanggal = datetime.now().strftime("%Y-%m-%d")
    waktu   = datetime.now().strftime("%H:%M:%S")
    guru    = logged_in_user or "—"
    if not records:
        return jsonify({"status": "error", "message": "records kosong"}), 400
    conn = get_db()
    cur  = conn.cursor()
    saved = 0
    for rec in records:
        sid    = rec.get("student_id")
        status = rec.get("status", "hadir")
        ket    = rec.get("keterangan", "")
        if not sid:
            continue
        cur.execute("SELECT id FROM attendance WHERE student_id=? AND tanggal=?", (sid, tanggal))
        row = cur.fetchone()
        if row:
            cur.execute(
                "UPDATE attendance SET status=?, keterangan=?, waktu=? WHERE id=?",
                (status, ket, waktu, row["id"])
            )
        else:
            cur.execute(
                "INSERT INTO attendance (student_id, tanggal, waktu, status, keterangan, created_by) VALUES (?,?,?,?,?,?)",
                (sid, tanggal, waktu, status, ket, guru)
            )
        saved += 1
    conn.commit()
    conn.close()
    return jsonify({"status": "ok", "saved": saved, "tanggal": tanggal})


@app.route("/api/attendance", methods=["GET"])
def get_attendance():
    tanggal = request.args.get("tanggal", datetime.now().strftime("%Y-%m-%d"))
    kelas   = request.args.get("kelas",   "").strip()
    conn  = get_db()
    cur   = conn.cursor()
    query = """
        SELECT s.id, s.nama, s.kelas, s.nisn,
               COALESCE(a.status, 'belum') AS status,
               a.waktu, a.keterangan
        FROM   students s
        LEFT   JOIN attendance a
               ON  s.id = a.student_id
               AND a.tanggal = ?
    """
    params = [tanggal]
    if kelas:
        query  += " WHERE s.kelas = ?"
        params.append(kelas.upper())
    query += " ORDER BY s.kelas, s.nama"
    cur.execute(query, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    total = len(rows)
    return jsonify({
        "tanggal":    tanggal,
        "attendance": rows,
        "summary": {
            "total": total,
            "hadir": sum(1 for r in rows if r["status"] == "hadir"),
            "sakit": sum(1 for r in rows if r["status"] == "sakit"),
            "izin":  sum(1 for r in rows if r["status"] == "izin"),
            "alpha": sum(1 for r in rows if r["status"] == "alpha"),
            "belum": sum(1 for r in rows if r["status"] == "belum"),
        }
    })


@app.route("/api/students/<int:student_id>", methods=["DELETE"])
def delete_student(student_id):
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("DELETE FROM attendance WHERE student_id=?", (student_id,))
    cur.execute("DELETE FROM students    WHERE id=?",        (student_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})


@app.route("/api/classes", methods=["GET"])
def list_classes():
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT DISTINCT kelas FROM students ORDER BY kelas")
    classes = [r["kelas"] for r in cur.fetchall()]
    conn.close()
    return jsonify({"classes": classes})

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