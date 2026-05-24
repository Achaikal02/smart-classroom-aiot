# backend/app.py
# Flask backend — menerima data dari mainYOLO.py dan menyajikannya ke dashboard

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from collections import deque
from datetime import datetime
import os
import subprocess
import sys

app = Flask(__name__)
CORS(app)  # izinkan request dari frontend (index.html)

# ── Penyimpanan data in-memory ───────────────────────────────────
MAX_HISTORY = 50
latest_data = {
    "total_siswa":      0,
    "angkat_tangan":    0,
    "menghadap_depan":  0,
    "menunduk":         0,
    "engagement_score": 0.0,
    "timestamp":        datetime.now().isoformat()
}
history      = deque(maxlen=MAX_HISTORY)
session_start = datetime.now().isoformat()

# ── State kamera ─────────────────────────────────────────────────
camera_running = False
camera_process = None   # subprocess mainYOLO.py

# Path ke mainYOLO.py (satu level di atas backend/, masuk cv-module/)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))          # …/backend
YOLO_SCRIPT = os.path.join(BASE_DIR, "..", "cv-module", "mainYOLO.py")

# ── Endpoint: terima data dari mainYOLO.py ───────────────────────
@app.route("/api/data", methods=["POST"])
def receive_data():
    global latest_data
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    latest_data = data
    history.append({
        "time":  data.get("timestamp", "")[-8:-3],  # HH:MM
        "value": round(float(data.get("engagement_score", 0)), 1)
    })
    return jsonify({"status": "ok"}), 200

# ── Endpoint: ambil data terbaru untuk dashboard ─────────────────
@app.route("/api/latest", methods=["GET"])
def get_latest():
    return jsonify({
        **latest_data,
        "history":        list(history),
        "session_start":  session_start,
        "camera_running": camera_running,
    })

# ── Endpoint: start / stop kamera ───────────────────────────────
@app.route("/api/control", methods=["POST"])
def control_camera():
    global camera_running, camera_process, latest_data, history, session_start

    action = (request.get_json() or {}).get("action", "")

    # ---- START ----
    if action == "start":
        if camera_running:
            return jsonify({"status": "already_running"}), 200

        # Reset state sesi
        latest_data = {
            "total_siswa": 0, "angkat_tangan": 0,
            "menghadap_depan": 0, "menunduk": 0,
            "engagement_score": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        history.clear()
        session_start = datetime.now().isoformat()

        script_path = os.path.abspath(YOLO_SCRIPT)
        if not os.path.exists(script_path):
            return jsonify({"error": f"mainYOLO.py tidak ditemukan di {script_path}"}), 500

        try:
            camera_process = subprocess.Popen(
                [sys.executable, script_path],
                cwd=os.path.dirname(script_path)
            )
            camera_running = True
            return jsonify({"status": "started"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ---- STOP ----
    elif action == "stop":
        if not camera_running:
            return jsonify({"status": "already_stopped"}), 200

        if camera_process and camera_process.poll() is None:
            camera_process.terminate()
            try:
                camera_process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                camera_process.kill()

        camera_process  = None
        camera_running  = False
        return jsonify({"status": "stopped"}), 200

    return jsonify({"error": "action harus 'start' atau 'stop'"}), 400

# ── Endpoint: cek status kamera ─────────────────────────────────
@app.route("/api/status", methods=["GET"])
def get_status():
    global camera_running, camera_process
    # subprocess bisa mati sendiri (mis. ESC ditekan di jendela YOLO)
    if camera_process and camera_process.poll() is not None:
        camera_running = False
        camera_process = None
    return jsonify({"running": camera_running}), 200

# ── Sajikan frontend (index.html) ────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# ── Jalankan ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Smart Classroom Backend berjalan")
    print("  Dashboard: http://127.0.0.1:5000")
    print("  API data : http://127.0.0.1:5000/api/latest")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)