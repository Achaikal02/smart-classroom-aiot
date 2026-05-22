from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from collections import deque
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

MAX_HISTORY = 50
latest_data = {
    "total_siswa": 0, "angkat_tangan": 0,
    "menghadap_depan": 0, "menunduk": 0,
    "engagement_score": 0.0,
    "timestamp": datetime.now().isoformat()
}
history = deque(maxlen=MAX_HISTORY)
session_start = datetime.now().isoformat()

@app.route("/api/data", methods=["POST"])
def receive_data():
    global latest_data
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    latest_data = data
    history.append({
        "time": data.get("timestamp", "")[-8:-3],
        "value": round(float(data.get("engagement_score", 0)), 1)
    })
    return jsonify({"status": "ok"}), 200

@app.route("/api/latest", methods=["GET"])
def get_latest():
    return jsonify({**latest_data, "history": list(history), "session_start": session_start})

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

if __name__ == "__main__":
    print("="*50)
    print("  Backend berjalan")
    print("  Dashboard: http://127.0.0.1:5000")
    print("="*50)
    app.run(host="0.0.0.0", port=5000, debug=False)
