import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "database.db")

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

# ── tabel guru ──────────────────────────────────────────────
cur.execute("""
CREATE TABLE IF NOT EXISTS teachers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nama TEXT NOT NULL,
    nip TEXT,
    face_encoding BLOB
)
""")

# ── tabel sesi monitoring ────────────────────────────────────
cur.execute("""
CREATE TABLE IF NOT EXISTS sessions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_id INTEGER,
    start_time TEXT,
    end_time TEXT,
    avg_engagement REAL,
    avg_raise_hand REAL,
    avg_focus REAL,
    avg_down REAL,
    FOREIGN KEY(teacher_id) REFERENCES teachers(id)
)
""")

# ── tabel log realtime ───────────────────────────────────────
cur.execute("""
CREATE TABLE IF NOT EXISTS engagement_logs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    timestamp TEXT,
    total_siswa INTEGER,
    angkat_tangan INTEGER,
    menghadap_depan INTEGER,
    menunduk INTEGER,
    engagement_score REAL,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
)
""")

# ── tabel siswa ──────────────────────────────────────────────
cur.execute("""
CREATE TABLE IF NOT EXISTS students(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nama TEXT NOT NULL,
    kelas TEXT NOT NULL,
    nisn TEXT,
    foto_path TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""")

# ── tabel absensi ────────────────────────────────────────────
cur.execute("""
CREATE TABLE IF NOT EXISTS attendance(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    tanggal TEXT NOT NULL,
    waktu TEXT,
    status TEXT DEFAULT 'hadir',
    keterangan TEXT,
    created_by TEXT,
    FOREIGN KEY(student_id) REFERENCES students(id)
)
""")

conn.commit()
conn.close()

print("✅ Database berhasil dibuat / diperbarui")
print(f"   Lokasi: {DB_PATH}")
print("   Tabel  : teachers, sessions, engagement_logs, students, attendance")