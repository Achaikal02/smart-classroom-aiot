import sqlite3

conn = sqlite3.connect("database.db")
cur = conn.cursor()

# tabel guru
cur.execute("""
CREATE TABLE IF NOT EXISTS teachers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nama TEXT NOT NULL,
    nip TEXT,
    face_encoding BLOB
)
""")

# tabel sesi monitoring
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

# tabel log realtime
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

conn.commit()
conn.close()

print("Database berhasil dibuat")