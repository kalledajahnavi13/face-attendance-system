import sqlite3
from datetime import datetime

DB_NAME = "attendance.db"


def create_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def add_student(student_id, name):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO students (student_id, name) VALUES (?, ?)",
            (student_id, name)
        )
        conn.commit()
        result = True
    except sqlite3.IntegrityError:
        result = False

    conn.close()
    return result


def get_students():
    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()
    cursor.execute("SELECT student_id, name FROM students")

    students = cursor.fetchall()

    conn.close()
    return students


def mark_attendance(student_id, name):
    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    cursor.execute("""
        SELECT * FROM attendance
        WHERE student_id = ? AND date = ?
    """, (student_id, today))

    already_marked = cursor.fetchone()

    if already_marked:
        conn.close()
        return False

    cursor.execute("""
        INSERT INTO attendance
        (student_id, name, date, time)
        VALUES (?, ?, ?, ?)
    """, (student_id, name, today, current_time))

    conn.commit()
    conn.close()

    return True


def get_attendance():
    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT student_id, name, date, time
        FROM attendance
        ORDER BY date DESC, time DESC
    """)

    data = cursor.fetchall()

    conn.close()

    return data