import sqlite3
from datetime import datetime

DB_PATH = "tickets.db"


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


import sqlite3
from datetime import datetime

DB_PATH = "tickets.db"


def add_notification(message, n_type="info"):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO notifications (message, type, created_at)
        VALUES (?, ?, ?)
    """, (message, n_type, datetime.now().isoformat()))

    conn.commit()
    conn.close()

def fetch_notifications():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, message, type, created_at
        FROM notifications
        WHERE is_read = 0
        ORDER BY created_at DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def mark_as_read(notification_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE notifications SET is_read = 1 WHERE id = ?",
        (notification_id,)
    )
    conn.commit()
    conn.close()
