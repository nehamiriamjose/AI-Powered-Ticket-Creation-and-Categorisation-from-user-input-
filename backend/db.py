import sqlite3
from datetime import datetime

DB_NAME = "tickets.db"


def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


def create_tables():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT,
            category TEXT,
            priority TEXT,
            status TEXT DEFAULT 'Open',
            created_at TEXT,
            updated_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT,
            type TEXT,
            created_at TEXT,
            is_read INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()


def insert_ticket(title, description, category, priority):
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat()

    cur.execute("""
        INSERT INTO tickets (title, description, category, priority, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'Open', ?, ?)
    """, (title, description, category, priority, now, now))

    conn.commit()
    conn.close()


def fetch_all_tickets():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, description, category, priority, status, created_at
        FROM tickets ORDER BY created_at DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return rows
