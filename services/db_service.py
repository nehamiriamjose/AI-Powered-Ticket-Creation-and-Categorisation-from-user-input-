# DB read/update helpers
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "tickets.db"


# -------------------------------------------------
# DB CONNECTION
# -------------------------------------------------
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


# -------------------------------------------------
# USERS TABLE (LOGIN SUPPORT)
# -------------------------------------------------
def create_users_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            role TEXT
        )
    """)

    # Insert demo users (safe: won't duplicate)
    cursor.execute(
        "INSERT OR IGNORE INTO users (username, role) VALUES (?, ?)",
        ("user1", "User")
    )
    cursor.execute(
        "INSERT OR IGNORE INTO users (username, role) VALUES (?, ?)",
        ("support1", "Support")
    )
    cursor.execute(
        "INSERT OR IGNORE INTO users (username, role) VALUES (?, ?)",
        ("admin1", "Admin")
    )

    conn.commit()
    conn.close()


def get_user_by_username(username):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT username, role FROM users WHERE username = ?",
        (username,)
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "username": row[0],
            "role": row[1]
        }
    return None


# -------------------------------------------------
# TICKETS TABLE
# -------------------------------------------------
def init_db():
    """Initialize database tables if they don't exist"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            priority TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Open',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

    # ✅ IMPORTANT: create users table ALSO
    create_users_table()


def fetch_tickets():
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM tickets ORDER BY created_at DESC",
        conn
    )
    conn.close()
    return df


def update_ticket_status(ticket_id, new_status):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE tickets
        SET status = ?, updated_at = ?
        WHERE id = ?
        """,
        (new_status, datetime.now().isoformat(), ticket_id)
    )

    conn.commit()
    conn.close()
