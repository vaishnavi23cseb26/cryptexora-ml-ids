"""
database.py
-----------
SQLite helpers. Two tables:
  users       - login accounts (passwords are stored HASHED, never as plain text)
  detections  - one row for every traffic analysis (the detection history)

Run  python database.py  to create the database manually.
(app.py also does this automatically when it starts.)
"""
import os
import sqlite3

from flask import current_app, g
from werkzeug.security import generate_password_hash

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'Security Analyst'
);

CREATE TABLE IF NOT EXISTS detections (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at         TEXT NOT NULL,
    username           TEXT NOT NULL,
    ip_address         TEXT NOT NULL,
    duration           REAL NOT NULL,
    packet_count       INTEGER NOT NULL,
    bytes_transferred  INTEGER NOT NULL,
    failed_connections INTEGER NOT NULL,
    prediction         TEXT NOT NULL,      -- 'normal' or 'suspicious'
    confidence         REAL NOT NULL,      -- percent
    severity           TEXT NOT NULL,      -- 'low', 'medium' or 'high'
    details            TEXT NOT NULL
);
"""


def get_db():
    """One database connection per request."""
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row  # rows behave like dictionaries
    return g.db


def close_db(error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create tables and the demo admin account (only if it does not exist)."""
    db = get_db()
    db.executescript(SCHEMA)
    admin_exists = db.execute("SELECT 1 FROM users WHERE username = 'admin'").fetchone()
    if not admin_exists:
        # The demo password can be changed with the ADMIN_PASSWORD environment variable.
        password = os.environ.get("ADMIN_PASSWORD", "admin123")
        db.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            ("admin", generate_password_hash(password, method="pbkdf2:sha256")),
        )
    db.commit()


if __name__ == "__main__":
    from app import app
    with app.app_context():
        init_db()
    print("Database ready:", app.config["DATABASE"])
