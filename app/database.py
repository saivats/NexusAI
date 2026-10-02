import sqlite3
import json
import time
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent.parent / "nexusai.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    domain TEXT,
    confidence REAL,
    calibrated_confidence REAL,
    action TEXT,
    faq_id TEXT,
    latency_ms REAL,
    resolved INTEGER DEFAULT 0,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    domain_guess TEXT,
    status TEXT DEFAULT 'open',
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    faq_id TEXT NOT NULL,
    rating INTEGER NOT NULL,
    timestamp REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_logs_session ON query_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_logs_domain ON query_logs(domain);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_feedback_faq ON feedback(faq_id);
"""


def _get_connection():
    conn = sqlite3.connect(str(DB_PATH), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = _get_connection()
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_db():
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def log_query(request_id, session_id, query, domain, confidence,
              calibrated_confidence, action, faq_id, latency_ms):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO query_logs
               (request_id, session_id, query, domain, confidence,
                calibrated_confidence, action, faq_id, latency_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (request_id, session_id, query, domain, confidence,
             calibrated_confidence, action, faq_id, latency_ms, time.time()),
        )


def log_sensitive_query(request_id, session_id):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO query_logs
               (request_id, session_id, query, domain, confidence,
                calibrated_confidence, action, faq_id, latency_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (request_id, session_id, "[REDACTED]", "sensitive", 1.0,
             1.0, "sensitive", None, 0, time.time()),
        )


def create_ticket(ticket_id, session_id, query, domain_guess):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO tickets (id, session_id, query, domain_guess, status, timestamp)
               VALUES (?, ?, ?, ?, 'open', ?)""",
            (ticket_id, session_id, query, domain_guess, time.time()),
        )


def get_tickets():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM tickets ORDER BY timestamp DESC LIMIT 100"
        ).fetchall()
        return [dict(r) for r in rows]


def add_feedback(session_id, faq_id, rating):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO feedback (session_id, faq_id, rating, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, faq_id, rating, time.time()),
        )


def get_metrics():
    with get_db() as conn:
        row = conn.execute("""
            SELECT
                COUNT(CASE WHEN action='answer' THEN 1 END) as answered,
                COUNT(CASE WHEN action='clarify' THEN 1 END) as clarified,
                COUNT(CASE WHEN action='handoff' THEN 1 END) as handed_off,
                COUNT(CASE WHEN action='sensitive' THEN 1 END) as sensitive,
                COUNT(*) as total,
                AVG(calibrated_confidence) as avg_confidence,
                AVG(latency_ms) as avg_latency_ms
            FROM query_logs
        """).fetchone()
        return dict(row) if row else {}


def get_domain_distribution():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT domain, COUNT(*) as count
            FROM query_logs
            WHERE domain IS NOT NULL AND domain != 'sensitive'
            GROUP BY domain
            ORDER BY count DESC
        """).fetchall()
        return {r["domain"]: r["count"] for r in rows}


def get_history(session_id, limit=20):
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM query_logs
               WHERE session_id = ? AND query != '[REDACTED]'
               ORDER BY timestamp DESC LIMIT ?""",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_handoff_queries():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT query, COUNT(*) as count
            FROM query_logs
            WHERE action = 'handoff' AND query != '[REDACTED]'
            GROUP BY query
            ORDER BY count DESC
            LIMIT 50
        """).fetchall()
        return [dict(r) for r in rows]


def get_lowest_rated_faqs():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT faq_id, AVG(rating) as avg_rating, COUNT(*) as count
            FROM feedback
            GROUP BY faq_id
            HAVING count >= 1
            ORDER BY avg_rating ASC
            LIMIT 20
        """).fetchall()
        return [dict(r) for r in rows]


def get_all_logs(limit=1000):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def mark_resolved(request_id, resolved=True):
    with get_db() as conn:
        conn.execute(
            "UPDATE query_logs SET resolved = ? WHERE request_id = ?",
            (1 if resolved else 0, request_id),
        )
