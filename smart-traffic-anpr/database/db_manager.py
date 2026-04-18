"""
db_manager.py — All database read/write operations.
Supports SQLite (default) and PostgreSQL.
Thread-safe for concurrent Streamlit + pipeline access.
"""

import sqlite3
import logging
import threading
from datetime import datetime
from pathlib import Path
from config import DATABASE_PATH, DB_ENGINE, POSTGRES_URL

logger = logging.getLogger(__name__)
_lock = threading.Lock()


class DatabaseManager:
    """Handles all database interactions for the ANPR system."""

    def __init__(self):
        self.db_path = str(DATABASE_PATH)
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self):
        if DB_ENGINE == "sqlite":
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn
        else:
            import psycopg2
            return psycopg2.connect(POSTGRES_URL)

    def _init_db(self):
        """Create tables if they do not exist."""
        with _lock:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS vehicle_log (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        vehicle_type    TEXT    NOT NULL,
                        plate_number    TEXT,
                        ocr_confidence  REAL,
                        video_source    TEXT,
                        timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS session_log (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_start   DATETIME,
                        session_end     DATETIME,
                        video_source    TEXT,
                        total_vehicles  INTEGER DEFAULT 0,
                        notes           TEXT
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp
                    ON vehicle_log(timestamp)
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_vehicle_type
                    ON vehicle_log(vehicle_type)
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_plate
                    ON vehicle_log(plate_number)
                """)

                # Check if we need sample data (empty database)
                cursor = conn.execute("SELECT COUNT(*) FROM vehicle_log")
                count = cursor.fetchone()[0]
                if count == 0:
                    logger.info("Inserting sample data for demo purposes...")
                    sample_data = [
                        ('truck', 'ABC123', 0.92, 'sample_traffic_1.mp4'),
                        ('car', 'XYZ456', 0.87, 'sample_traffic_1.mp4'),
                        ('motorcycle', None, None, 'sample_traffic_1.mp4'),
                        ('bus', 'DEF789', 0.91, 'sample_traffic_1.mp4'),
                    ]
                    conn.executemany("""
                        INSERT INTO vehicle_log (vehicle_type, plate_number, ocr_confidence, video_source)
                        VALUES (?, ?, ?, ?)
                    """, sample_data)

                conn.commit()
        logger.info("Database initialized.")

    def insert_vehicle(
        self,
        vehicle_type: str,
        plate_number: str | None,
        ocr_confidence: float | None,
        video_source: str = ""
    ) -> int:
        """
        Insert a detected vehicle record.
        Returns the new row ID.
        """
        with _lock:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO vehicle_log
                        (vehicle_type, plate_number, ocr_confidence, video_source, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    vehicle_type,
                    plate_number,
                    ocr_confidence,
                    video_source,
                    datetime.now().isoformat()
                ))
                conn.commit()
                row_id = cursor.lastrowid
                logger.debug(
                    f"DB insert: {vehicle_type} | {plate_number} | row_id={row_id}"
                )
                return row_id

    def get_recent_logs(self, limit: int = 200) -> list[dict]:
        """Return the most recent N vehicle log records."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT id, vehicle_type, plate_number, ocr_confidence,
                       video_source, timestamp
                FROM vehicle_log
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_counts_by_class(self) -> dict:
        """Return total count per vehicle class."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT vehicle_type, COUNT(*) as count
                FROM vehicle_log
                GROUP BY vehicle_type
            """).fetchall()
            return {r["vehicle_type"]: r["count"] for r in rows}

    def get_hourly_trend(self, hours: int = 24) -> list[dict]:
        """Return per-hour vehicle counts for the last N hours."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT
                    strftime('%Y-%m-%d %H:00', timestamp) AS hour,
                    COUNT(*) AS count
                FROM vehicle_log
                WHERE timestamp >= datetime('now', ? || ' hours')
                GROUP BY hour
                ORDER BY hour ASC
            """, (f"-{hours}",)).fetchall()
            return [dict(r) for r in rows]

    def search_by_plate(self, query: str) -> list[dict]:
        """Search vehicle log by partial plate number."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM vehicle_log
                WHERE plate_number LIKE ?
                ORDER BY timestamp DESC
                LIMIT 100
            """, (f"%{query.upper()}%",)).fetchall()
            return [dict(r) for r in rows]

    def get_summary_stats(self) -> dict:
        """Return overall statistics for dashboard summary cards."""
        with self._get_connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM vehicle_log"
            ).fetchone()[0]

            plates_read = conn.execute(
                "SELECT COUNT(*) FROM vehicle_log WHERE plate_number IS NOT NULL"
            ).fetchone()[0]

            avg_conf = conn.execute(
                "SELECT AVG(ocr_confidence) FROM vehicle_log WHERE ocr_confidence IS NOT NULL"
            ).fetchone()[0]

            return {
                "total_vehicles":   total,
                "plates_read":      plates_read,
                "ocr_success_rate": round((plates_read / total * 100) if total else 0, 1),
                "avg_ocr_conf":     round(avg_conf or 0, 3)
            }
