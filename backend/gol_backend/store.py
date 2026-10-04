"""SQLite storage for runs and their per-report metrics."""
from __future__ import annotations

import sqlite3
import threading
import time

FIELDS = ("tick", "generation", "population", "avg_energy", "avg_fitness", "max_fitness")


class Store:
    def __init__(self, path: str = ":memory:"):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, learning TEXT, started REAL);
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    tick INTEGER, generation INTEGER, population INTEGER,
                    avg_energy REAL, avg_fitness REAL, max_fitness REAL, ts REAL);
                CREATE INDEX IF NOT EXISTS metrics_run ON metrics (run_id, id);
                """
            )

    def add(self, run_id: str, learning: str, m: dict) -> int:
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO runs (run_id, learning, started) VALUES (?, ?, ?)",
                (run_id, learning, now))
            cur = self._db.execute(
                f"INSERT INTO metrics (run_id, {', '.join(FIELDS)}, ts) VALUES (?, {', '.join('?' * len(FIELDS))}, ?)",
                (run_id, *(m[f] for f in FIELDS), now))
            return cur.lastrowid

    def runs(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                """SELECT r.run_id, r.learning, r.started, COUNT(m.id) AS reports,
                          COALESCE(MAX(m.tick), 0) AS tick, COALESCE(MAX(m.ts), r.started) AS last_seen
                   FROM runs r LEFT JOIN metrics m ON m.run_id = r.run_id
                   GROUP BY r.run_id ORDER BY r.started DESC""").fetchall()
        return [dict(r) for r in rows]

    def metrics(self, run_id: str, after_id: int = 0, limit: int = 5000) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                f"SELECT id, {', '.join(FIELDS)} FROM metrics WHERE run_id = ? AND id > ? ORDER BY id LIMIT ?",
                (run_id, after_id, limit)).fetchall()
        return [dict(r) for r in rows]
