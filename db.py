import sqlite3
from typing import Dict, Any, List

DB_PATH = "eval_runs.sqlite"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            who TEXT,
            task TEXT NOT NULL,
            provider TEXT NOT NULL,
            model_version TEXT NOT NULL,
            prompt_id TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            prompt_text TEXT NOT NULL,
            test_data_id TEXT NOT NULL,
            test_data_hash TEXT NOT NULL,
            test_data_filename TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT,
            output_text TEXT,
            latency_ms INTEGER,
            tokens_in INTEGER,
            tokens_out INTEGER,
            cost_est_usd REAL,
            score_overall REAL,
            score_json TEXT,
            schema_valid INTEGER,
            required_sections_present INTEGER,
            numeric_flags_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def insert_run(row: Dict[str, Any]) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    keys = list(row.keys())
    placeholders = ",".join(["?"] * len(keys))
    cur.execute(
        f"INSERT INTO eval_runs ({','.join(keys)}) VALUES ({placeholders})",
        [row[k] for k in keys],
    )
    conn.commit()
    conn.close()


def update_run(run_id: str, updates: Dict[str, Any]) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    sets = ",".join([f"{k}=?" for k in updates.keys()])
    cur.execute(
        f"UPDATE eval_runs SET {sets} WHERE run_id=?",
        [*updates.values(), run_id],
    )
    conn.commit()
    conn.close()


def fetch_runs(limit: int = 50) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM eval_runs ORDER BY datetime(created_at) DESC LIMIT ?",
        (limit,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

