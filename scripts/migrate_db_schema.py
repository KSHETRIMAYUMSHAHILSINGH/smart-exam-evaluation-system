"""Small SQLite migration for the upgraded trained_score field."""
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "exam_eval.db"


def add_column_if_missing(conn, table, column, ddl):
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        print(f"Added {table}.{column}")
    else:
        print(f"Already present: {table}.{column}")


def main():
    conn = sqlite3.connect(DB)
    add_column_if_missing(conn, "evaluations", "trained_score", "FLOAT")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
