"""배트맨(공식 스포츠토토) '프로토 승부식' 배당 스냅샷을 저장하는 SQLite DB.

logs/predictions.db(우리 모델의 예측 이력)와는 완전히 별개의 파일/스키마다 —
성격이 다른 데이터(모델 예측 vs 외부 배당 사이트)를 섞지 않기 위함이다.
"""
from __future__ import annotations

import pathlib
import sqlite3

DB_PATH = pathlib.Path(__file__).resolve().parents[2] / "logs" / "betman_odds.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS proto_odds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_date TEXT NOT NULL,
    kickoff TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    odds_h REAL NOT NULL,
    odds_d REAL NOT NULL,
    odds_a REAL NOT NULL,
    implied_h REAL NOT NULL,
    implied_d REAL NOT NULL,
    implied_a REAL NOT NULL,
    scraped_at TEXT NOT NULL,
    actual_result TEXT,
    actual_fthg INTEGER,
    actual_ftag INTEGER,
    scored_at TEXT,
    UNIQUE(match_date, home_team, away_team)
);
"""


def connect(db_path: pathlib.Path = DB_PATH) -> sqlite3.Connection:
    db_path = pathlib.Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def insert_odds(conn: sqlite3.Connection, record: dict) -> None:
    """배당 스냅샷 1건을 저장한다. 같은 경기가 이미 있으면 배당만 최신 값으로 덮어쓴다
    (배당은 마감 전까지 변동하므로 재실행 시 갱신하는 게 자연스럽다)."""
    conn.execute(
        """
        INSERT INTO proto_odds
            (match_date, kickoff, home_team, away_team, odds_h, odds_d, odds_a,
             implied_h, implied_d, implied_a, scraped_at)
        VALUES (:match_date, :kickoff, :home_team, :away_team, :odds_h, :odds_d, :odds_a,
                :implied_h, :implied_d, :implied_a, :scraped_at)
        ON CONFLICT(match_date, home_team, away_team) DO UPDATE SET
            kickoff = excluded.kickoff,
            odds_h = excluded.odds_h, odds_d = excluded.odds_d, odds_a = excluded.odds_a,
            implied_h = excluded.implied_h, implied_d = excluded.implied_d, implied_a = excluded.implied_a,
            scraped_at = excluded.scraped_at
        """,
        record,
    )
    conn.commit()


def fetch_unscored(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM proto_odds WHERE actual_result IS NULL").fetchall()


def fetch_scored(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM proto_odds WHERE actual_result IS NOT NULL ORDER BY match_date"
    ).fetchall()


def record_result(
    conn: sqlite3.Connection,
    match_date: str,
    home_team: str,
    away_team: str,
    fthg: int,
    ftag: int,
    ftr: str,
    scored_at: str,
) -> None:
    conn.execute(
        """
        UPDATE proto_odds
        SET actual_result = ?, actual_fthg = ?, actual_ftag = ?, scored_at = ?
        WHERE match_date = ? AND home_team = ? AND away_team = ?
        """,
        (ftr, fthg, ftag, scored_at, match_date, home_team, away_team),
    )
    conn.commit()
