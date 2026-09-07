"""예측 이력을 저장하는 SQLite DB(logs/predictions.db)의 스키마와 CRUD.

이 모듈은 연결/스키마 관리와 단순 쿼리만 담당한다 (I/O 레이어). 예측값 계산이나
채점 로직은 여기 두지 않고 호출하는 쪽(predict_next_round.py, collect_results.py,
generate_badge.py)에 맡긴다.
"""
from __future__ import annotations

import pathlib
import sqlite3

DB_PATH = pathlib.Path(__file__).resolve().parents[2] / "logs" / "predictions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    model_h REAL NOT NULL,
    model_d REAL NOT NULL,
    model_a REAL NOT NULL,
    market_h REAL,
    market_d REAL,
    market_a REAL,
    predicted_at TEXT NOT NULL,
    actual_result TEXT,
    actual_fthg INTEGER,
    actual_ftag INTEGER,
    scored_at TEXT,
    UNIQUE(match_date, home_team, away_team)
);
"""

# 이미 배포된 DB 파일(logs/predictions.db)에 새 컬럼을 안전하게 추가하기 위한
# 마이그레이션 목록 — CREATE TABLE IF NOT EXISTS는 기존 테이블에 컬럼을
# 추가해주지 않으므로, connect()에서 없는 컬럼만 골라 ALTER TABLE로 붙인다.
_MIGRATION_COLUMNS = {
    "most_likely_score": "TEXT",
    "btts_yes_prob": "REAL",
    "over_2_5_prob": "REAL",
}


def connect(db_path: pathlib.Path = DB_PATH) -> sqlite3.Connection:
    """DB에 연결하고(없으면 파일 생성) 스키마를 보장한 뒤 커넥션을 반환한다."""
    db_path = pathlib.Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    conn.commit()

    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
    for col, col_type in _MIGRATION_COLUMNS.items():
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")
    conn.commit()
    return conn


def insert_prediction(conn: sqlite3.Connection, record: dict) -> None:
    """예측 1건을 저장한다. 같은 (날짜, 홈팀, 원정팀) 조합이 이미 있으면
    모델/시장 확률만 최신 값으로 덮어쓴다 (실제 결과가 이미 채워졌다면 유지).

    most_likely_score/btts_yes_prob/over_2_5_prob는 선택 항목이라 record에
    없어도(None) 된다 — predict_markets()를 안 쓰는 호출부와의 하위 호환.
    """
    full_record = {
        "match_date": record["match_date"],
        "home_team": record["home_team"],
        "away_team": record["away_team"],
        "model_h": record["model_h"],
        "model_d": record["model_d"],
        "model_a": record["model_a"],
        "market_h": record.get("market_h"),
        "market_d": record.get("market_d"),
        "market_a": record.get("market_a"),
        "most_likely_score": record.get("most_likely_score"),
        "btts_yes_prob": record.get("btts_yes_prob"),
        "over_2_5_prob": record.get("over_2_5_prob"),
        "predicted_at": record["predicted_at"],
    }
    conn.execute(
        """
        INSERT INTO predictions
            (match_date, home_team, away_team, model_h, model_d, model_a,
             market_h, market_d, market_a, most_likely_score, btts_yes_prob,
             over_2_5_prob, predicted_at)
        VALUES (:match_date, :home_team, :away_team, :model_h, :model_d, :model_a,
                :market_h, :market_d, :market_a, :most_likely_score, :btts_yes_prob,
                :over_2_5_prob, :predicted_at)
        ON CONFLICT(match_date, home_team, away_team) DO UPDATE SET
            model_h = excluded.model_h,
            model_d = excluded.model_d,
            model_a = excluded.model_a,
            market_h = excluded.market_h,
            market_d = excluded.market_d,
            market_a = excluded.market_a,
            most_likely_score = excluded.most_likely_score,
            btts_yes_prob = excluded.btts_yes_prob,
            over_2_5_prob = excluded.over_2_5_prob,
            predicted_at = excluded.predicted_at
        """,
        full_record,
    )
    conn.commit()


def fetch_unscored(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """아직 실제 결과가 채워지지 않은 예측 목록을 반환한다."""
    return conn.execute(
        "SELECT * FROM predictions WHERE actual_result IS NULL"
    ).fetchall()


def fetch_scored(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """실제 결과가 채워진(채점 완료된) 예측 목록을 날짜순으로 반환한다."""
    return conn.execute(
        "SELECT * FROM predictions WHERE actual_result IS NOT NULL ORDER BY match_date"
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
    """예정돼 있던 예측에 실제 경기 결과를 채운다."""
    conn.execute(
        """
        UPDATE predictions
        SET actual_result = ?, actual_fthg = ?, actual_ftag = ?, scored_at = ?
        WHERE match_date = ? AND home_team = ? AND away_team = ?
        """,
        (ftr, fthg, ftag, scored_at, match_date, home_team, away_team),
    )
    conn.commit()
