"""저장된 배트맨 프로토 배당 중 실제 결과가 아직 없는 경기를 찾아, 최신 시즌
CSV에서 확정된 결과를 매칭해 기록한다.

pipeline/collect_results.py와 매칭 방식은 같지만(현재 시즌 CSV 재다운로드),
배트맨 배당 DB(logs/betman_odds.db)에 독립적으로 기록한다 — 우리 모델의
예측 채점(logs/predictions.db)과 섞지 않는다.

사용법:
    python src/ingest/download.py --seasons <현재 시즌>   # 먼저 최신화
    python src/betman/score_odds.py
"""
from __future__ import annotations

import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from betman.db import connect, fetch_unscored, record_result
from features.build_features import load_all_seasons

RAW_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
SEASON_FILES = ["E0_2223.csv", "E0_2324.csv", "E0_2425.csv", "E0_2526.csv", "E0_2627.csv"]


def _match_key(date, home_team: str, away_team: str) -> tuple[str, str, str]:
    date_str = date.date().isoformat() if hasattr(date, "date") else str(date)
    return (date_str, home_team, away_team)


def find_new_results(matches: pd.DataFrame, unscored: list) -> list[dict]:
    """확정된 결과 중, 아직 채점되지 않은 배당 스냅샷과 매칭되는 것만 골라낸다 (순수 함수)."""
    results_by_key = {
        _match_key(row.Date, row.HomeTeam, row.AwayTeam): (int(row.FTHG), int(row.FTAG), row.FTR)
        for row in matches.itertuples()
    }

    found = []
    for odds_row in unscored:
        key = (odds_row["match_date"], odds_row["home_team"], odds_row["away_team"])
        if key in results_by_key:
            fthg, ftag, ftr = results_by_key[key]
            found.append({
                "match_date": odds_row["match_date"],
                "home_team": odds_row["home_team"],
                "away_team": odds_row["away_team"],
                "fthg": fthg, "ftag": ftag, "ftr": ftr,
            })
    return found


def score_odds(conn, matches: pd.DataFrame) -> int:
    unscored = fetch_unscored(conn)
    new_results = find_new_results(matches, unscored)
    scored_at = dt.datetime.now(dt.timezone.utc).isoformat()

    for r in new_results:
        record_result(
            conn,
            match_date=r["match_date"], home_team=r["home_team"], away_team=r["away_team"],
            fthg=r["fthg"], ftag=r["ftag"], ftr=r["ftr"],
            scored_at=scored_at,
        )
    return len(new_results)


def main() -> None:
    paths = [RAW_DIR / f for f in SEASON_FILES]
    matches = load_all_seasons(paths)

    conn = connect()
    n = score_odds(conn, matches)
    print(f"[betman] {n}건의 프로토 배당에 실제 결과를 기록했습니다.")


if __name__ == "__main__":
    main()
