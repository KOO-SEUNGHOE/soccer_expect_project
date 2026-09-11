"""저장된 예측 중 실제 결과가 아직 채워지지 않은 경기를 찾아, 최신 시즌 CSV에서
확정된 결과를 매칭해 logs/predictions.db에 기록한다.

사용법:
    python src/pipeline/collect_results.py

실행 전 `python src/ingest/download.py --seasons <현재 시즌>`로 현재 시즌 CSV를
최신 상태로 갱신해 둬야 지난 라운드 결과가 반영된다 (football-data.co.uk는
과거 결과만 제공하므로 예정 경기 일정과 달리 별도 API 없이 이 CSV 재다운로드로
"지난 라운드 실제 결과 수집"을 해결할 수 있다).
"""
from __future__ import annotations

import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from features.build_features import load_all_seasons
from pipeline.db import connect, fetch_unscored, record_result

RAW_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
SEASON_FILES = ["E0_2223.csv", "E0_2324.csv", "E0_2425.csv", "E0_2526.csv", "E0_2627.csv"]


def _match_key(date, home_team: str, away_team: str) -> tuple[str, str, str]:
    date_str = date.date().isoformat() if hasattr(date, "date") else str(date)
    return (date_str, home_team, away_team)


def find_new_results(matches: pd.DataFrame, unscored: list) -> list[dict]:
    """확정된 결과 중, 아직 채점되지 않은 예측과 매칭되는 것만 골라낸다 (순수 함수)."""
    results_by_key = {
        _match_key(row.Date, row.HomeTeam, row.AwayTeam): (int(row.FTHG), int(row.FTAG), row.FTR)
        for row in matches.itertuples()
    }

    found = []
    for pred in unscored:
        key = (pred["match_date"], pred["home_team"], pred["away_team"])
        if key in results_by_key:
            fthg, ftag, ftr = results_by_key[key]
            found.append({
                "match_date": pred["match_date"],
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "fthg": fthg,
                "ftag": ftag,
                "ftr": ftr,
            })
    return found


def collect_results(conn, matches: pd.DataFrame) -> int:
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
    n = collect_results(conn, matches)
    print(f"[collect_results] {n}건의 예측에 실제 결과를 기록했습니다.")


if __name__ == "__main__":
    main()
