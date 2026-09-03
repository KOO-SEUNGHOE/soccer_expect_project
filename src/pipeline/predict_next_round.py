"""보유한 모든 과거 데이터로 모델을 학습하고, 예정된 다음 라운드 경기에 대한
승/무/패 확률을 계산해 logs/predictions.db에 저장한다.

사용법:
    python src/pipeline/predict_next_round.py

기본 모델은 베이스라인(팀 더미만 사용)이다 — README/CLAUDE.md에 기록된
백테스트 결과, 최근 폼/홈-원정 편차/휴식일수를 추가한 모델이 아직 베이스라인보다
낫다는 근거가 없기 때문에(오히려 소폭 나쁨) 실전 예측에는 더 나은 쪽을 쓴다.

실행 전 `python src/ingest/download.py --seasons <현재 시즌>`로 데이터를 최신
상태로 갱신해 둬야 한다.
"""
from __future__ import annotations

import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from features.build_features import load_all_seasons
from ingest.fixtures import fetch_upcoming_fixtures
from model.poisson_model import PoissonFootballModel
from pipeline.db import connect, insert_prediction

RAW_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
SEASON_FILES = ["E0_2223.csv", "E0_2324.csv", "E0_2425.csv", "E0_2526.csv"]


def build_predictions(matches: pd.DataFrame, fixtures: list[dict]) -> list[dict]:
    """학습된 모델과 예정 경기 목록으로 DB에 넣을 예측 레코드 목록을 만든다 (순수 함수).

    학습 데이터에 없는 팀(승격팀 등)이 낀 경기는 건너뛴다.
    """
    model = PoissonFootballModel().fit(matches)
    predicted_at = dt.datetime.now(dt.timezone.utc).isoformat()

    records = []
    for fx in fixtures:
        home_team, away_team = fx["home_team"], fx["away_team"]
        try:
            probs = model.predict_proba(home_team, away_team)
        except ValueError:
            continue  # 학습 데이터에 없는 팀 (승격팀 등)

        records.append({
            "match_date": fx["date"],
            "home_team": home_team,
            "away_team": away_team,
            "model_h": probs["H"], "model_d": probs["D"], "model_a": probs["A"],
            "market_h": None, "market_d": None, "market_a": None,
            "predicted_at": predicted_at,
        })
    return records


def main() -> None:
    paths = [RAW_DIR / f for f in SEASON_FILES]
    matches = load_all_seasons(paths)

    fixtures = fetch_upcoming_fixtures()
    records = build_predictions(matches, fixtures)

    conn = connect()
    for record in records:
        insert_prediction(conn, record)

    print(f"[predict_next_round] {len(records)}건의 예측을 저장했습니다 (예정 경기 {len(fixtures)}건 중).")


if __name__ == "__main__":
    main()
