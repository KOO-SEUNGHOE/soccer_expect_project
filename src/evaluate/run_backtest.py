"""저장된 전체 시즌 데이터로 백테스트를 실행하고, 베이스라인(피처 없음) 모델과
피처(최근 폼/홈-원정 편차/휴식일수) 추가 모델을 배당(시장)과 함께 비교한다.

사용법:
    python src/evaluate/run_backtest.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from evaluate.backtest import run_walkforward_backtest, summarize
from features.build_features import load_all_seasons

RAW_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
SEASON_FILES = ["E0_2223.csv", "E0_2324.csv", "E0_2425.csv", "E0_2526.csv"]

FEATURE_COLS = ["form", "venue_form", "rest_days"]


def main() -> None:
    paths = [RAW_DIR / f for f in SEASON_FILES]
    matches = load_all_seasons(paths)
    print(f"전체 경기 수: {len(matches)} ({matches['Date'].min().date()} ~ {matches['Date'].max().date()})")

    baseline = run_walkforward_backtest(matches, feature_cols=None)
    enhanced = run_walkforward_backtest(matches, feature_cols=FEATURE_COLS)

    print("\n[베이스라인: is_home + team + opponent만 사용]")
    print(summarize(baseline))

    print("\n[피처 추가: + 최근 폼/홈-원정 편차/휴식일수]")
    print(summarize(enhanced))


if __name__ == "__main__":
    main()
