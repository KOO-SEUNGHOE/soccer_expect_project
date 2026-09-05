"""저장된 전체 시즌 데이터로 백테스트를 실행하고, 베이스라인(피처 없음) 모델과
여러 피처 조합 모델을 배당(시장)과 함께 비교한다.

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

FEATURE_SETS = {
    "베이스라인 (피처 없음)": None,
    "+ 최근 폼/홈-원정 편차/휴식일수": ["form", "venue_form", "rest_days"],
    "+ 상대전적(H2H)만": ["h2h"],
    "+ 홈-원정 편차 + 상대전적(H2H)": ["venue_form", "h2h"],
}


def main() -> None:
    paths = [RAW_DIR / f for f in SEASON_FILES]
    matches = load_all_seasons(paths)
    print(f"전체 경기 수: {len(matches)} ({matches['Date'].min().date()} ~ {matches['Date'].max().date()})")

    for label, feature_cols in FEATURE_SETS.items():
        result = run_walkforward_backtest(matches, feature_cols=feature_cols)
        print(f"\n[{label}]")
        print(summarize(result))


if __name__ == "__main__":
    main()
