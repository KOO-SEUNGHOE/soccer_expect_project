"""시간순으로 앞부분 데이터로 학습 -> 뒷부분 예측 -> 실제 결과와 비교하는 백테스트.

CLAUDE.md 원칙: 랜덤 split이 아니라 반드시 시간순으로 분리한다 (미래 데이터로
과거를 맞추는 문제 방지). 평가지표는 accuracy 하나만 보지 않고 Brier score /
로그손실을 함께 본다.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from features.build_features import clean_for_market_comparison, implied_probabilities
from features.rolling_features import add_rolling_features
from model.poisson_model import PoissonFootballModel

OUTCOME_TO_ONEHOT = {"H": (1, 0, 0), "D": (0, 1, 0), "A": (0, 0, 1)}


def brier_score(probs: list[dict], outcomes: list[str]) -> float:
    """다중 클래스(H/D/A) Brier score. 낮을수록 좋음 (0 = 완벽, 랜덤 추측은 대략 0.66)."""
    total = 0.0
    for p, y in zip(probs, outcomes):
        y_vec = OUTCOME_TO_ONEHOT[y]
        total += sum((p[k] - y_vec[i]) ** 2 for i, k in enumerate(["H", "D", "A"]))
    return total / len(outcomes)


def log_loss(probs: list[dict], outcomes: list[str], eps: float = 1e-15) -> float:
    """다중 클래스 로그손실. 낮을수록 좋음."""
    total = 0.0
    for p, y in zip(probs, outcomes):
        total += -np.log(max(p[y], eps))
    return total / len(outcomes)


def run_walkforward_backtest(
    matches: pd.DataFrame,
    min_train_matches: int = 380,
    retrain_every: int = 10,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """min_train_matches 만큼 학습한 뒤, retrain_every 경기(EPL 한 라운드=10경기)
    묶음마다 그 시점까지의 전체 데이터로 재학습하며 다음 묶음을 예측한다.

    매 경기마다 재학습하는 대신 라운드 단위로 재학습하는 것은, 실전 주간
    파이프라인("매주 금요일 다음 라운드 예측")의 재학습 주기를 그대로 모사하기
    위함이다 — 시즌이 여러 개 이어질 때 한 번만 학습한 고정 모델을 계속 쓰면
    스쿼드가 바뀐 뒤 시즌에 대해 학습 시점이 과도하게 뒤처지는 문제가 생긴다.

    feature_cols가 주어지면 add_rolling_features로 계산한 최근 폼/홈-원정 편차/
    휴식일수 등을 모델 공변량으로 함께 사용한다.
    """
    matches = matches.sort_values("Date").reset_index(drop=True)
    matches = add_rolling_features(matches)
    feature_cols = feature_cols or []

    records = []
    n = len(matches)
    cursor = min_train_matches
    while cursor < n:
        train = matches.iloc[:cursor]
        test_chunk = clean_for_market_comparison(matches.iloc[cursor:cursor + retrain_every])
        model = PoissonFootballModel(feature_cols=feature_cols).fit(train)

        for _, row in test_chunk.iterrows():
            home_features = {c: row[f"home_{c}"] for c in feature_cols}
            away_features = {c: row[f"away_{c}"] for c in feature_cols}
            if any(pd.isna(v) for v in list(home_features.values()) + list(away_features.values())):
                continue  # 표본 부족(시즌 초반/승격팀 등)으로 피처가 없는 경기는 평가에서 제외

            try:
                model_probs = model.predict_proba(row["HomeTeam"], row["AwayTeam"], home_features, away_features)
            except Exception:
                continue  # 학습 데이터에 없던 팀(승격팀 등)은 이 베이스라인에서 건너뜀

            market_probs = implied_probabilities(row)
            records.append({
                "Date": row["Date"],
                "HomeTeam": row["HomeTeam"],
                "AwayTeam": row["AwayTeam"],
                "FTR": row["FTR"],
                "model_H": model_probs["H"], "model_D": model_probs["D"], "model_A": model_probs["A"],
                "market_H": market_probs[0], "market_D": market_probs[1], "market_A": market_probs[2],
            })

        cursor += retrain_every

    return pd.DataFrame(records)


def summarize(results: pd.DataFrame) -> dict:
    model_probs = [{"H": r.model_H, "D": r.model_D, "A": r.model_A} for r in results.itertuples()]
    market_probs = [{"H": r.market_H, "D": r.market_D, "A": r.market_A} for r in results.itertuples()]
    outcomes = results["FTR"].tolist()

    return {
        "n_matches": len(results),
        "model_brier": brier_score(model_probs, outcomes),
        "market_brier": brier_score(market_probs, outcomes),
        "model_logloss": log_loss(model_probs, outcomes),
        "market_logloss": log_loss(market_probs, outcomes),
    }
