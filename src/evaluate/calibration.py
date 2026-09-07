"""예측 확률의 캘리브레이션(보정도)을 계산한다.

CLAUDE.md 모델링 원칙: "평가지표는 Brier score, 로그손실, 캘리브레이션 플롯을
기본으로 사용한다" — Brier/로그손실만으로는 "70% 확신한 예측이 실제로 70% 정도
맞는지"를 직접 보여주지 못해서 별도로 필요하다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_calibration(
    results: pd.DataFrame,
    prob_cols: dict[str, str],
    n_bins: int = 10,
) -> pd.DataFrame:
    """다중 클래스(H/D/A) 예측을 "원-vs-나머지"로 풀어서 캘리브레이션 곡선을 계산한다.

    각 경기·각 클래스(H/D/A)마다 (예측확률, 실제로 그 결과였는지 0/1) 쌍을 만들고
    전부 모아서, 예측확률 구간(n_bins개, 0~1 균등폭)별로 평균 예측확률과 실제
    적중 비율(관측 빈도)을 계산한다. 두 값이 가까울수록(대각선에 가까울수록)
    잘 보정된 것이다.

    prob_cols: 예) {"H": "model_H", "D": "model_D", "A": "model_A"}
    """
    predicted = []
    actual = []
    for outcome, col in prob_cols.items():
        predicted.append(results[col].to_numpy())
        actual.append((results["FTR"] == outcome).astype(float).to_numpy())
    predicted = np.concatenate(predicted)
    actual = np.concatenate(actual)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(predicted, bin_edges) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "bin_center": float((bin_edges[b] + bin_edges[b + 1]) / 2),
            "predicted_mean": float(predicted[mask].mean()),
            "actual_freq": float(actual[mask].mean()),
            "n": n,
        })
    return pd.DataFrame(rows)
