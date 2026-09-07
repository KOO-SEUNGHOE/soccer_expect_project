import pandas as pd
import pytest

from evaluate.calibration import compute_calibration


def test_perfectly_calibrated_predictions():
    # 100경기, 항상 H 확률 0.7로 예측하고 실제로 딱 70번 H가 나오게 구성.
    outcomes = ["H"] * 70 + ["D"] * 15 + ["A"] * 15
    results = pd.DataFrame({
        "FTR": outcomes,
        "model_H": [0.7] * 100,
        "model_D": [0.15] * 100,
        "model_A": [0.15] * 100,
    })
    curve = compute_calibration(results, {"H": "model_H", "D": "model_D", "A": "model_A"}, n_bins=10)

    row_h = curve[curve["predicted_mean"].round(2) == 0.7]
    assert len(row_h) == 1
    assert row_h.iloc[0]["actual_freq"] == pytest.approx(0.70, abs=0.01)
    assert row_h.iloc[0]["n"] == 100  # H를 예측한 100개 행 전부 이 구간에 들어감


def test_bins_without_any_prediction_are_omitted():
    results = pd.DataFrame({
        "FTR": ["H", "H"],
        "model_H": [0.9, 0.9],
        "model_D": [0.05, 0.05],
        "model_A": [0.05, 0.05],
    })
    curve = compute_calibration(results, {"H": "model_H", "D": "model_D", "A": "model_A"}, n_bins=10)
    # 예측확률이 0.05(D,A)와 0.9(H) 근처에만 몰려 있으니 두 구간만 나와야 한다.
    assert len(curve) == 2
    assert curve["n"].sum() == 2 * 3  # 2경기 x 3클래스(H/D/A)


def test_calibration_total_count_matches_n_matches_times_three_classes():
    results = pd.DataFrame({
        "FTR": ["H", "D", "A"],
        "model_H": [0.5, 0.3, 0.2],
        "model_D": [0.3, 0.4, 0.3],
        "model_A": [0.2, 0.3, 0.5],
    })
    curve = compute_calibration(results, {"H": "model_H", "D": "model_D", "A": "model_A"}, n_bins=5)
    assert curve["n"].sum() == 3 * 3  # 3경기 x 3클래스(H/D/A)
