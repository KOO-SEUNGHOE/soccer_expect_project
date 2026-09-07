import pandas as pd
import pytest

from evaluate.backtest import blend_probabilities, summarize_ensemble


def _results():
    return pd.DataFrame([
        {"FTR": "H", "model_H": 0.6, "model_D": 0.2, "model_A": 0.2,
         "market_H": 0.5, "market_D": 0.3, "market_A": 0.2},
        {"FTR": "A", "model_H": 0.3, "model_D": 0.3, "model_A": 0.4,
         "market_H": 0.2, "market_D": 0.3, "market_A": 0.5},
    ])


def test_blend_probabilities_model_weight_one_equals_model():
    blended = blend_probabilities(_results(), model_weight=1.0)
    assert blended[0] == pytest.approx({"H": 0.6, "D": 0.2, "A": 0.2})


def test_blend_probabilities_model_weight_zero_equals_market():
    blended = blend_probabilities(_results(), model_weight=0.0)
    assert blended[0] == pytest.approx({"H": 0.5, "D": 0.3, "A": 0.2})


def test_blend_probabilities_half_weight_is_average():
    blended = blend_probabilities(_results(), model_weight=0.5)
    assert blended[0]["H"] == pytest.approx(0.55)
    assert blended[0]["D"] == pytest.approx(0.25)
    assert blended[0]["A"] == pytest.approx(0.2)


def test_blend_probabilities_always_sums_to_one():
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for row in blend_probabilities(_results(), model_weight=w):
            assert sum(row.values()) == pytest.approx(1.0)


def test_summarize_ensemble_reports_weight_and_n():
    stats = summarize_ensemble(_results(), model_weight=0.5)
    assert stats["model_weight"] == 0.5
    assert stats["n_matches"] == 2
    assert "brier" in stats and "logloss" in stats
