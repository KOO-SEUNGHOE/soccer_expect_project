import pandas as pd
import pytest

from features.build_features import implied_probabilities


def test_implied_probabilities_sum_to_one():
    row = pd.Series({"AvgH": 2.0, "AvgD": 3.5, "AvgA": 4.0})
    p_h, p_d, p_a = implied_probabilities(row)
    assert p_h + p_d + p_a == pytest.approx(1.0)


def test_implied_probabilities_removes_margin():
    # 마진 제거 전 1/odds 합(overround)은 항상 1보다 커야 북메이커가 이윤을 남긴다.
    row = pd.Series({"AvgH": 2.0, "AvgD": 3.5, "AvgA": 4.0})
    overround = 1 / row["AvgH"] + 1 / row["AvgD"] + 1 / row["AvgA"]
    assert overround > 1.0

    p_h, _p_d, _p_a = implied_probabilities(row)
    # 마진 제거 후에는 정규화되어 정확히 1/overround 배만큼 줄어든 값이어야 한다.
    assert p_h == pytest.approx((1 / row["AvgH"]) / overround)


def test_implied_probabilities_favorite_has_highest_probability():
    row = pd.Series({"AvgH": 1.5, "AvgD": 4.0, "AvgA": 6.0})
    p_h, p_d, p_a = implied_probabilities(row)
    assert p_h > p_d > p_a
