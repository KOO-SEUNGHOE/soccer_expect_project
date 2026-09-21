import pandas as pd
import pytest

from analytics.descriptive_stats import (
    home_advantage_stats,
    lookup_market_hit_rate,
    market_odds_calibration,
    odds_movement_accuracy,
    referee_card_stats,
    season_label_from_path,
    team_attack_defense_profile,
)


def _toy_matches() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
        "HomeTeam": ["A", "B", "A"],
        "AwayTeam": ["B", "A", "B"],
        "FTHG": [2, 1, 0],
        "FTAG": [0, 1, 0],
        "FTR": ["H", "D", "D"],
        "HTHG": [1, 0, 0], "HTAG": [0, 0, 0],
        "Referee": ["Ref1", "Ref1", "Ref2"],
        "HS": [10, 8, 6], "AS": [5, 9, 7],
        "HST": [6, 4, 3], "AST": [2, 5, 3],
        "HF": [10, 11, 9], "AF": [12, 10, 8],
        "HC": [5, 4, 3], "AC": [3, 6, 4],
        "HY": [1, 2, 1], "AY": [2, 1, 0],
        "HR": [0, 0, 0], "AR": [0, 0, 1],
        "B365H": [1.5, 2.5, 2.0], "B365D": [4.0, 3.2, 3.3], "B365A": [6.0, 2.8, 3.5],
        "B365CH": [1.4, 2.6, 2.0], "B365CD": [4.2, 3.1, 3.3], "B365CA": [6.5, 2.7, 3.5],
    })


def test_team_attack_defense_profile_aggregates_home_and_away():
    profile = team_attack_defense_profile(_toy_matches())
    team_a = profile[profile["팀"] == "A"].iloc[0]
    # A는 홈 2번(2, 0골), 원정 1번(1골) = 평균 (2+0+1)/3
    assert team_a["경기수"] == 3
    assert team_a["평균득점"] == pytest.approx((2 + 0 + 1) / 3)


def test_referee_card_stats_excludes_low_sample_referees():
    stats = referee_card_stats(_toy_matches(), min_matches=2)
    referees = set(stats["심판"])
    assert "Ref1" in referees  # 2경기
    assert "Ref2" not in referees  # 1경기 < min_matches


def test_home_advantage_stats_percentages_sum_to_100():
    stats = home_advantage_stats(_toy_matches())
    assert stats["home_win_pct"] + stats["draw_pct"] + stats["away_win_pct"] == pytest.approx(100.0)
    assert stats["n_matches"] == 3


def test_odds_movement_accuracy_reports_hit_rates_in_valid_range():
    stats = odds_movement_accuracy(_toy_matches())
    assert stats["n_matches"] == 3
    assert 0.0 <= stats["시가_적중률"] <= 100.0
    assert 0.0 <= stats["종가_적중률"] <= 100.0


def test_season_label_from_path_parses_filename():
    assert season_label_from_path("data/raw/E0_2627.csv") == "2026-27"
    assert season_label_from_path("E0_2223.csv") == "2022-23"


def test_season_label_from_path_rejects_unrecognized_filename():
    with pytest.raises(ValueError):
        season_label_from_path("data/raw/unexpected_name.csv")


def _toy_market_matches() -> pd.DataFrame:
    # 배당이 전부 동일(3.0/3.0/3.0, implied 확률 각 1/3)한 10경기.
    # 실제로는 무승부가 4번(40%), 홈/원정승은 각 3번(30%)만 나오게 구성해
    # "같은 배당이라도 마켓마다 실제 적중률이 다를 수 있다"는 걸 검증한다.
    ftr = ["D", "D", "D", "D", "H", "H", "H", "A", "A", "A"]
    return pd.DataFrame({"FTR": ftr, "AvgH": [3.0] * 10, "AvgD": [3.0] * 10, "AvgA": [3.0] * 10})


def test_market_odds_calibration_separates_hit_rate_by_market():
    curve = market_odds_calibration(_toy_market_matches(), n_bins=10)

    draw_row = curve[curve["시장"] == "무승부"].iloc[0]
    assert draw_row["predicted_mean"] == pytest.approx(1 / 3, abs=1e-9)
    assert draw_row["actual_freq"] == pytest.approx(0.4, abs=1e-9)
    assert draw_row["n"] == 10

    home_row = curve[curve["시장"] == "홈승"].iloc[0]
    # 홈승도 implied 확률은 무승부와 똑같이 1/3이지만 실제 적중률은 다르다(0.3).
    assert home_row["predicted_mean"] == pytest.approx(1 / 3, abs=1e-9)
    assert home_row["actual_freq"] == pytest.approx(0.3, abs=1e-9)


def test_lookup_market_hit_rate_matches_calibration_row():
    curve = market_odds_calibration(_toy_market_matches(), n_bins=10)
    result = lookup_market_hit_rate(curve, "D", predicted_prob=1 / 3, n_bins=10)
    assert result is not None
    hit_rate, n = result
    assert hit_rate == pytest.approx(0.4, abs=1e-9)
    assert n == 10


def test_lookup_market_hit_rate_returns_none_for_empty_bin():
    curve = market_odds_calibration(_toy_market_matches(), n_bins=10)
    # 실제 표본이 전혀 없는 확률대(0.9~1.0)를 조회하면 None이어야 한다.
    assert lookup_market_hit_rate(curve, "A", predicted_prob=0.95, n_bins=10) is None
