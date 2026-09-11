import pandas as pd
import pytest

from analytics.descriptive_stats import (
    home_advantage_stats,
    odds_movement_accuracy,
    referee_card_stats,
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
