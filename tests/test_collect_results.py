import pandas as pd

from pipeline.collect_results import find_new_results


def _matches():
    return pd.DataFrame([
        {"Date": pd.Timestamp("2026-01-10"), "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
         "FTHG": 2, "FTAG": 1, "FTR": "H"},
        {"Date": pd.Timestamp("2026-01-17"), "HomeTeam": "Liverpool", "AwayTeam": "Everton",
         "FTHG": 1, "FTAG": 1, "FTR": "D"},
    ])


def test_finds_result_for_matching_unscored_prediction():
    unscored = [{"match_date": "2026-01-10", "home_team": "Arsenal", "away_team": "Chelsea"}]
    found = find_new_results(_matches(), unscored)
    assert len(found) == 1
    assert found[0] == {
        "match_date": "2026-01-10", "home_team": "Arsenal", "away_team": "Chelsea",
        "fthg": 2, "ftag": 1, "ftr": "H",
    }


def test_ignores_prediction_without_a_confirmed_result_yet():
    unscored = [{"match_date": "2026-02-01", "home_team": "Arsenal", "away_team": "Chelsea"}]
    found = find_new_results(_matches(), unscored)
    assert found == []


def test_does_not_match_wrong_fixture_on_same_date():
    unscored = [{"match_date": "2026-01-10", "home_team": "Arsenal", "away_team": "Everton"}]
    found = find_new_results(_matches(), unscored)
    assert found == []
