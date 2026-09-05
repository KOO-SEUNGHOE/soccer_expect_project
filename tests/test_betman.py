import pandas as pd
import pytest

from betman import db
from betman.proto_odds import _parse_odds_text, implied_probabilities_from_odds
from betman.score_odds import find_new_results


def test_implied_probabilities_from_odds_sum_to_one():
    p_h, p_d, p_a = implied_probabilities_from_odds(1.90, 3.30, 3.40)
    assert p_h + p_d + p_a == pytest.approx(1.0)


def test_implied_probabilities_from_odds_favors_lower_odds():
    p_h, p_d, p_a = implied_probabilities_from_odds(1.11, 6.90, 13.00)
    assert p_h > p_d > p_a


def test_parse_odds_text_strips_movement_indicator():
    assert _parse_odds_text("1.65배당률 상승") == 1.65
    assert _parse_odds_text("3.55") == 3.55


def test_parse_odds_text_raises_on_garbage():
    with pytest.raises(ValueError):
        _parse_odds_text("상승")


def _sample_record(**overrides):
    record = {
        "match_date": "2026-09-05", "kickoff": "2026-09-05T23:00:00",
        "home_team": "Brighton", "away_team": "Leeds",
        "odds_h": 1.90, "odds_d": 3.30, "odds_a": 3.40,
        "implied_h": 0.468, "implied_d": 0.270, "implied_a": 0.262,
        "scraped_at": "2026-09-05T00:00:00+00:00",
    }
    record.update(overrides)
    return record


def test_db_insert_and_fetch_unscored(tmp_path):
    conn = db.connect(tmp_path / "betman_odds.db")
    db.insert_odds(conn, _sample_record())

    rows = db.fetch_unscored(conn)
    assert len(rows) == 1
    assert rows[0]["home_team"] == "Brighton"
    assert rows[0]["actual_result"] is None


def test_db_insert_is_idempotent_and_refreshes_odds(tmp_path):
    conn = db.connect(tmp_path / "betman_odds.db")
    db.insert_odds(conn, _sample_record(odds_h=1.90))
    db.insert_odds(conn, _sample_record(odds_h=1.75))  # 마감 전 배당 변동 재수집 시나리오

    rows = db.fetch_unscored(conn)
    assert len(rows) == 1
    assert rows[0]["odds_h"] == 1.75


def test_db_record_result_moves_to_scored(tmp_path):
    conn = db.connect(tmp_path / "betman_odds.db")
    db.insert_odds(conn, _sample_record())

    db.record_result(
        conn,
        match_date="2026-09-05", home_team="Brighton", away_team="Leeds",
        fthg=1, ftag=0, ftr="H", scored_at="2026-09-06T00:00:00+00:00",
    )

    assert db.fetch_unscored(conn) == []
    scored = db.fetch_scored(conn)
    assert len(scored) == 1
    assert scored[0]["actual_result"] == "H"


def test_find_new_results_matches_by_date_and_teams():
    matches = pd.DataFrame([
        {"Date": pd.Timestamp("2026-09-05"), "HomeTeam": "Brighton", "AwayTeam": "Leeds",
         "FTHG": 1, "FTAG": 0, "FTR": "H"},
    ])
    unscored = [{"match_date": "2026-09-05", "home_team": "Brighton", "away_team": "Leeds"}]
    found = find_new_results(matches, unscored)
    assert found == [{
        "match_date": "2026-09-05", "home_team": "Brighton", "away_team": "Leeds",
        "fthg": 1, "ftag": 0, "ftr": "H",
    }]


def test_find_new_results_ignores_unplayed_match():
    matches = pd.DataFrame([
        {"Date": pd.Timestamp("2026-09-05"), "HomeTeam": "Brighton", "AwayTeam": "Leeds",
         "FTHG": 1, "FTAG": 0, "FTR": "H"},
    ])
    unscored = [{"match_date": "2026-09-12", "home_team": "Arsenal", "away_team": "Chelsea"}]
    assert find_new_results(matches, unscored) == []
