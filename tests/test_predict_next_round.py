import pandas as pd
import pytest

from pipeline.predict_next_round import build_predictions


def _toy_matches():
    rows = [
        ("A", "B", 2, 0), ("B", "C", 1, 1), ("C", "A", 0, 2),
        ("B", "A", 1, 2), ("C", "B", 2, 1), ("A", "C", 3, 0),
        ("A", "B", 1, 1), ("B", "C", 0, 2), ("C", "A", 1, 3),
    ]
    return pd.DataFrame(
        [{"HomeTeam": h, "AwayTeam": a, "FTHG": hg, "FTAG": ag} for h, a, hg, ag in rows]
    )


def test_build_predictions_produces_one_record_per_valid_fixture():
    fixtures = [{"date": "2026-03-01", "home_team": "A", "away_team": "B"}]
    records = build_predictions(_toy_matches(), fixtures)

    assert len(records) == 1
    r = records[0]
    assert r["home_team"] == "A" and r["away_team"] == "B"
    assert r["model_h"] + r["model_d"] + r["model_a"] == pytest.approx(1.0, abs=1e-6)
    assert r["market_h"] is None


def test_build_predictions_skips_unknown_team():
    fixtures = [{"date": "2026-03-01", "home_team": "A", "away_team": "존재하지않는팀"}]
    records = build_predictions(_toy_matches(), fixtures)
    assert records == []
