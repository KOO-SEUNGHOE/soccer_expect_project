import datetime as dt

import pytest

from betman import db as betman_db
from betman.fetch_results import (
    FinishedMatch,
    find_stored_date,
    parse_finished_matches,
    parse_kst_date,
    result_code,
    score_matches,
)
from pipeline import db as pred_db


def _row(when="26.10.10(토) 23:00", teams=("첼시", "본머스"), scores=("5", "1"), badge="종료"):
    return {"when": when, "teams": list(teams), "scores": list(scores), "badge": badge}


def test_parse_kst_date():
    assert parse_kst_date("26.10.10(토) 23:00") == dt.date(2026, 10, 10)
    assert parse_kst_date("26.09.02(수)") == dt.date(2026, 9, 2)


def test_parse_kst_date_rejects_garbage():
    with pytest.raises(ValueError):
        parse_kst_date("오늘 저녁")


def test_parse_finished_matches_keeps_only_finished_with_scores():
    rows = [
        _row(),  # 종료
        _row(teams=("아스널", "리즈U"), scores=("", ""), badge="경기전"),  # 아직 안 함
        _row(teams=("리버풀", "맨체스C"), scores=("", ""), badge="종료"),  # 종료인데 스코어 없음(이상 케이스)
    ]
    matches = parse_finished_matches(rows)
    assert matches == [FinishedMatch(dt.date(2026, 10, 10), "Chelsea", "Bournemouth", 5, 1)]


def test_parse_finished_matches_skips_unmapped_team(capsys):
    matches = parse_finished_matches([_row(teams=("모르는팀", "첼시"))])
    assert matches == []
    assert "모르는팀" in capsys.readouterr().err


def test_result_code():
    assert result_code(2, 1) == "H"
    assert result_code(0, 2) == "A"
    assert result_code(1, 1) == "D"


def test_find_stored_date_matches_kst_date_or_previous_day():
    # predictions.db는 UTC 기준 날짜라 KST 일요일 새벽 경기가 토요일로 저장돼 있다.
    match = FinishedMatch(dt.date(2026, 10, 11), "Man United", "Tottenham", 1, 1)
    rows = [{"home_team": "Man United", "away_team": "Tottenham", "match_date": "2026-10-10"}]
    assert find_stored_date(rows, match) == "2026-10-10"

    rows_far = [{"home_team": "Man United", "away_team": "Tottenham", "match_date": "2026-10-20"}]
    assert find_stored_date(rows_far, match) is None


def test_score_matches_updates_both_dbs_with_their_own_date_convention(tmp_path):
    pred_conn = pred_db.connect(tmp_path / "predictions.db")
    pred_db.insert_prediction(pred_conn, {
        "match_date": "2026-10-10", "home_team": "Man United", "away_team": "Tottenham",
        "model_h": 0.4, "model_d": 0.3, "model_a": 0.3,
        "market_h": None, "market_d": None, "market_a": None,
        "predicted_at": "2026-10-09T00:00:00+00:00",
    })
    bet_conn = betman_db.connect(tmp_path / "betman_odds.db")
    betman_db.insert_odds(bet_conn, {
        "match_date": "2026-10-11", "kickoff": "2026-10-11T01:30:00",
        "home_team": "Man United", "away_team": "Tottenham",
        "odds_h": 2.5, "odds_d": 3.3, "odds_a": 2.8,
        "implied_h": 0.38, "implied_d": 0.29, "implied_a": 0.33,
        "scraped_at": "2026-10-10T00:00:00+00:00",
    })
    finished = [FinishedMatch(dt.date(2026, 10, 11), "Man United", "Tottenham", 1, 1)]

    assert score_matches(pred_conn, pred_db.fetch_unscored, pred_db.record_result, finished) == 1
    assert score_matches(bet_conn, betman_db.fetch_unscored, betman_db.record_result, finished) == 1

    assert pred_db.fetch_unscored(pred_conn) == []
    assert betman_db.fetch_unscored(bet_conn) == []
    assert pred_db.fetch_scored(pred_conn)[0]["actual_result"] == "D"


def test_score_matches_is_noop_when_nothing_unscored(tmp_path):
    conn = pred_db.connect(tmp_path / "predictions.db")
    finished = [FinishedMatch(dt.date(2026, 10, 11), "Man United", "Tottenham", 1, 1)]
    assert score_matches(conn, pred_db.fetch_unscored, pred_db.record_result, finished) == 0
