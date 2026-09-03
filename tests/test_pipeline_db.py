from pipeline import db


def _sample_record(**overrides):
    record = {
        "match_date": "2026-01-10",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "model_h": 0.5, "model_d": 0.25, "model_a": 0.25,
        "market_h": None, "market_d": None, "market_a": None,
        "predicted_at": "2026-01-05T00:00:00+00:00",
    }
    record.update(overrides)
    return record


def test_insert_and_fetch_unscored(tmp_path):
    conn = db.connect(tmp_path / "predictions.db")
    db.insert_prediction(conn, _sample_record())

    unscored = db.fetch_unscored(conn)
    assert len(unscored) == 1
    assert unscored[0]["home_team"] == "Arsenal"
    assert unscored[0]["actual_result"] is None


def test_insert_is_idempotent_on_same_fixture(tmp_path):
    conn = db.connect(tmp_path / "predictions.db")
    db.insert_prediction(conn, _sample_record(model_h=0.5))
    db.insert_prediction(conn, _sample_record(model_h=0.9))  # 재예측(재학습 후 갱신) 시나리오

    rows = db.fetch_unscored(conn)
    assert len(rows) == 1
    assert rows[0]["model_h"] == 0.9


def test_record_result_moves_prediction_to_scored(tmp_path):
    conn = db.connect(tmp_path / "predictions.db")
    db.insert_prediction(conn, _sample_record())

    db.record_result(
        conn,
        match_date="2026-01-10", home_team="Arsenal", away_team="Chelsea",
        fthg=2, ftag=1, ftr="H", scored_at="2026-01-11T00:00:00+00:00",
    )

    assert db.fetch_unscored(conn) == []
    scored = db.fetch_scored(conn)
    assert len(scored) == 1
    assert scored[0]["actual_result"] == "H"
    assert scored[0]["actual_fthg"] == 2
