import pandas as pd
import pytest

from model.poisson_model import PoissonFootballModel, _dc_tau


def _toy_matches() -> pd.DataFrame:
    # 3개 팀이 서로 여러 번 맞붙는 작은 합성 데이터 (홈팀이 대체로 더 많이 득점하도록 구성).
    rows = []
    results = [
        ("A", "B", 2, 0), ("B", "C", 1, 1), ("C", "A", 0, 2),
        ("B", "A", 1, 2), ("C", "B", 2, 1), ("A", "C", 3, 0),
        ("A", "B", 1, 1), ("B", "C", 0, 2), ("C", "A", 1, 3),
    ]
    for home, away, hg, ag in results:
        rows.append({"HomeTeam": home, "AwayTeam": away, "FTHG": hg, "FTAG": ag})
    return pd.DataFrame(rows)


def test_predict_proba_sums_to_one():
    model = PoissonFootballModel().fit(_toy_matches())
    probs = model.predict_proba("A", "B")
    assert probs["H"] + probs["D"] + probs["A"] == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_predict_proba_unknown_team_raises():
    model = PoissonFootballModel().fit(_toy_matches())
    with pytest.raises(ValueError):
        model.predict_proba("A", "존재하지않는팀")


def test_predict_proba_with_feature_cols_sums_to_one():
    matches = _toy_matches()
    matches["home_form"] = [3.0, 1.0, 0.0, 3.0, 1.0, 3.0, 1.0, 0.0, 3.0]
    matches["away_form"] = [0.0, 1.0, 3.0, 1.0, 3.0, 0.0, 1.0, 3.0, 1.0]

    model = PoissonFootballModel(feature_cols=["form"]).fit(matches)
    probs = model.predict_proba(
        "A", "B",
        home_features={"form": 3.0},
        away_features={"form": 0.0},
    )
    assert probs["H"] + probs["D"] + probs["A"] == pytest.approx(1.0, abs=1e-6)


def test_predict_proba_with_l2_alpha_sums_to_one():
    model = PoissonFootballModel(l2_alpha=1.0).fit(_toy_matches())
    probs = model.predict_proba("A", "B")
    assert probs["H"] + probs["D"] + probs["A"] == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_predict_proba_with_l2_alpha_and_feature_cols_sums_to_one():
    matches = _toy_matches()
    matches["home_form"] = [3.0, 1.0, 0.0, 3.0, 1.0, 3.0, 1.0, 0.0, 3.0]
    matches["away_form"] = [0.0, 1.0, 3.0, 1.0, 3.0, 0.0, 1.0, 3.0, 1.0]

    model = PoissonFootballModel(feature_cols=["form"], l2_alpha=0.5).fit(matches)
    probs = model.predict_proba(
        "A", "B",
        home_features={"form": 3.0},
        away_features={"form": 0.0},
    )
    assert probs["H"] + probs["D"] + probs["A"] == pytest.approx(1.0, abs=1e-6)


def test_dc_tau_only_adjusts_four_low_scores():
    # rho != 0일 때 (0,0)/(1,0)/(0,1)/(1,1)만 1이 아니어야 하고, 나머지는 전부 1.
    rho = -0.15
    lh, la = 1.4, 1.1
    assert _dc_tau(0, 0, lh, la, rho) == pytest.approx(1 - lh * la * rho)
    assert _dc_tau(0, 1, lh, la, rho) == pytest.approx(1 + lh * rho)
    assert _dc_tau(1, 0, lh, la, rho) == pytest.approx(1 + la * rho)
    assert _dc_tau(1, 1, lh, la, rho) == pytest.approx(1 - rho)
    for x, y in [(0, 2), (2, 0), (2, 2), (3, 1), (1, 3)]:
        assert _dc_tau(x, y, lh, la, rho) == 1.0


def test_dc_tau_zero_rho_is_no_correction():
    for x, y in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]:
        assert _dc_tau(x, y, 1.3, 0.9, 0.0) == 1.0


def test_predict_proba_with_dixon_coles_sums_to_one():
    model = PoissonFootballModel(use_dixon_coles=True).fit(_toy_matches())
    assert -1.0 <= model.rho <= 1.0  # minimize_scalar가 지정한 범위 안에서 값을 찾았는지
    probs = model.predict_proba("A", "B")
    assert probs["H"] + probs["D"] + probs["A"] == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_predict_proba_with_dixon_coles_and_feature_cols_sums_to_one():
    matches = _toy_matches()
    matches["home_form"] = [3.0, 1.0, 0.0, 3.0, 1.0, 3.0, 1.0, 0.0, 3.0]
    matches["away_form"] = [0.0, 1.0, 3.0, 1.0, 3.0, 0.0, 1.0, 3.0, 1.0]

    model = PoissonFootballModel(feature_cols=["form"], use_dixon_coles=True).fit(matches)
    probs = model.predict_proba(
        "A", "B",
        home_features={"form": 3.0},
        away_features={"form": 0.0},
    )
    assert probs["H"] + probs["D"] + probs["A"] == pytest.approx(1.0, abs=1e-6)
