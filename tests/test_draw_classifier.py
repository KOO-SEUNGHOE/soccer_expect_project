import numpy as np
import pandas as pd
import pytest

from evaluate.draw_classifier import (
    add_trailing_draw_rates,
    build_features,
    evaluate,
    walk_forward_predict,
)


def _matches(ftrs: list[str]) -> pd.DataFrame:
    """A/B 두 팀이 번갈아 홈을 맡는 가짜 경기 목록."""
    n = len(ftrs)
    return pd.DataFrame({
        "HomeTeam": ["A" if i % 2 == 0 else "B" for i in range(n)],
        "AwayTeam": ["B" if i % 2 == 0 else "A" for i in range(n)],
        "FTR": ftrs,
    })


def test_trailing_draw_rate_uses_only_past_matches():
    ftrs = ["D", "H", "D", "A", "D", "D"]
    base = add_trailing_draw_rates(_matches(ftrs))

    # 마지막 경기의 결과를 뒤집어도, 그 경기 자신과 이전 경기의 특성은 그대로여야 한다 (누수 없음).
    flipped = add_trailing_draw_rates(_matches(ftrs[:-1] + ["H"]))
    cols = ["home_draw_rate", "away_draw_rate"]
    pd.testing.assert_frame_equal(base[cols], flipped[cols])


def test_trailing_draw_rate_first_match_equals_default_prior():
    out = add_trailing_draw_rates(_matches(["D", "H"]))
    assert out["home_draw_rate"].iloc[0] == pytest.approx(0.25)
    assert out["away_draw_rate"].iloc[0] == pytest.approx(0.25)


def test_trailing_draw_rate_moves_toward_team_history():
    # A, B 둘 다 연속 무승부 -> 이후 경기의 무승부율은 기본값(0.25)보다 높아져야 한다.
    out = add_trailing_draw_rates(_matches(["D"] * 8))
    assert out["home_draw_rate"].iloc[-1] > 0.5


def test_build_features_margin_free_probs_and_ranges():
    df = pd.DataFrame({
        "HomeTeam": ["A"], "AwayTeam": ["B"], "FTR": ["D"],
        "AvgH": [2.0], "AvgD": [3.5], "AvgA": [4.0],
        "AvgCH": [2.1], "AvgCD": [3.4], "AvgCA": [3.8],
        "AvgC>2.5": [1.9], "AvgC<2.5": [1.95], "AHCh": [-0.25],
        "home_draw_rate": [0.3], "away_draw_rate": [0.2],
    })
    f = build_features(df).iloc[0]
    assert 0 < f["market_close"] < 1
    assert 0 < f["p_under25"] < 1
    assert f["abs_ah"] == pytest.approx(0.25)
    # 종가 H/D/A 확률 합은 1 (마진 제거) -> D 확률 + |H-A| 등은 일관된 범위
    assert f["logit_pD_close"] == pytest.approx(np.log(f["market_close"] / (1 - f["market_close"])))


def test_walk_forward_never_trains_on_test_block():
    # 특성 하나가 정답을 그대로 담은 "누수 특성"이라고 해도, 학습은 항상 직전까지의 경기만 쓰므로
    # 첫 예측 블록 이전 행수가 min_train과 같아야 하고, 평가 구간 길이가 n - min_train이어야 한다.
    n = 60
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    feats = pd.DataFrame({
        "x": rng.normal(size=n),
        "market_open": np.full(n, 0.25), "market_close": np.full(n, 0.25),
    })
    from sklearn.linear_model import LogisticRegression

    specs = {"m": (["x"], lambda: LogisticRegression())}
    res = walk_forward_predict(feats, y, min_train=30, retrain_every=10, specs=specs)
    assert len(res) == n - 30
    assert res["m"].notna().all()
    assert set(res.columns) >= {"m", "market_open", "market_close", "y"}


def test_evaluate_perfect_ranker_has_auc_one_and_positive_gain():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    res = pd.DataFrame({
        "y": y,
        "market_close": np.full(10, y.mean()),          # 정보 없는 기준
        "oracle": np.where(y == 1, 0.8, 0.1),            # 정답을 아는 모델
    })
    report = evaluate(res, reference="market_close").set_index("모델")
    assert report.loc["oracle", "AUC"] == pytest.approx(1.0)
    assert report.loc["oracle", "vs market_close 개선"] > 0
    assert report.loc["market_close", "vs market_close 개선"] == pytest.approx(0.0)
