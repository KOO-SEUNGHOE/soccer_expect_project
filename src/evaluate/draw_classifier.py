"""무승부 전용 ML 분류기 실험 (walk-forward).

질문: 시장(배당)이 이미 주는 무승부 확률(1X2 배당에서 마진 제거)보다, 같은 시점에
쓸 수 있는 정보를 더 쓰는 ML 모델이 "이 경기가 무승부로 끝날지"를 더 잘 맞히는가?

이 프로젝트의 지금까지 결론은 "과거 득점으로 만든 모델은 시장을 못 이긴다"였다.
여기서는 방향을 바꿔, 시장이 이미 가진 정보(배당) 위에 시장이 잘 안 쓰는 구조
정보(오버/언더 라인, 아시안 핸디캡 라인, 배당 시가→종가 변동, 팀별 최근 무승부율)를
얹어 무승부 확률만 따로 보정/예측할 수 있는지 본다.

설계 원칙 (CLAUDE.md "모델링 원칙" 준수)
- 시간순 walk-forward: 매 블록마다 "그 블록 이전 경기들로만" 학습하고 다음 경기를 예측.
- 하이퍼파라미터는 테스트 결과를 보기 전에 코드에 고정 (테스트 셋으로 튜닝하지 않음).
- 팀별 최근 무승부율은 "그 경기 이전 경기만" 써서 계산 (누수 방지, 테스트로 검증).
- 비교 기준은 시장(배당) 자체: 시가(Avg)와 종가(AvgC) 두 가지. 종가는 더 많은 정보를
  담고 있으니, ML이 이기는 게 "종가 정보를 쓴 덕분"인지 "진짜 구조 정보 덕분"인지
  구분하려면 종가 기준을 이겨야 의미가 있다.
"""
from __future__ import annotations

import pathlib
from collections import deque
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAW_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam", "FTR",
    "AvgH", "AvgD", "AvgA",            # 시가(마감 전 평균) 1X2 배당
    "AvgCH", "AvgCD", "AvgCA",         # 종가 1X2 배당
    "AvgC>2.5", "AvgC<2.5",            # 종가 오버/언더 2.5 배당
    "AHCh",                            # 종가 아시안 핸디캡 라인 (홈 기준)
]

# EPL 한 시즌은 380경기. 최소 1시즌은 학습 데이터로 확보한 뒤부터 예측한다
# (기존 backtest.py의 min_train_matches=380과 같아서 평가 구간이 동일하다).
MIN_TRAIN_MATCHES = 380
RETRAIN_EVERY = 10  # 라운드(10경기)마다 재학습 — 기존 walk-forward와 동일

TRAILING_WINDOW = 10        # 팀별 최근 무승부율을 볼 직전 경기 수
PRIOR_STRENGTH = 5          # 표본이 적을 때 리그 평균 무승부율 쪽으로 당기는 강도(가상 경기 수)
DEFAULT_DRAW_RATE = 0.25    # 첫 경기 시점에는 리그 평균을 모르므로 쓰는 값


def load_experiment_data(paths: list[str | pathlib.Path]) -> pd.DataFrame:
    """시즌 CSV들을 실험에 필요한 컬럼만 골라 날짜순(동일 날짜는 원래 순서 유지)으로 합친다."""
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in RAW_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"필수 컬럼 누락({p}): {missing}")
        frames.append(df[RAW_COLUMNS].copy())
    data = pd.concat(frames, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y")
    data = data.dropna(subset=["FTR", "AvgCH", "AvgCD", "AvgCA"])
    return data.sort_values("Date", kind="stable").reset_index(drop=True)


def _margin_free_probs(*odds: pd.Series) -> list[pd.Series]:
    """배당들의 역수를 합이 1이 되게 정규화한다 (북메이커 마진 제거, build_features와 같은 방식)."""
    inv = [1.0 / o for o in odds]
    total = sum(inv)
    return [i / total for i in inv]


def _shrunk_rate(past: deque | tuple, prior: float, prior_strength: int) -> float:
    """(무승부 수 + 강도*리그평균) / (경기 수 + 강도) — 표본이 적을수록 리그 평균에 가깝다."""
    return (sum(past) + prior_strength * prior) / (len(past) + prior_strength)


def add_trailing_draw_rates(
    df: pd.DataFrame, window: int = TRAILING_WINDOW, prior_strength: int = PRIOR_STRENGTH
) -> pd.DataFrame:
    """각 경기 시점 기준, 홈/원정 팀의 "직전 window경기 무승부율"을 계산한다 (누수 없음).

    - 경기를 날짜순으로 한 줄씩 훑으며, 특성은 "그 경기 결과를 반영하기 전" 상태에서 먼저 뽑고
      그다음에 결과로 상태를 갱신한다. 즉 어떤 경기의 특성은 그 경기 이전 경기만 본다.
    - 표본이 적은 팀(승격팀/시즌 초반)은 (무승부 수 + prior_strength*리그평균) /
      (경기 수 + prior_strength)로 리그 평균 쪽으로 당겨 극단값을 막는다.
    """
    history: dict[str, deque] = {}
    league_draws = 0
    home_rates: list[float] = []
    away_rates: list[float] = []

    # league_n(지금까지 본 경기 수)은 enumerate 인덱스 = "이 경기 이전 경기 수"와 같다.
    for league_n, (home, away, ftr) in enumerate(zip(df["HomeTeam"], df["AwayTeam"], df["FTR"])):
        prior = league_draws / league_n if league_n else DEFAULT_DRAW_RATE

        home_rates.append(_shrunk_rate(history.get(home, ()), prior, prior_strength))
        away_rates.append(_shrunk_rate(history.get(away, ()), prior, prior_strength))

        is_draw = 1 if ftr == "D" else 0
        for team in (home, away):
            history.setdefault(team, deque(maxlen=window)).append(is_draw)
        league_draws += is_draw

    out = df.copy()
    out["home_draw_rate"] = home_rates
    out["away_draw_rate"] = away_rates
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """실험용 특성 행렬을 만든다. 전부 킥오프 전에 알 수 있는 정보다."""
    _open_h, open_d, _open_a = _margin_free_probs(df["AvgH"], df["AvgD"], df["AvgA"])
    close_h, close_d, close_a = _margin_free_probs(df["AvgCH"], df["AvgCD"], df["AvgCA"])
    under_close, _over_close = _margin_free_probs(df["AvgC<2.5"], df["AvgC>2.5"])

    feats = pd.DataFrame(index=df.index)
    feats["market_open"] = open_d                  # 비교 기준(시가 무승부 확률)
    feats["market_close"] = close_d                # 비교 기준(종가 무승부 확률)
    feats["logit_pD_close"] = np.log(close_d / (1 - close_d))
    feats["abs_pH_pA"] = (close_h - close_a).abs()  # 두 팀이 백중세일수록 0에 가까움
    feats["p_under25"] = under_close                # 저득점 경기 예상일수록 무승부가 잦음
    feats["abs_ah"] = df["AHCh"].abs()              # 핸디캡 라인이 0에 가까울수록 백중세
    feats["pD_move"] = close_d - open_d             # 마감까지 시장이 무승부를 올렸나/내렸나
    feats["home_draw_rate"] = df["home_draw_rate"]
    feats["away_draw_rate"] = df["away_draw_rate"]
    return feats


# 특성 묶음 — 어느 정보가 도움이 되는지 분리해서 보려는 ablation 구성
F_MARKET = ["logit_pD_close"]
F_SHAPE = ["abs_pH_pA", "p_under25", "abs_ah", "pD_move"]
F_TEAM = ["home_draw_rate", "away_draw_rate"]

# 하이퍼파라미터는 여기서 한 번 고정한다 (테스트 성능을 보고 바꾸지 않는다).
LOGIT_C = 1.0
GBM_PARAMS = {
    "max_depth": 2, "learning_rate": 0.05, "max_iter": 150,
    "l2_regularization": 1.0, "min_samples_leaf": 40, "random_state": 0,
}


def _logit_factory() -> Callable:
    return lambda: make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=LOGIT_C, max_iter=1000)
    )


def _gbm_factory() -> Callable:
    return lambda: HistGradientBoostingClassifier(**GBM_PARAMS)


MODEL_SPECS: dict[str, tuple[list[str], Callable]] = {
    "recal(종가배당만)": (F_MARKET, _logit_factory()),
    "logit+구조(O/U,AH,변동)": (F_MARKET + F_SHAPE, _logit_factory()),
    "logit+팀무승부율": (F_MARKET + F_TEAM, _logit_factory()),
    "logit 전체": (F_MARKET + F_SHAPE + F_TEAM, _logit_factory()),
    "GBM 전체": (F_MARKET + F_SHAPE + F_TEAM, _gbm_factory()),
}


def walk_forward_predict(
    feats: pd.DataFrame,
    y: np.ndarray,
    min_train: int = MIN_TRAIN_MATCHES,
    retrain_every: int = RETRAIN_EVERY,
    specs: dict[str, tuple[list[str], Callable]] | None = None,
) -> pd.DataFrame:
    """시간순으로 블록마다 "그 이전 경기만" 학습해 다음 블록의 무승부 확률을 예측한다.

    반환 DataFrame은 평가 구간(min_train 이후) 행만 담고, 모델별 예측 확률 컬럼과
    정답(y), 비교 기준(market_open/market_close) 컬럼을 가진다.
    """
    specs = specs or MODEL_SPECS
    n = len(feats)
    preds = {name: np.full(n, np.nan) for name in specs}

    for start in range(min_train, n, retrain_every):
        end = min(start + retrain_every, n)
        for name, (cols, factory) in specs.items():
            model = factory()
            model.fit(feats.iloc[:start][cols], y[:start])
            preds[name][start:end] = model.predict_proba(feats.iloc[start:end][cols])[:, 1]

    result = pd.DataFrame(preds, index=feats.index).iloc[min_train:].copy()
    result["market_open"] = feats["market_open"].iloc[min_train:].to_numpy()
    result["market_close"] = feats["market_close"].iloc[min_train:].to_numpy()
    result["y"] = y[min_train:]
    return result


def evaluate(
    results: pd.DataFrame,
    reference: str = "market_close",
    top_fracs: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> pd.DataFrame:
    """모델별 로그손실/Brier/AUC, 기준 대비 로그손실 차이(±SE), 상위 k% 무승부 적중률을 계산한다.

    - 로그손실/Brier: 낮을수록 좋다. "무승부 vs 아님" 이진 문제로 계산.
    - AUC: 무작위로 뽑은 (무승부, 비무승부) 한 쌍에서 무승부 쪽에 더 높은 확률을 줄 확률.
      0.5면 못 가림, 1이면 완벽. 확률의 "값"이 아니라 "순서"만 평가한다.
    - 상위 k% 적중률: 모델이 무승부 확률 높다고 본 상위 k% 경기 중 실제 무승부 비율.
      전체 무승부율보다 얼마나 높은지가 "무승부 후보 고르기" 능력이다.
    - 로그손실 차이: 경기별 (기준 손실 - 모델 손실) 평균. 양수면 모델이 낫다.
      SE로 1.96배 범위를 보면 우연인지 감이 온다.
    """
    y = results["y"].to_numpy()
    eps = 1e-6
    cols = [c for c in results.columns if c != "y"]

    def per_match_logloss(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    ref_loss = per_match_logloss(results[reference].to_numpy())
    rows = []
    for col in cols:
        p = results[col].to_numpy()
        loss = per_match_logloss(p)
        diff = ref_loss - loss
        row = {
            "모델": col,
            "로그손실": float(loss.mean()),
            "Brier": float(brier_score_loss(y, np.clip(p, eps, 1 - eps))),
            "AUC": float(roc_auc_score(y, p)),
            f"vs {reference} 개선": float(diff.mean()),
            "개선 SE": float(diff.std(ddof=1) / np.sqrt(len(diff))),
        }
        order = np.argsort(-p, kind="stable")
        for frac in top_fracs:
            k = round(len(p) * frac)
            row[f"상위{int(frac * 100)}% 적중률"] = float(y[order[:k]].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(paths: list[str | pathlib.Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """데이터 로드 -> 특성 생성 -> walk-forward 예측 -> 평가까지 한 번에 수행한다."""
    data = add_trailing_draw_rates(load_experiment_data(paths))
    feats = build_features(data)
    y = (data["FTR"] == "D").astype(int).to_numpy()
    results = walk_forward_predict(feats, y)
    return results, evaluate(results)


def main() -> None:
    raw_dir = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
    paths = sorted(raw_dir.glob("E0_*.csv"))
    results, report = run_experiment(paths)

    base_rate = results["y"].mean()
    print(f"평가 경기 수: {len(results)}  (실제 무승부 {int(results['y'].sum())}건, 무승부율 {base_rate:.3f})")
    print(f"읽는 법: 로그손실/Brier는 낮을수록 좋음, AUC·상위k% 적중률은 높을수록 좋음 "
          f"(적중률 비교 기준 = 전체 무승부율 {base_rate:.3f})\n")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
