"""승/무/패 예측과 무관한, 순수 탐색적(descriptive) 분석 함수 모음.

raw CSV에는 스코어/배당 말고도 슈팅, 유효슈팅, 코너킥, 카드, 심판, 배당
시가/종가 같은 컬럼이 있는데 예측 모델(src/model)에서는 데이터 누수 방지를
위해 이 중 극히 일부만 쓴다. 여기서는 예측 정확도를 주장하지 않고, 이미
확정된 과거 경기 데이터를 그대로 요약/집계해서 보여주는 용도로만 쓴다.
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pandas as pd

from evaluate.calibration import compute_calibration
from features.build_features import implied_probabilities

STATS_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG",
    "Referee", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
    "AvgH", "AvgD", "AvgA",
    "B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA",
]

# raw CSV 파일명(예: "E0_2627.csv")에서 시즌 표기("2026-27")를 뽑아낸다.
# 시즌 필터(전체 시즌 vs 이번 시즌만) UI를 위해 필요한 최소 정보다.
_SEASON_CODE_RE = re.compile(r"E0_(\d{2})(\d{2})")


def season_label_from_path(path: str | pathlib.Path) -> str:
    """"E0_2627.csv" -> "2026-27" 처럼 파일명에서 시즌 표기를 만든다."""
    m = _SEASON_CODE_RE.search(pathlib.Path(path).stem)
    if not m:
        raise ValueError(f"파일명에서 시즌 코드를 찾지 못함: {path}")
    start, end = m.groups()
    return f"20{start}-{end}"


def load_match_stats(path: str | pathlib.Path) -> pd.DataFrame:
    """raw CSV 한 개를 분석용 확장 컬럼(슈팅/카드/심판/배당 변동 포함)으로 불러온다."""
    df = pd.read_csv(path)
    missing = [c for c in STATS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df = df[STATS_COLUMNS].copy()
    df["시즌"] = season_label_from_path(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["FTHG", "FTAG", "FTR"])
    return df


def load_all_match_stats(paths: list[str | pathlib.Path]) -> pd.DataFrame:
    """여러 시즌 raw CSV를 분석용 확장 컬럼으로 불러와 날짜순으로 이어 붙인다."""
    frames = [load_match_stats(p) for p in paths]
    return pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)


def team_attack_defense_profile(df: pd.DataFrame) -> pd.DataFrame:
    """팀별 평균 득점/실점/슈팅/유효슈팅/코너/카드를 홈+원정 합산 관점으로 집계한다."""
    home = df.rename(columns={
        "HomeTeam": "team", "FTHG": "goals_for", "FTAG": "goals_against",
        "HS": "shots", "HST": "shots_on_target", "HC": "corners", "HY": "yellow", "HR": "red",
    })[["team", "goals_for", "goals_against", "shots", "shots_on_target", "corners", "yellow", "red"]]
    away = df.rename(columns={
        "AwayTeam": "team", "FTAG": "goals_for", "FTHG": "goals_against",
        "AS": "shots", "AST": "shots_on_target", "AC": "corners", "AY": "yellow", "AR": "red",
    })[["team", "goals_for", "goals_against", "shots", "shots_on_target", "corners", "yellow", "red"]]
    combined = pd.concat([home, away], ignore_index=True)

    agg = combined.groupby("team").agg(
        경기수=("goals_for", "count"),
        평균득점=("goals_for", "mean"),
        평균실점=("goals_against", "mean"),
        평균슈팅=("shots", "mean"),
        평균유효슈팅=("shots_on_target", "mean"),
        평균코너=("corners", "mean"),
        평균경고=("yellow", "mean"),
        평균퇴장=("red", "mean"),
    ).reset_index().rename(columns={"team": "팀"})

    agg["슈팅정확도%"] = (agg["평균유효슈팅"] / agg["평균슈팅"] * 100).round(1)
    agg["결정력"] = (agg["평균득점"] / agg["평균유효슈팅"]).round(3)
    return agg.sort_values("평균득점", ascending=False).reset_index(drop=True)


def referee_card_stats(df: pd.DataFrame, min_matches: int = 3) -> pd.DataFrame:
    """심판별 평균 카드 수. 표본이 너무 적은 심판(기본 3경기 미만)은 제외한다."""
    work = df.copy()
    work["total_yellow"] = work["HY"] + work["AY"]
    work["total_red"] = work["HR"] + work["AR"]
    agg = work.groupby("Referee").agg(
        경기수=("total_yellow", "count"),
        평균경고=("total_yellow", "mean"),
        평균퇴장=("total_red", "mean"),
    ).reset_index().rename(columns={"Referee": "심판"})
    agg = agg[agg["경기수"] >= min_matches]
    return agg.sort_values("평균경고", ascending=False).reset_index(drop=True)


def home_advantage_stats(df: pd.DataFrame) -> dict:
    """리그 전체 홈 어드밴티지 크기: 홈/무/원정 비율과 평균 득점."""
    result_counts = df["FTR"].value_counts(normalize=True)
    return {
        "n_matches": len(df),
        "home_win_pct": float(result_counts.get("H", 0.0) * 100),
        "draw_pct": float(result_counts.get("D", 0.0) * 100),
        "away_win_pct": float(result_counts.get("A", 0.0) * 100),
        "avg_home_goals": float(df["FTHG"].mean()),
        "avg_away_goals": float(df["FTAG"].mean()),
    }


def _favorite_side(home_odds: float, draw_odds: float, away_odds: float) -> str:
    """세 배당 중 가장 낮은(=가장 유력한) 쪽의 코드(H/D/A)를 반환한다."""
    odds = {"H": home_odds, "D": draw_odds, "A": away_odds}
    return min(odds, key=odds.get)


def odds_movement_accuracy(df: pd.DataFrame) -> dict:
    """배당 시가(오픈) 대비 종가(마감, 킥오프 직전) 유력팀 적중률을 비교한다.

    종가는 라인업 발표 등 시가 이후 정보까지 반영된 값이라, 종가 적중률이
    시가보다 높다면 "마감 직전 배당이 더 정확한 정보를 담고 있다"는 뜻이고,
    이 프로젝트의 예측 정확도 개선과는 별개로 시장 자체의 특성을 보여주는
    순수 관찰 지표다.
    """
    d = df.dropna(subset=["B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA"]).copy()
    d["시가유력"] = d.apply(lambda r: _favorite_side(r["B365H"], r["B365D"], r["B365A"]), axis=1)
    d["종가유력"] = d.apply(lambda r: _favorite_side(r["B365CH"], r["B365CD"], r["B365CA"]), axis=1)
    return {
        "n_matches": len(d),
        "시가_적중률": float((d["시가유력"] == d["FTR"]).mean() * 100),
        "종가_적중률": float((d["종가유력"] == d["FTR"]).mean() * 100),
        "시가종가_유력팀_전환_비율": float((d["시가유력"] != d["종가유력"]).mean() * 100),
    }


_MARKET_LABELS = {"H": "홈승", "D": "무승부", "A": "원정승"}


def market_odds_calibration(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """배당(implied 확률) 구간별로 실제 그 결과가 얼마나 자주 나왔는지 H/D/A
    각각 따로 계산한다 — "무승부 배당이 3.40(implied 29%)일 때 실제로 무승부가
    몇 %로 나왔는가" 같은 질문에 답하기 위함이다(사용자 요청, 2026-09-21).

    evaluate.calibration.compute_calibration()과 완전히 같은 통계 기법(원-vs-
    나머지 캘리브레이션)을 모델 확률이 아니라 시장(배당) 확률에 적용한다.
    H/D/A를 하나로 풀지 않고 마켓별로 따로 계산하는 이유: 한 곡선으로 합치면
    "동일 확률대에서 무승부만 유독 저평가/고평가"인 패턴이 홈/원정승 표본과
    섞여 사라진다.
    """
    d = df.dropna(subset=["AvgH", "AvgD", "AvgA"]).copy()
    probs = d.apply(lambda r: implied_probabilities(r), axis=1, result_type="expand")
    d["market_H"], d["market_D"], d["market_A"] = probs[0], probs[1], probs[2]

    curves = []
    for code, col in [("H", "market_H"), ("D", "market_D"), ("A", "market_A")]:
        curve = compute_calibration(d, {code: col}, n_bins=n_bins)
        if curve.empty:
            continue
        curve["시장"] = _MARKET_LABELS[code]
        curves.append(curve)
    combined = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(
        columns=["bin_center", "predicted_mean", "actual_freq", "n", "시장"]
    )
    combined["배당(대략)"] = (1 / combined["predicted_mean"]).round(2)
    # "23전 18승"처럼 표본 수 대비 적중 횟수를 바로 읽을 수 있게 정수 카운트도 남긴다
    # (actual_freq는 0/1 평균이라 * n이 거의 정확히 정수가 됨 — 부동소수 오차만 반올림).
    combined["적중"] = (combined["actual_freq"] * combined["n"]).round().astype(int)
    return combined


def lookup_market_hit_rate(
    calibration_df: pd.DataFrame, market_code: str, predicted_prob: float, n_bins: int = 10
) -> tuple[float, int] | None:
    """새 배당의 implied 확률이 속하는 과거 구간의 실제 적중 비율·표본수를 찾는다.

    market_odds_calibration()과 반드시 같은 n_bins으로 호출해야 구간 경계가
    맞는다. 해당 구간에 과거 표본이 아예 없으면 None.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = int(np.clip(np.digitize([predicted_prob], bin_edges)[0] - 1, 0, n_bins - 1))
    bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2

    label = _MARKET_LABELS[market_code]
    match = calibration_df[
        (calibration_df["시장"] == label) & np.isclose(calibration_df["bin_center"], bin_center)
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    return float(row["actual_freq"]), int(row["n"])
