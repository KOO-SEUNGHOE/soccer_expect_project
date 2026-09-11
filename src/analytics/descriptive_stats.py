"""승/무/패 예측과 무관한, 순수 탐색적(descriptive) 분석 함수 모음.

raw CSV에는 스코어/배당 말고도 슈팅, 유효슈팅, 코너킥, 카드, 심판, 배당
시가/종가 같은 컬럼이 있는데 예측 모델(src/model)에서는 데이터 누수 방지를
위해 이 중 극히 일부만 쓴다. 여기서는 예측 정확도를 주장하지 않고, 이미
확정된 과거 경기 데이터를 그대로 요약/집계해서 보여주는 용도로만 쓴다.
"""
from __future__ import annotations

import pathlib

import pandas as pd

STATS_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG",
    "Referee", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
    "B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA",
]


def load_match_stats(path: str | pathlib.Path) -> pd.DataFrame:
    """raw CSV 한 개를 분석용 확장 컬럼(슈팅/카드/심판/배당 변동 포함)으로 불러온다."""
    df = pd.read_csv(path)
    missing = [c for c in STATS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df = df[STATS_COLUMNS].copy()
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
