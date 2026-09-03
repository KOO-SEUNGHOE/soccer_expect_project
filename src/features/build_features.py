"""raw CSV를 불러와 모델링에 필요한 최소 컬럼으로 정리한다.

데이터 누수 방지 원칙(CLAUDE.md 참고): 여기서는 각 경기의 '결과'와 '배당'만
정리하고, 폼/롤링 통계 같은 시점 의존적 피처는 model 단계에서 학습 시점
기준으로 별도 계산한다.
"""
from __future__ import annotations

import pathlib

import pandas as pd

RAW_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "AvgH", "AvgD", "AvgA",  # 여러 북메이커 평균 종가 배당 (마진 포함)
]


def load_raw_csv(path: str | pathlib.Path) -> pd.DataFrame:
    """raw CSV 한 개를 읽어 날짜순으로 정렬된 DataFrame으로 반환한다."""
    df = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df = df[RAW_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["FTHG", "FTAG", "FTR"])
    return df


def load_all_seasons(paths: list[str | pathlib.Path]) -> pd.DataFrame:
    """여러 시즌 raw CSV를 각각 정리한 뒤 날짜순으로 이어 붙인다.

    시즌 파일들을 그냥 concat만 하면 되지 않고 각 파일 내부에서 이미 날짜순
    정렬을 보장한 뒤 합치고 다시 전체 정렬하는 이유는, 파일명 순서(예: 시즌
    코드 문자열 정렬)가 항상 실제 날짜 순서와 일치한다고 가정하지 않기 위함이다.
    """
    frames = [load_raw_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)


def clean_for_market_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """배당 대비 성능 비교용: 배당(AvgH/AvgD/AvgA)이 결측인 경기는 제외한다.
    (일부 라운드는 특정 북메이커 데이터가 비어 있어 소수 경기가 빠질 수 있음)
    """
    return df.dropna(subset=["AvgH", "AvgD", "AvgA"]).reset_index(drop=True)


def implied_probabilities(row: pd.Series) -> tuple[float, float, float]:
    """배당(AvgH/AvgD/AvgA)에서 북메이커 마진(overround)을 제거한 '진짜' 확률을 계산한다.

    방법: 각 배당의 역수(1/odds)를 구한 뒤, 합이 1이 되도록 정규화한다.
    마진 제거 없이 그냥 1/odds를 확률로 쓰면 항상 합이 1보다 커서
    북메이커가 남기는 이윤만큼 왜곡된 값이 된다.
    """
    inv_h, inv_d, inv_a = 1 / row["AvgH"], 1 / row["AvgD"], 1 / row["AvgA"]
    overround = inv_h + inv_d + inv_a
    return inv_h / overround, inv_d / overround, inv_a / overround
