"""주간 예측 파이프라인의 결과를 보여주는 Streamlit 대시보드.

실행:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from evaluate.backtest import run_walkforward_backtest, summarize
from evaluate.generate_badge import compute_stats
from features.build_features import load_all_seasons
from pipeline.db import connect, fetch_scored, fetch_unscored

RAW_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
SEASON_FILES = ["E0_2223.csv", "E0_2324.csv", "E0_2425.csv", "E0_2526.csv"]
ENHANCED_FEATURE_COLS = ["form", "venue_form", "rest_days"]
OUTCOME_VECTOR = {"H": (1, 0, 0), "D": (0, 1, 0), "A": (0, 0, 1)}

st.set_page_config(page_title="football-predictor", layout="wide")
st.title("⚽ football-predictor 대시보드")


@st.cache_data(show_spinner="2022-23~2025-26 시즌 walk-forward 백테스트 실행 중 (라운드마다 재학습)...")
def load_backtest_summaries() -> tuple[dict, dict]:
    matches = load_all_seasons([RAW_DIR / f for f in SEASON_FILES])
    baseline = run_walkforward_backtest(matches, feature_cols=None)
    enhanced = run_walkforward_backtest(matches, feature_cols=ENHANCED_FEATURE_COLS)
    return summarize(baseline), summarize(enhanced)


def _match_brier(row: pd.Series) -> float:
    y_vec = OUTCOME_VECTOR[row["actual_result"]]
    return sum((row[f"model_{k}"] - y) ** 2 for k, y in zip(("h", "d", "a"), y_vec))


tab_backtest, tab_live, tab_upcoming = st.tabs(["백테스트", "실전 성능", "다음 라운드 예측"])

with tab_backtest:
    st.subheader("전체 시즌 데이터 walk-forward 백테스트")
    st.caption(
        "라운드(10경기)마다 그 시점까지의 데이터로 재학습하며 다음 라운드를 예측한 결과. "
        "낮을수록(=0에 가까울수록) 좋은 지표입니다."
    )
    baseline_summary, enhanced_summary = load_backtest_summaries()

    col1, col2, col3 = st.columns(3)
    col1.metric("베이스라인 Brier", f"{baseline_summary['model_brier']:.3f}")
    col2.metric(
        "+최근 폼/홈-원정 편차/휴식일수 Brier",
        f"{enhanced_summary['model_brier']:.3f}",
        delta=f"{enhanced_summary['model_brier'] - baseline_summary['model_brier']:+.3f} (베이스라인 대비)",
        delta_color="inverse",
    )
    col3.metric("시장(배당, 마진 제거) Brier", f"{baseline_summary['market_brier']:.3f}")

    st.dataframe(
        pd.DataFrame({"베이스라인": baseline_summary, "+피처": enhanced_summary}),
        use_container_width=True,
    )
    st.caption(
        "현재는 추가 피처가 베이스라인보다 나은 성능을 보이지 않습니다 "
        "(자세한 원인 분석은 README 참고). 실전 예측(다음 라운드 예측 탭)에는 "
        "지금까지 더 나은 성능을 보인 베이스라인 모델을 사용합니다."
    )

with tab_live:
    st.subheader("실전 예측 채점 이력")
    st.caption("주간 파이프라인이 실제로 생성했고 결과가 확정된 예측만 표시합니다.")
    conn = connect()
    scored_rows = fetch_scored(conn)

    if not scored_rows:
        st.info("아직 채점된 실전 예측이 없습니다. 주간 파이프라인이 몇 라운드 돌아간 뒤 표시됩니다.")
    else:
        df = pd.DataFrame([dict(r) for r in scored_rows])
        stats = compute_stats(scored_rows)
        st.metric("누적 Brier score", f"{stats['brier']:.3f}", help=f"{stats['n']}경기 기준")

        df["match_brier"] = df.apply(_match_brier, axis=1)
        df["누적 Brier"] = df["match_brier"].expanding().mean()
        st.line_chart(df.set_index("match_date")["누적 Brier"])

        st.dataframe(
            df[["match_date", "home_team", "away_team", "model_h", "model_d", "model_a", "actual_result"]],
            use_container_width=True,
        )

with tab_upcoming:
    st.subheader("다음 라운드 예측 (결과 미확정)")
    conn = connect()
    unscored_rows = fetch_unscored(conn)

    if not unscored_rows:
        st.info("저장된 예정 경기 예측이 없습니다. `python src/pipeline/predict_next_round.py`를 먼저 실행하세요.")
    else:
        df = pd.DataFrame([dict(r) for r in unscored_rows])
        st.dataframe(
            df[["match_date", "home_team", "away_team", "model_h", "model_d", "model_a"]],
            use_container_width=True,
        )
