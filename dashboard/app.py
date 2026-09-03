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
OUTCOME_LABEL = {"H": "홈팀 승", "D": "무승부", "A": "원정팀 승"}

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


def _predicted_outcome_code(row: pd.Series) -> str:
    """세 확률 중 가장 높은 쪽의 코드(H/D/A)를 반환한다."""
    probs = {"H": row["홈승%"], "D": row["무승부%"], "A": row["원정승%"]}
    return max(probs, key=probs.get)


def _predicted_outcome_text(row: pd.Series) -> str:
    """예측 결과를 사람이 읽을 문장으로 요약한다 (예: "Liverpool 승 우세 (75%)")."""
    probs = {"H": row["홈승%"], "D": row["무승부%"], "A": row["원정승%"]}
    best = _predicted_outcome_code(row)
    pct = probs[best]
    if best == "D":
        return f"무승부 우세 ({pct:.0f}%)"
    team = row["홈팀"] if best == "H" else row["원정팀"]
    return f"{team} 승 우세 ({pct:.0f}%)"


def _with_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    """model_h/d/a(0~1 소수)를 퍼센트(0~100) 컬럼으로 바꾸고 한글 이름을 붙인다."""
    out = df.rename(columns={
        "match_date": "날짜", "home_team": "홈팀", "away_team": "원정팀",
    })
    out["홈승%"] = out["model_h"] * 100
    out["무승부%"] = out["model_d"] * 100
    out["원정승%"] = out["model_a"] * 100
    return out


PROB_COLUMN_CONFIG = {
    "홈승%": st.column_config.ProgressColumn("홈승 확률", format="%.0f%%", min_value=0, max_value=100),
    "무승부%": st.column_config.ProgressColumn("무승부 확률", format="%.0f%%", min_value=0, max_value=100),
    "원정승%": st.column_config.ProgressColumn("원정승 확률", format="%.0f%%", min_value=0, max_value=100),
}

tab_backtest, tab_live, tab_upcoming = st.tabs(["백테스트", "실전 성능", "다음 라운드 예측"])

with tab_backtest:
    st.subheader("전체 시즌 데이터 walk-forward 백테스트")
    st.caption(
        "라운드(10경기)마다 그 시점까지의 데이터로 재학습하며 다음 라운드를 예측한 결과. "
        "**Brier score는 낮을수록(=0에 가까울수록) 예측이 정확하다는 뜻**입니다 "
        "(완벽한 예측=0, 아무렇게나 찍은 예측≈0.66)."
    )
    baseline_summary, enhanced_summary = load_backtest_summaries()

    col1, col2, col3 = st.columns(3)
    col1.metric("① 베이스라인 모델", f"{baseline_summary['model_brier']:.3f}")
    col2.metric(
        "② +최근 폼/홈-원정 편차/휴식일수",
        f"{enhanced_summary['model_brier']:.3f}",
        delta=f"{enhanced_summary['model_brier'] - baseline_summary['model_brier']:+.3f} (①보다 나쁨)",
        delta_color="inverse",
    )
    col3.metric("③ 시장(북메이커 배당)", f"{baseline_summary['market_brier']:.3f}")
    st.caption("③(시장)이 가장 낮습니다 = 아직 이 모델은 배당 시장보다 정확하지 못합니다.")

    with st.expander("경기 수 등 세부 지표 보기"):
        detail = pd.DataFrame({"① 베이스라인": baseline_summary, "② +피처 추가": enhanced_summary})
        detail.index = detail.index.map({
            "n_matches": "평가한 경기 수",
            "model_brier": "모델 Brier score",
            "market_brier": "시장 Brier score",
            "model_logloss": "모델 로그손실",
            "market_logloss": "시장 로그손실",
        })
        st.dataframe(detail, use_container_width=True)

    st.caption(
        "현재는 ②(피처 추가)가 ①(베이스라인)보다 나은 성능을 보이지 않습니다 "
        "(자세한 원인 분석은 README 참고). 그래서 실전 예측(다음 라운드 예측 탭)에는 "
        "지금까지 더 나은 성능을 보인 ① 베이스라인 모델을 사용합니다."
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
        st.metric(
            "실전 누적 Brier score",
            f"{stats['brier']:.3f}",
            help=f"{stats['n']}경기 기준. 낮을수록 좋음 (완벽=0, 무작위≈0.66)",
        )

        df["match_brier"] = df.apply(_match_brier, axis=1)
        df["누적 Brier score"] = df["match_brier"].expanding().mean()
        st.caption("아래로 갈수록(=날짜가 지날수록) 값이 어떻게 변하는지 보여줍니다. 낮을수록 좋습니다.")
        st.line_chart(df.set_index("match_date")["누적 Brier score"])

        display = _with_percent_columns(df)
        display["실제 결과"] = display["actual_result"].map(OUTCOME_LABEL)
        predicted_code = display.apply(_predicted_outcome_code, axis=1)
        display["예측 판정"] = (predicted_code == display["actual_result"]).map({True: "✅ 적중", False: "❌ 빗나감"})
        st.dataframe(
            display[["날짜", "홈팀", "원정팀", "홈승%", "무승부%", "원정승%", "실제 결과", "예측 판정"]],
            column_config=PROB_COLUMN_CONFIG,
            hide_index=True,
            use_container_width=True,
        )

with tab_upcoming:
    st.subheader("다음 라운드 예측 (결과 미확정)")
    conn = connect()
    unscored_rows = fetch_unscored(conn)

    if not unscored_rows:
        st.info("저장된 예정 경기 예측이 없습니다. `python src/pipeline/predict_next_round.py`를 먼저 실행하세요.")
    else:
        df = _with_percent_columns(pd.DataFrame([dict(r) for r in unscored_rows]))
        df["예상 결과"] = df.apply(_predicted_outcome_text, axis=1)
        st.dataframe(
            df[["날짜", "홈팀", "원정팀", "예상 결과", "홈승%", "무승부%", "원정승%"]],
            column_config=PROB_COLUMN_CONFIG,
            hide_index=True,
            use_container_width=True,
        )
