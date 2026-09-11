"""주간 예측 파이프라인의 결과를 보여주는 Streamlit 대시보드.

실행:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import pathlib
import sys

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from analytics.descriptive_stats import (
    home_advantage_stats,
    load_all_match_stats,
    odds_movement_accuracy,
    referee_card_stats,
    team_attack_defense_profile,
)
from betman.db import connect as betman_connect
from betman.db import fetch_scored as betman_fetch_scored
from betman.db import fetch_unscored as betman_fetch_unscored
from evaluate.backtest import run_walkforward_backtest, summarize
from evaluate.calibration import compute_calibration
from evaluate.generate_badge import compute_stats
from features.build_features import load_all_seasons
from pipeline.db import connect, fetch_scored, fetch_unscored

RAW_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
SEASON_FILES = ["E0_2223.csv", "E0_2324.csv", "E0_2425.csv", "E0_2526.csv", "E0_2627.csv"]
ENHANCED_FEATURE_COLS = ["form", "venue_form", "rest_days"]
OUTCOME_VECTOR = {"H": (1, 0, 0), "D": (0, 1, 0), "A": (0, 0, 1)}
OUTCOME_LABEL = {"H": "홈팀 승", "D": "무승부", "A": "원정팀 승"}

st.set_page_config(page_title="football-predictor", page_icon="⚽", layout="wide")

# ---------------------------------------------------------------------------
# 스타일 — 2026 대시보드 트렌드(다크모드 기본, 절제된 팔레트, bento 카드,
# 데이터엔 모노스페이스 폰트) 반영. Streamlit 자체 커스터마이징 범위 안에서
# CSS만 얹는 방식이라 프레임워크 구조는 그대로 두고 표면만 다듬는다.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

    .hero-title { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.1rem; }
    .hero-sub { color: #9ca3af; font-size: 0.95rem; margin-bottom: 1.6rem; }

    .section-title { font-size: 1.15rem; font-weight: 700; margin: 0.2rem 0 0.6rem 0; }

    /* bento 카드 (st.container(border=True)) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01));
        box-shadow: 0 4px 24px rgba(0,0,0,0.22);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 0.3rem 0.2rem; }

    .card-highlight div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid rgba(34,197,94,0.45) !important;
        background: linear-gradient(180deg, rgba(34,197,94,0.10), rgba(34,197,94,0.02));
    }

    .pill {
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
        background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.3);
    }

    /* 숫자는 모노스페이스로 — 데이터 대시보드 트렌드 */
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 0.82rem; opacity: 0.85; }

    button[data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }

    hr { margin: 1.6rem 0; opacity: 0.15; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero-title">⚽ football-predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">EPL 승/무/패 예측 · 배당 시장 비교 · 배트맨 프로토 배당 대시보드</div>',
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="2022-23~2025-26 시즌 walk-forward 백테스트 실행 중 (라운드마다 재학습)...")
def load_backtest_summaries() -> tuple[dict, dict, dict, pd.DataFrame]:
    matches = load_all_seasons([RAW_DIR / f for f in SEASON_FILES])
    baseline = run_walkforward_backtest(matches, feature_cols=None)
    enhanced = run_walkforward_backtest(matches, feature_cols=ENHANCED_FEATURE_COLS)
    dixon_coles = run_walkforward_backtest(matches, feature_cols=None, use_dixon_coles=True)
    return summarize(baseline), summarize(enhanced), summarize(dixon_coles), dixon_coles


@st.cache_data(show_spinner="슈팅/코너/카드/배당변동 데이터 집계 중...")
def load_match_stats() -> pd.DataFrame:
    return load_all_match_stats([RAW_DIR / f for f in SEASON_FILES])


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


def render_calibration_chart(curve: pd.DataFrame) -> None:
    """예측확률 vs 실제 적중 비율 산점도 + y=x 기준선. 점이 대각선에 가까울수록
    잘 보정된 것이다 (점 크기 = 그 구간에 몇 개 예측이 있었는지)."""
    diagonal = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    line = alt.Chart(diagonal).mark_line(strokeDash=[5, 4], color="#6b7280").encode(
        x=alt.X("x", scale=alt.Scale(domain=[0, 1])), y=alt.Y("y", scale=alt.Scale(domain=[0, 1]))
    )
    points = alt.Chart(curve).mark_circle(color="#22c55e", opacity=0.85).encode(
        x=alt.X("predicted_mean", scale=alt.Scale(domain=[0, 1]), title="예측 확률(구간 평균)"),
        y=alt.Y("actual_freq", scale=alt.Scale(domain=[0, 1]), title="실제 적중 비율"),
        size=alt.Size("n", title="표본 수", scale=alt.Scale(range=[40, 500])),
        tooltip=[
            alt.Tooltip("predicted_mean", title="예측확률", format=".2f"),
            alt.Tooltip("actual_freq", title="실제빈도", format=".2f"),
            alt.Tooltip("n", title="표본수"),
        ],
    )
    st.altair_chart((line + points).properties(height=320), use_container_width=True)


PROB_COLUMN_CONFIG = {
    "홈승%": st.column_config.ProgressColumn("홈승 확률", format="%.0f%%", min_value=0, max_value=100),
    "무승부%": st.column_config.ProgressColumn("무승부 확률", format="%.0f%%", min_value=0, max_value=100),
    "원정승%": st.column_config.ProgressColumn("원정승 확률", format="%.0f%%", min_value=0, max_value=100),
}

MARKET_COLUMN_CONFIG = {
    **PROB_COLUMN_CONFIG,
    "최유력 스코어": st.column_config.TextColumn("최유력 스코어"),
    "BTTS%": st.column_config.ProgressColumn("양팀득점(BTTS) 확률", format="%.0f%%", min_value=0, max_value=100),
    "오버2.5%": st.column_config.ProgressColumn("오버 2.5골 확률", format="%.0f%%", min_value=0, max_value=100),
}


def _favorite_code(row: pd.Series) -> str:
    """배당이 가장 낮은(=가장 유력한) 쪽의 코드(H/D/A)를 반환한다."""
    odds = {"H": row["odds_h"], "D": row["odds_d"], "A": row["odds_a"]}
    return min(odds, key=odds.get)


def _favorite_text(row: pd.Series) -> str:
    code = _favorite_code(row)
    if code == "D":
        return f"무승부 우세 (배당 {row['odds_d']:.2f})"
    team = row["home_team"] if code == "H" else row["away_team"]
    return f"{team} 승 우세 (배당 {row[f'odds_{code.lower()}']:.2f})"


ODDS_COLUMN_CONFIG = {
    "odds_h": st.column_config.NumberColumn("홈승 배당", format="%.2f"),
    "odds_d": st.column_config.NumberColumn("무승부 배당", format="%.2f"),
    "odds_a": st.column_config.NumberColumn("원정승 배당", format="%.2f"),
}

tab_backtest, tab_live, tab_upcoming, tab_betman, tab_insights = st.tabs(
    ["📊 백테스트", "🎯 실전 성능", "🔮 다음 라운드 예측", "🎟️ 배트맨 프로토", "📈 팀 데이터 분석"]
)

with tab_backtest:
    st.markdown('<div class="section-title">전체 시즌 데이터 walk-forward 백테스트</div>', unsafe_allow_html=True)
    st.caption(
        "라운드(10경기)마다 그 시점까지의 데이터로 재학습하며 다음 라운드를 예측한 결과. "
        "**Brier score는 낮을수록(=0에 가까울수록) 예측이 정확하다는 뜻**입니다 "
        "(완벽한 예측=0, 아무렇게나 찍은 예측≈0.66)."
    )
    baseline_summary, enhanced_summary, dc_summary, dc_results = load_backtest_summaries()

    col1, col2, col3, col4 = st.columns(4)
    with col1, st.container(border=True):
        st.metric("① 베이스라인", f"{baseline_summary['model_brier']:.3f}")
    with col2:
        st.markdown('<div class="card-highlight">', unsafe_allow_html=True)
        with st.container(border=True):
            st.metric(
                "② +Dixon-Coles 🟢 실전 사용",
                f"{dc_summary['model_brier']:.4f}",
                delta=f"{dc_summary['model_brier'] - baseline_summary['model_brier']:+.4f} (①보다 개선)",
                delta_color="normal",
            )
        st.markdown("</div>", unsafe_allow_html=True)
    with col3, st.container(border=True):
        st.metric(
            "③ +폼/편차/휴식일수",
            f"{enhanced_summary['model_brier']:.3f}",
            delta=f"{enhanced_summary['model_brier'] - baseline_summary['model_brier']:+.3f} (①보다 나쁨)",
            delta_color="inverse",
        )
    with col4, st.container(border=True):
        st.metric("④ 시장(북메이커 배당)", f"{baseline_summary['market_brier']:.3f}")

    st.caption("④(시장)이 가장 낮습니다 = 아직 이 모델은 배당 시장보다 정확하지 못합니다.")

    st.markdown('<div class="section-title" style="margin-top:1.4rem;">캘리브레이션 — 확률이 실제로 맞나?</div>', unsafe_allow_html=True)
    st.caption(
        "②(Dixon-Coles, 실전 사용 모델) 기준. 점이 점선(y=x) 위에 있으면 그 확률 구간에서 "
        "**과소평가**(예: 30%라 했는데 실제로 더 자주 일어남), 아래에 있으면 **과대평가**입니다."
    )
    with st.container(border=True):
        calibration_curve = compute_calibration(
            dc_results, {"H": "model_H", "D": "model_D", "A": "model_A"}, n_bins=10
        )
        render_calibration_chart(calibration_curve)

    with st.expander("경기 수 등 세부 지표 보기"):
        detail = pd.DataFrame({
            "① 베이스라인": baseline_summary,
            "② +Dixon-Coles": dc_summary,
            "③ +피처 추가": enhanced_summary,
        })
        detail.index = detail.index.map({
            "n_matches": "평가한 경기 수",
            "model_brier": "모델 Brier score",
            "market_brier": "시장 Brier score",
            "model_logloss": "모델 로그손실",
            "market_logloss": "시장 로그손실",
        })
        st.dataframe(detail, use_container_width=True)

    st.caption(
        "③(피처 추가)이 ①보다 나은 성능을 보이지 않아 실전에는 안 씁니다 (자세한 원인 분석은 "
        "README 참고). 모델-시장 앙상블도 시도했지만 시장 대비 이득이 사실상 없었습니다 "
        "(README 참고). 실전 예측에는 ① 베이스라인 + ② Dixon-Coles 저득점(무승부) 보정을 씁니다."
    )

with tab_live:
    st.markdown('<div class="section-title">실전 예측 채점 이력</div>', unsafe_allow_html=True)
    st.caption("주간 파이프라인이 실제로 생성했고 결과가 확정된 예측만 표시합니다.")
    conn = connect()
    scored_rows = fetch_scored(conn)

    if not scored_rows:
        st.info("아직 채점된 실전 예측이 없습니다. 주간 파이프라인이 몇 라운드 돌아간 뒤 표시됩니다.")
    else:
        df = pd.DataFrame([dict(r) for r in scored_rows])
        stats = compute_stats(scored_rows)
        with st.container(border=True):
            st.metric(
                "실전 누적 Brier score",
                f"{stats['brier']:.3f}",
                help=f"{stats['n']}경기 기준. 낮을수록 좋음 (완벽=0, 무작위≈0.66)",
            )

        df["match_brier"] = df.apply(_match_brier, axis=1)
        df["누적 Brier score"] = df["match_brier"].expanding().mean()
        st.caption("아래로 갈수록(=날짜가 지날수록) 값이 어떻게 변하는지 보여줍니다. 낮을수록 좋습니다.")
        with st.container(border=True):
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
    st.markdown('<div class="section-title">다음 라운드 예측 (결과 미확정)</div>', unsafe_allow_html=True)
    conn = connect()
    unscored_rows = fetch_unscored(conn)

    if not unscored_rows:
        st.info("저장된 예정 경기 예측이 없습니다. `python src/pipeline/predict_next_round.py`를 먼저 실행하세요.")
    else:
        df = _with_percent_columns(pd.DataFrame([dict(r) for r in unscored_rows]))
        df["예상 결과"] = df.apply(_predicted_outcome_text, axis=1)
        df["최유력 스코어"] = df.get("most_likely_score")
        df["BTTS%"] = df.get("btts_yes_prob", pd.Series(dtype=float)) * 100
        df["오버2.5%"] = df.get("over_2_5_prob", pd.Series(dtype=float)) * 100
        st.caption(
            "최유력 스코어/BTTS(양팀득점)/오버-언더는 모델이 이미 계산해둔 스코어 확률 분포에서 "
            "뽑아낸 값입니다. 이 값들이 비어 있으면 아직 이전 버전 모델로 저장된 예측이라 "
            "다음 주간 파이프라인 실행 후 채워집니다."
        )
        st.dataframe(
            df[["날짜", "홈팀", "원정팀", "예상 결과", "홈승%", "무승부%", "원정승%",
                "최유력 스코어", "BTTS%", "오버2.5%"]],
            column_config=MARKET_COLUMN_CONFIG,
            hide_index=True,
            use_container_width=True,
        )

with tab_betman:
    st.markdown('<div class="section-title">🎟️ 배트맨(공식 스포츠토토) 프로토 승부식 — 측정된 배당</div>', unsafe_allow_html=True)
    st.markdown('<span class="pill">메인 모델과 별개 섹션</span>', unsafe_allow_html=True)
    st.caption(
        "위 탭들(우리 모델)과는 완전히 별개의 섹션입니다. football-data.co.uk 대신 "
        "한국 공식 스포츠토토 배트맨의 '프로토 승부식' 실제 고정 배당(승/무/패)을 보여줍니다. "
        "이 데이터는 자동화 파이프라인에 포함되지 않고, 로컬에서 수동으로 갱신합니다: "
        "`python src/betman/proto_odds.py` (최초 1회 `playwright install chromium` 필요)."
    )
    betman_conn = betman_connect()

    st.markdown('<div class="section-title" style="margin-top:1rem;">예정 경기 배당</div>', unsafe_allow_html=True)
    betman_unscored = betman_fetch_unscored(betman_conn)
    if not betman_unscored:
        st.info("저장된 배당이 없습니다. `python src/betman/proto_odds.py`를 먼저 실행하세요.")
    else:
        odds_df = pd.DataFrame([dict(r) for r in betman_unscored])
        odds_df["예상 결과"] = odds_df.apply(_favorite_text, axis=1)
        odds_df = odds_df.rename(columns={
            "match_date": "날짜", "home_team": "홈팀", "away_team": "원정팀",
        })
        st.dataframe(
            odds_df[["날짜", "홈팀", "원정팀", "예상 결과", "odds_h", "odds_d", "odds_a"]],
            column_config=ODDS_COLUMN_CONFIG,
            hide_index=True,
            use_container_width=True,
        )

    st.markdown('<div class="section-title" style="margin-top:1rem;">배당 기준 적중률 (결과 확정된 경기만)</div>', unsafe_allow_html=True)
    betman_scored = betman_fetch_scored(betman_conn)
    if not betman_scored:
        st.info(
            "아직 채점된 경기가 없습니다. `python src/ingest/download.py --seasons <현재 시즌>`로 "
            "데이터를 갱신한 뒤 `python src/betman/score_odds.py`를 실행하세요."
        )
    else:
        scored_df = pd.DataFrame([dict(r) for r in betman_scored])
        favorite_code = scored_df.apply(_favorite_code, axis=1)
        scored_df["적중"] = favorite_code == scored_df["actual_result"]
        accuracy = scored_df["적중"].mean() * 100
        with st.container(border=True):
            st.metric(
                "최저배당(유력팀) 적중률",
                f"{accuracy:.0f}%",
                help=f"{len(scored_df)}경기 기준. 배당이 가장 낮은 결과를 '예상'으로 봤을 때 실제로 맞은 비율.",
            )

        scored_df["실제 결과"] = scored_df["actual_result"].map(OUTCOME_LABEL)
        scored_df["판정"] = scored_df["적중"].map({True: "✅ 적중", False: "❌ 빗나감"})
        scored_df = scored_df.rename(columns={
            "match_date": "날짜", "home_team": "홈팀", "away_team": "원정팀",
        })
        st.dataframe(
            scored_df[["날짜", "홈팀", "원정팀", "odds_h", "odds_d", "odds_a", "실제 결과", "판정"]],
            column_config=ODDS_COLUMN_CONFIG,
            hide_index=True,
            use_container_width=True,
        )

with tab_insights:
    st.markdown('<div class="section-title">📈 팀 데이터 분석</div>', unsafe_allow_html=True)
    st.markdown('<span class="pill">순수 탐색적 분석 · 예측이 아님</span>', unsafe_allow_html=True)
    st.caption(
        "raw 데이터에는 스코어/배당 외에도 슈팅, 유효슈팅, 코너킥, 카드, 심판, 배당 시가/종가가 "
        "있는데 예측 모델은 데이터 누수 방지 때문에 이 중 일부만 씁니다. 여기서는 이미 확정된 "
        "과거 경기를 그대로 집계/요약만 합니다 — 승부 예측 정확도와는 무관합니다."
    )
    stats_df = load_match_stats()

    st.markdown('<div class="section-title" style="margin-top:1rem;">리그 전체 홈 어드밴티지</div>', unsafe_allow_html=True)
    home_stats = home_advantage_stats(stats_df)
    c1, c2, c3, c4 = st.columns(4)
    with c1, st.container(border=True):
        st.metric("홈승 비율", f"{home_stats['home_win_pct']:.0f}%")
    with c2, st.container(border=True):
        st.metric("무승부 비율", f"{home_stats['draw_pct']:.0f}%")
    with c3, st.container(border=True):
        st.metric("원정승 비율", f"{home_stats['away_win_pct']:.0f}%")
    with c4, st.container(border=True):
        st.metric(
            "평균 득점(홈 vs 원정)",
            f"{home_stats['avg_home_goals']:.2f} : {home_stats['avg_away_goals']:.2f}",
            help=f"{home_stats['n_matches']}경기 기준",
        )

    st.markdown('<div class="section-title" style="margin-top:1.4rem;">팀별 공격/수비 프로필</div>', unsafe_allow_html=True)
    st.caption("홈+원정 통합 평균. 슈팅정확도% = 유효슈팅/전체슈팅, 결정력 = 득점/유효슈팅 (높을수록 기회를 잘 살림).")
    with st.container(border=True):
        profile = team_attack_defense_profile(stats_df)
        goal_chart_df = profile.melt(
            id_vars="팀", value_vars=["평균득점", "평균실점"], var_name="구분", value_name="값"
        )
        chart = alt.Chart(goal_chart_df).mark_bar().encode(
            x=alt.X("팀", sort=profile.sort_values("평균득점", ascending=False)["팀"].tolist()),
            y=alt.Y("값", title="경기당 평균"),
            color=alt.Color(
                "구분",
                scale=alt.Scale(domain=["평균득점", "평균실점"], range=["#22c55e", "#ef4444"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            xOffset="구분",
            tooltip=["팀", "구분", alt.Tooltip("값", format=".2f")],
        ).properties(height=340)
        st.altair_chart(chart, use_container_width=True)

    with st.expander("팀별 상세 지표 (슈팅/코너/카드/결정력)"):
        st.dataframe(
            profile[["팀", "경기수", "평균득점", "평균실점", "평균슈팅", "평균유효슈팅",
                     "슈팅정확도%", "결정력", "평균코너", "평균경고", "평균퇴장"]],
            hide_index=True,
            use_container_width=True,
        )

    st.markdown('<div class="section-title" style="margin-top:1.4rem;">심판별 카드 성향</div>', unsafe_allow_html=True)
    st.caption("경기당 평균 경고(옐로카드) 수. 표본이 3경기 미만인 심판은 제외했습니다.")
    with st.container(border=True):
        referee_df = referee_card_stats(stats_df).head(15)
        ref_chart = alt.Chart(referee_df).mark_bar(color="#f59e0b").encode(
            x=alt.X("심판", sort="-y"),
            y=alt.Y("평균경고", title="경기당 평균 경고 수"),
            tooltip=["심판", "경기수", alt.Tooltip("평균경고", format=".2f")],
        ).properties(height=320)
        st.altair_chart(ref_chart, use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:1.4rem;">배당 시가 vs 종가 — 어느 쪽이 결과를 더 잘 맞혔나</div>', unsafe_allow_html=True)
    st.caption(
        "종가(킥오프 직전 마감 배당)는 라인업 발표 등 시가 이후 정보까지 반영됩니다. "
        "종가 적중률이 시가보다 높다면, 마감 직전 배당일수록 더 정확한 정보를 담고 있다는 뜻입니다."
    )
    movement = odds_movement_accuracy(stats_df)
    m1, m2, m3 = st.columns(3)
    with m1, st.container(border=True):
        st.metric("시가(오픈) 유력팀 적중률", f"{movement['시가_적중률']:.1f}%")
    with m2, st.container(border=True):
        st.metric(
            "종가(마감) 유력팀 적중률",
            f"{movement['종가_적중률']:.1f}%",
            delta=f"{movement['종가_적중률'] - movement['시가_적중률']:+.1f}%p",
        )
    with m3, st.container(border=True):
        st.metric(
            "시가→종가 유력팀 전환 비율",
            f"{movement['시가종가_유력팀_전환_비율']:.1f}%",
            help="시가와 종가에서 가장 유력한 팀(가장 낮은 배당)이 바뀐 경기의 비율",
        )
