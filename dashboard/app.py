"""주간 예측 파이프라인의 결과를 보여주는 Streamlit 대시보드.

실행:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from analytics.descriptive_stats import (
    home_advantage_stats,
    load_all_match_stats,
    lookup_market_hit_rate,
    market_odds_calibration,
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

# 팀 데이터 분석의 "Top 5 리더보드"용 최소 표본 경기 수는 고정값이 아니라
# 그 범위(전체 시즌 or 이번 시즌만)에서 가장 많이 뛴 팀 대비 비율로 정한다.
# 2026-27 시즌에 막 승격한 팀(헐 시티 등)은 지금까지 3경기뿐이라 "최소 실점
# 0.00"처럼 극단값이 우연히 1위를 차지할 수 있었다(사용자 피드백,
# 2026-09-12) — 고정 컷(예: 10경기)은 "이번 시즌만" 범위에서는 모든 팀이
# 아직 10경기를 못 채워 전부 걸러지는 문제가 있어, 비율 기반으로 범위가
# 바뀌어도 자동으로 맞게 스케일되게 했다. 전체 표에는 경기수 컬럼과 함께
# 그대로 나오므로 데이터가 사라지진 않는다.
LEADERBOARD_MIN_MATCHES_RATIO = 0.3

# 심판은 팀과 달리 "신규 승격팀 vs 5시즌 베테랑" 같은 구조적 표본 격차가 없고
# (매 라운드 배정되는 심판 수가 팀 수보다 적어 원래도 편차가 큼), 그보다는
# 순수하게 "1경기만 본 표본"의 노이즈를 거르는 게 목적이라 팀처럼 비율이
# 아닌 낮은 고정값을 쓴다. 이전엔 3이었는데 "이번 시즌만" 범위에서는 가장
# 많이 배정된 심판도 아직 3경기뿐이라 사실상 1명만 남는 문제가 있어 2로 낮춤.
REFEREE_MIN_MATCHES = 2

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

    /* 결과 분포 스택 바 (팀 데이터 분석 탭) — 축구 통계 사이트(Sofascore 등)의
       순위표/분포 바 패턴을 참고: 숫자 나열보다 비율 막대 하나가 한눈에 더 잘 들어온다. */
    .result-bar {
        display: flex; height: 16px; border-radius: 999px; overflow: hidden;
        margin: 0.7rem 0 0.6rem 0; background: rgba(255,255,255,0.05);
    }
    .result-bar span { display: block; height: 100%; }
    .result-legend { display: flex; gap: 1.3rem; font-size: 0.82rem; color: #9ca3af; flex-wrap: wrap; }
    .result-legend .dot {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 0.4rem;
    }
    .result-legend b { color: #e5e7eb; font-family: 'JetBrains Mono', monospace; }

    /* 순위 배지 — 리더보드 스타일 표의 1열 */
    .rank-chip {
        display: inline-flex; align-items: center; justify-content: center;
        width: 22px; height: 22px; border-radius: 7px; font-size: 0.72rem; font-weight: 700;
        font-family: 'JetBrains Mono', monospace; background: rgba(255,255,255,0.08); color: #9ca3af;
        flex-shrink: 0;
    }
    .rank-chip.top { background: rgba(34,197,94,0.18); color: #4ade80; }

    /* 미니 리더보드 (Top 5 공격/수비) — Sofascore 순위표의 "막대 안에 값이 보이는"
       스캔하기 쉬운 형태를 참고 */
    .lb-list { display: flex; flex-direction: column; gap: 0.55rem; }
    .lb-row { display: flex; align-items: center; gap: 0.6rem; }
    .lb-team {
        width: 112px; flex-shrink: 0; font-size: 0.85rem;
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .lb-bar-track { flex: 1; height: 8px; border-radius: 999px; background: rgba(255,255,255,0.06); overflow: hidden; }
    .lb-bar-fill { display: block; height: 100%; border-radius: 999px; }
    .lb-value {
        width: 54px; text-align: right; flex-shrink: 0;
        font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 600;
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


# 1위-2위 확률 격차가 이 아래면 "이 경기는 한쪽으로 확신하기 어렵다"는 뜻으로 본다.
# 사용자 피드백(정배 위주 예측은 원치 않음)에 따라, 무승부를 억지로 1순위로
# 밀어 올리는 대신(데이터상 근거가 없음) 격차가 좁은 경기를 있는 그대로
# "박빙"이라고 밝혀서 과도하게 확신에 찬 "OO팀 승 우세" 문구를 피한다.
CLOSE_MATCH_MARGIN_PP = 18.0
VERY_CLOSE_MATCH_MARGIN_PP = 8.0


def _match_closeness(probs: dict[str, float]) -> tuple[str, float]:
    """확률(0~100) 딕셔너리에서 1위-2위 격차를 보고 박빙 여부를 라벨링한다."""
    ranked = sorted(probs.values(), reverse=True)
    margin = ranked[0] - ranked[1]
    if margin < VERY_CLOSE_MATCH_MARGIN_PP:
        return "🔥 초박빙", margin
    if margin < CLOSE_MATCH_MARGIN_PP:
        return "박빙", margin
    return "확실", margin


def _predicted_outcome_text(row: pd.Series) -> str:
    """예측 결과를 사람이 읽을 문장으로 요약한다.

    격차가 좁은 경기는 "Liverpool 승 우세" 같은 단정적 문구 대신 무승부
    확률을 함께 보여줘서, 실제로는 애매한 경기를 확신에 찬 것처럼 보여주지
    않는다 (예: "Liverpool 우세하나 박빙 (48% · 무승부 30%)").
    """
    probs = {"H": row["홈승%"], "D": row["무승부%"], "A": row["원정승%"]}
    best = _predicted_outcome_code(row)
    pct = probs[best]
    label, _margin = _match_closeness(probs)
    if best == "D":
        return f"무승부 우세 ({pct:.0f}%)"
    team = row["홈팀"] if best == "H" else row["원정팀"]
    if label == "확실":
        return f"{team} 승 우세 ({pct:.0f}%)"
    return f"{team} 우세하나 {label} ({pct:.0f}% · 무승부 {probs['D']:.0f}%)"


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


MARKET_COLOR_SCALE = alt.Scale(domain=["홈승", "무승부", "원정승"], range=["#22c55e", "#9ca3af", "#ef4444"])


def render_market_calibration_chart(curve: pd.DataFrame) -> None:
    """배당(implied 확률) vs 실제 적중 비율을 H/D/A 세 선으로 따로 보여준다.

    render_calibration_chart와 달리 한 선으로 풀지 않는다 — 무승부만의 가격
    왜곡 패턴이 홈/원정승과 섞이면 안 보이기 때문(사용자 요청, 2026-09-21).
    """
    diagonal = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    line = alt.Chart(diagonal).mark_line(strokeDash=[5, 4], color="#6b7280").encode(
        x=alt.X("x", scale=alt.Scale(domain=[0, 1])), y=alt.Y("y", scale=alt.Scale(domain=[0, 1]))
    )
    series = alt.Chart(curve).mark_line(point=True).encode(
        x=alt.X("predicted_mean", scale=alt.Scale(domain=[0, 1]), title="배당 implied 확률"),
        y=alt.Y("actual_freq", scale=alt.Scale(domain=[0, 1]), title="실제 적중 비율"),
        color=alt.Color("시장", scale=MARKET_COLOR_SCALE, legend=alt.Legend(title=None, orient="top")),
        tooltip=[
            "시장",
            alt.Tooltip("배당(대략)", title="배당(대략)"),
            alt.Tooltip("predicted_mean", title="implied 확률", format=".2f"),
            alt.Tooltip("actual_freq", title="실제 적중률", format=".2f"),
            alt.Tooltip("n", title="표본수"),
        ],
    )
    st.altair_chart((line + series).properties(height=360), use_container_width=True)


PROB_COLUMN_CONFIG = {
    "홈승%": st.column_config.ProgressColumn("홈승 확률", format="%.0f%%", min_value=0, max_value=100),
    "무승부%": st.column_config.ProgressColumn("무승부 확률", format="%.0f%%", min_value=0, max_value=100),
    "원정승%": st.column_config.ProgressColumn("원정승 확률", format="%.0f%%", min_value=0, max_value=100),
}

MARKET_COLUMN_CONFIG = {
    **PROB_COLUMN_CONFIG,
    "박빙도": st.column_config.TextColumn("박빙도", help="1위-2위 확률 격차가 좁을수록 결과를 점치기 어려운 경기"),
    "최유력 스코어": st.column_config.TextColumn("최유력 스코어"),
    "BTTS%": st.column_config.ProgressColumn("양팀득점(BTTS) 확률", format="%.0f%%", min_value=0, max_value=100),
    "오버2.5%": st.column_config.ProgressColumn("오버 2.5골 확률", format="%.0f%%", min_value=0, max_value=100),
}


def _favorite_code(row: pd.Series) -> str:
    """배당이 가장 낮은(=가장 유력한) 쪽의 코드(H/D/A)를 반환한다."""
    odds = {"H": row["odds_h"], "D": row["odds_d"], "A": row["odds_a"]}
    return min(odds, key=odds.get)


def _favorite_text(row: pd.Series) -> str:
    """배당 기준 예상 결과를 사람이 읽을 문장으로 요약한다.

    _predicted_outcome_text와 같은 이유로, 시장 확률(격차)이 좁은 경기는
    "OO팀 승 우세"라고 단정하지 않고 박빙임을 함께 보여준다.
    """
    probs = {"H": row["홈승%"], "D": row["무승부%"], "A": row["원정승%"]}
    code = _favorite_code(row)
    label, _margin = _match_closeness(probs)
    if code == "D":
        return f"무승부 우세 (배당 {row['odds_d']:.2f})"
    team = row["home_team"] if code == "H" else row["away_team"]
    if label == "확실":
        return f"{team} 승 우세 (배당 {row[f'odds_{code.lower()}']:.2f})"
    return f"{team} 우세하나 {label} (배당 {row[f'odds_{code.lower()}']:.2f} · 무승부 {probs['D']:.0f}%)"


ODDS_COLUMN_CONFIG = {
    "박빙도": st.column_config.TextColumn("박빙도", help="1위-2위 시장확률 격차가 좁을수록 결과를 점치기 어려운 경기"),
    "odds_h": st.column_config.NumberColumn("홈승 배당", format="%.2f"),
    "odds_d": st.column_config.NumberColumn("무승부 배당", format="%.2f"),
    "odds_a": st.column_config.NumberColumn("원정승 배당", format="%.2f"),
    **PROB_COLUMN_CONFIG,
}


def _render_result_distribution(home_stats: dict) -> None:
    """홈승/무/원정승 비율을 숫자 나열 대신 스택 바 하나로 보여준다(Sofascore류
    순위표의 분포 바 패턴 참고) — 세 숫자를 따로 읽는 것보다 한눈에 비교하기 쉽다."""
    st.markdown(
        f"""
        <div class="result-bar">
          <span style="width:{home_stats['home_win_pct']:.2f}%; background:#22c55e;"></span>
          <span style="width:{home_stats['draw_pct']:.2f}%; background:#6b7280;"></span>
          <span style="width:{home_stats['away_win_pct']:.2f}%; background:#ef4444;"></span>
        </div>
        <div class="result-legend">
          <span><span class="dot" style="background:#22c55e;"></span>홈승 <b>{home_stats['home_win_pct']:.0f}%</b></span>
          <span><span class="dot" style="background:#6b7280;"></span>무승부 <b>{home_stats['draw_pct']:.0f}%</b></span>
          <span><span class="dot" style="background:#ef4444;"></span>원정승 <b>{home_stats['away_win_pct']:.0f}%</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_mini_leaderboard(df: pd.DataFrame, value_col: str, value_fmt: str, bar_color: str, top_n: int = 5) -> None:
    """상위 N개 팀을 순위 배지 + 인라인 막대 형태의 리더보드로 렌더링한다."""
    top = df.head(top_n).reset_index(drop=True)
    max_val = float(top[value_col].max()) or 1.0
    rows = []
    for i, row in top.iterrows():
        rank = i + 1
        bar_pct = max(4.0, row[value_col] / max_val * 100)
        rank_class = "rank-chip top" if rank <= 3 else "rank-chip"
        rows.append(
            f'<div class="lb-row">'
            f'<span class="{rank_class}">{rank}</span>'
            f'<span class="lb-team" title="{row["팀"]}">{row["팀"]}</span>'
            f'<span class="lb-bar-track"><span class="lb-bar-fill" '
            f'style="width:{bar_pct:.0f}%; background:{bar_color};"></span></span>'
            f'<span class="lb-value">{value_fmt.format(row[value_col])}</span>'
            f"</div>"
        )
    st.markdown(f'<div class="lb-list">{"".join(rows)}</div>', unsafe_allow_html=True)


def _team_profile_column_config(profile: pd.DataFrame) -> dict:
    """ProgressColumn의 max_value를 실제 데이터 최댓값 기준으로 동적으로 잡아
    매직 넘버 없이 컬럼마다 적절한 스케일로 막대가 보이게 한다."""
    def prog(col: str, fmt: str, max_value: float | None = None) -> st.column_config.ProgressColumn:
        return st.column_config.ProgressColumn(
            col, format=fmt, min_value=0, max_value=max_value or float(profile[col].max()) * 1.05
        )

    return {
        "순위": st.column_config.NumberColumn("순위", width="small"),
        "평균득점": prog("평균득점", "%.2f"),
        "평균실점": prog("평균실점", "%.2f"),
        "결정력": prog("결정력", "%.3f"),
        "슈팅정확도%": prog("슈팅정확도%", "%.1f%%", max_value=100.0),
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
        closeness = df.apply(
            lambda r: _match_closeness({"H": r["홈승%"], "D": r["무승부%"], "A": r["원정승%"]}), axis=1
        )
        df["박빙도"] = [c[0] for c in closeness]
        df["예상 결과"] = df.apply(_predicted_outcome_text, axis=1)
        df["최유력 스코어"] = df.get("most_likely_score")
        df["BTTS%"] = df.get("btts_yes_prob", pd.Series(dtype=float)) * 100
        df["오버2.5%"] = df.get("over_2_5_prob", pd.Series(dtype=float)) * 100

        league_draw_pct = home_advantage_stats(load_match_stats())["draw_pct"]
        n_close = int((df["박빙도"] != "확실").sum())
        n_draw_favorable = int((df["무승부%"] > league_draw_pct).sum())
        m1, m2, m3 = st.columns(3)
        with m1, st.container(border=True):
            st.metric("이번 라운드 경기 수", f"{len(df)}경기")
        with m2, st.container(border=True):
            st.metric("박빙 경기", f"{n_close}경기", help="1위-2위 확률 격차 18%p 미만")
        with m3, st.container(border=True):
            st.metric(
                "리그 평균보다 무승부 확률 높은 경기",
                f"{n_draw_favorable}경기",
                help=f"리그 전체 평균 무승부 비율({league_draw_pct:.0f}%)보다 이 경기의 무승부 예측 확률이 높은 경우",
            )
        st.caption(
            "정배(가장 낮은 배당)만 밀어주는 방식 대신, 1위-2위 확률 격차가 좁은 경기는 "
            "'박빙'으로 그대로 보여줍니다. 무승부를 억지로 1순위로 올리진 않지만(데이터상 근거가 "
            "없으면 하지 않습니다), 최소한 애매한 경기를 확신에 찬 것처럼 포장하지 않습니다. "
            "최유력 스코어/BTTS(양팀득점)/오버-언더는 모델이 이미 계산해둔 스코어 확률 분포에서 "
            "뽑아낸 값입니다."
        )
        st.dataframe(
            df[["날짜", "홈팀", "원정팀", "박빙도", "예상 결과", "홈승%", "무승부%", "원정승%",
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
        odds_df["홈승%"] = odds_df["implied_h"] * 100
        odds_df["무승부%"] = odds_df["implied_d"] * 100
        odds_df["원정승%"] = odds_df["implied_a"] * 100
        closeness = odds_df.apply(
            lambda r: _match_closeness({"H": r["홈승%"], "D": r["무승부%"], "A": r["원정승%"]}), axis=1
        )
        odds_df["박빙도"] = [c[0] for c in closeness]
        odds_df["예상 결과"] = odds_df.apply(_favorite_text, axis=1)

        hist_market_curve = market_odds_calibration(load_match_stats(), n_bins=10)

        def _historical_hit_rate_pct(row: pd.Series, code: str, implied_col: str) -> float | None:
            result = lookup_market_hit_rate(hist_market_curve, code, row[implied_col], n_bins=10)
            return result[0] * 100 if result else None

        odds_df["홈승 과거적중%"] = odds_df.apply(lambda r: _historical_hit_rate_pct(r, "H", "implied_h"), axis=1)
        odds_df["무승부 과거적중%"] = odds_df.apply(lambda r: _historical_hit_rate_pct(r, "D", "implied_d"), axis=1)
        odds_df["원정승 과거적중%"] = odds_df.apply(lambda r: _historical_hit_rate_pct(r, "A", "implied_a"), axis=1)

        odds_df = odds_df.rename(columns={
            "match_date": "날짜", "home_team": "홈팀", "away_team": "원정팀",
        })

        n_close = int((odds_df["박빙도"] != "확실").sum())
        n_draw_favorite = int((odds_df.apply(_favorite_code, axis=1) == "D").sum())
        b1, b2 = st.columns(2)
        with b1, st.container(border=True):
            st.metric("박빙 경기", f"{n_close}/{len(odds_df)}경기", help="배당(implied) 1위-2위 격차 18%p 미만")
        with b2, st.container(border=True):
            st.metric(
                "무승부가 최유력인 경기",
                f"{n_draw_favorite}경기",
                help="배당이 세 결과 중 가장 낮은(implied 확률이 가장 높은) 쪽이 무승부인 경기",
            )
        st.caption(
            "배당(implied 확률)을 %로도 함께 보여줍니다. 무승부가 '단독 1위'인 경기가 거의 없는 건 "
            "우리 모델만의 한계가 아니라 **실제 시장 배당 자체가 그렇습니다** — 아래 '배당 시가 vs 종가' "
            "탭(팀 데이터 분석)에서 보듯 리그 전체 무승부 비율은 24% 안팎으로, 홈승(44%)보다 항상 "
            "낮은 게 정상입니다. 두 팀이 정말 백중세일 때만 무승부가 1위로 올라옵니다."
        )
        st.caption(
            "'과거적중%' 컬럼은 이 경기의 배당(implied 확률)과 **비슷한 배당대였던 과거 경기들**이 "
            "실제로 그 결과로 끝난 비율입니다(팀 데이터 분석 탭 '배당대별 실제 적중률' 참고). "
            "예: 무승부 배당 implied 29%인데 과거적중%가 35%라면, 이 배당대에서 시장이 무승부를 "
            "과소평가해온 편이라는 뜻 — 표본이 부족한 배당대는 빈 칸으로 남습니다."
        )
        st.dataframe(
            odds_df[["날짜", "홈팀", "원정팀", "박빙도", "예상 결과",
                     "odds_h", "홈승%", "홈승 과거적중%",
                     "odds_d", "무승부%", "무승부 과거적중%",
                     "odds_a", "원정승%", "원정승 과거적중%"]],
            column_config={
                **ODDS_COLUMN_CONFIG,
                "홈승 과거적중%": st.column_config.ProgressColumn("홈승 과거적중률", format="%.0f%%", min_value=0, max_value=100),
                "무승부 과거적중%": st.column_config.ProgressColumn("무승부 과거적중률", format="%.0f%%", min_value=0, max_value=100),
                "원정승 과거적중%": st.column_config.ProgressColumn("원정승 과거적중률", format="%.0f%%", min_value=0, max_value=100),
            },
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
    all_seasons = sorted(stats_df["시즌"].unique())
    current_season = all_seasons[-1]
    scope_full = f"전체 시즌 ({all_seasons[0]}~{current_season} · {len(stats_df)}경기)"
    scope_current = f"이번 시즌만 ({current_season})"
    scope = st.radio("데이터 범위", [scope_full, scope_current], horizontal=True, label_visibility="collapsed")
    if scope == scope_current:
        stats_df = stats_df[stats_df["시즌"] == current_season].reset_index(drop=True)

    home_stats = home_advantage_stats(stats_df)
    col_dist, col_goals = st.columns([2, 1])
    with col_dist, st.container(border=True):
        st.markdown(f'<div class="section-title">리그 전체 결과 분포 <span style="color:#6b7280; font-weight:500; font-size:0.8rem;">· {home_stats["n_matches"]}경기</span></div>', unsafe_allow_html=True)
        _render_result_distribution(home_stats)
    with col_goals, st.container(border=True):
        st.markdown('<div class="section-title">평균 득점</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="display:flex; align-items:baseline; gap:0.5rem; margin-top:0.4rem;">'
            f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:1.6rem; font-weight:700; color:#4ade80;">{home_stats["avg_home_goals"]:.2f}</span>'
            f'<span style="color:#6b7280; font-size:0.85rem;">홈</span>'
            f'<span style="color:#6b7280;">vs</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:1.6rem; font-weight:700; color:#f87171;">{home_stats["avg_away_goals"]:.2f}</span>'
            f'<span style="color:#6b7280; font-size:0.85rem;">원정</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title" style="margin-top:1.4rem;">팀별 공격/수비 리더보드</div>', unsafe_allow_html=True)
    st.caption("홈+원정 통합 평균. 결정력 = 득점/유효슈팅(기회를 잘 살리는 정도). 아래 표는 검색·정렬이 됩니다(헤더 클릭).")
    profile = team_attack_defense_profile(stats_df)
    min_matches = max(1, math.ceil(profile["경기수"].max() * LEADERBOARD_MIN_MATCHES_RATIO))
    qualified = profile[profile["경기수"] >= min_matches]

    lb_col1, lb_col2 = st.columns(2)
    with lb_col1, st.container(border=True):
        st.markdown("**⚽ 득점 Top 5**")
        st.caption(f"최소 {min_matches}경기 이상 표본만 (막 승격한 팀 등 소수 경기 극단값 방지 위해 제외)")
        _render_mini_leaderboard(qualified.sort_values("평균득점", ascending=False), "평균득점", "{:.2f}", "#22c55e")
    with lb_col2, st.container(border=True):
        st.markdown("**🧱 최소 실점 Top 5**")
        st.caption(f"최소 {min_matches}경기 이상 표본만 (막 승격한 팀 등 소수 경기 극단값 방지 위해 제외)")
        _render_mini_leaderboard(qualified.sort_values("평균실점", ascending=True), "평균실점", "{:.2f}", "#f87171")

    search_col, sort_col = st.columns([2, 1])
    with search_col:
        team_search = st.text_input("🔍 팀 검색", placeholder="예: Liverpool", label_visibility="collapsed")
    with sort_col:
        sort_choice = st.selectbox(
            "정렬 기준", ["평균득점 높은순", "평균실점 낮은순", "결정력 높은순", "슈팅정확도 높은순"],
            label_visibility="collapsed",
        )
    sort_map = {
        "평균득점 높은순": ("평균득점", False), "평균실점 낮은순": ("평균실점", True),
        "결정력 높은순": ("결정력", False), "슈팅정확도 높은순": ("슈팅정확도%", False),
    }
    sort_field, sort_asc = sort_map[sort_choice]
    table = profile.sort_values(sort_field, ascending=sort_asc).reset_index(drop=True)
    if team_search:
        table = table[table["팀"].str.contains(team_search, case=False, na=False)].reset_index(drop=True)
    table.insert(0, "순위", range(1, len(table) + 1))

    st.dataframe(
        table[["순위", "팀", "경기수", "평균득점", "평균실점", "평균슈팅", "평균유효슈팅",
               "슈팅정확도%", "결정력", "평균코너", "평균경고", "평균퇴장"]],
        column_config=_team_profile_column_config(profile),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown('<div class="section-title" style="margin-top:1.4rem;">심판별 카드 성향</div>', unsafe_allow_html=True)
    st.caption(f"경기당 평균 경고(옐로카드) 수 상위 15명. 표본이 {REFEREE_MIN_MATCHES}경기 미만인 심판은 제외했습니다.")
    with st.container(border=True):
        referee_df = referee_card_stats(stats_df, min_matches=REFEREE_MIN_MATCHES).head(15)
        ref_chart = alt.Chart(referee_df).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            y=alt.Y("심판", sort="-x", title=None),
            x=alt.X("평균경고", title="경기당 평균 경고 수"),
            color=alt.Color("평균경고", scale=alt.Scale(scheme="oranges"), legend=None),
            tooltip=["심판", "경기수", alt.Tooltip("평균경고", format=".2f")],
        ).properties(height=380)
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

    st.markdown('<div class="section-title" style="margin-top:1.4rem;">배당대별 실제 적중률 — 이 배당일 때 이 결과가 잘 나오나?</div>', unsafe_allow_html=True)
    st.caption(
        "예: 무승부 배당이 약 3.40(implied 29%)이던 경기들만 모아서, 실제로 그중 몇 %가 "
        "무승부로 끝났는지 봅니다. 점선(y=x)보다 위면 그 배당대에서 시장이 해당 결과를 "
        "**과소평가**(실제로 더 자주 나옴), 아래면 **과대평가**한다는 뜻입니다. 홈승/무승부/원정승을 "
        "한 선으로 합치지 않고 따로 그린 이유는 무승부만의 왜곡 패턴이 섞여 안 보이는 걸 막기 위해서입니다."
    )
    with st.container(border=True):
        market_curve = market_odds_calibration(stats_df, n_bins=10)
        render_market_calibration_chart(market_curve)

    with st.expander("배당구간별 정확한 수치 보기"):
        table = market_curve.sort_values(["시장", "배당(대략)"])[["시장", "배당(대략)", "predicted_mean", "actual_freq", "n"]].copy()
        table["predicted_mean"] *= 100
        table["actual_freq"] *= 100
        table = table.rename(columns={"predicted_mean": "implied 확률%", "actual_freq": "실제 적중률%", "n": "표본수"})
        st.dataframe(
            table,
            column_config={
                "implied 확률%": st.column_config.ProgressColumn("implied 확률", format="%.0f%%", min_value=0, max_value=100),
                "실제 적중률%": st.column_config.ProgressColumn("실제 적중률", format="%.0f%%", min_value=0, max_value=100),
            },
            hide_index=True,
            use_container_width=True,
        )
