"""경기 시점 이전 데이터만 사용해 팀별 '최근 폼', '홈/원정 편차', '휴식일수'를 계산한다.

데이터 누수 방지 원칙(CLAUDE.md 모델링 원칙 4): 각 경기 행의 피처는 반드시 그
경기 이전에 이미 끝난 경기 결과만으로 계산한다. 팀 이력이 form_window보다
짧은 경우(시즌 초반, 승격팀 등)는 표본이 부족하다고 보고 값을 억지로 채우지
않고 NaN으로 남긴다 — 호출 측(모델 학습/백테스트)에서 이를 걸러낸다.
"""
from __future__ import annotations

import pandas as pd

# 시즌 사이 여름 휴식기 등 이례적으로 긴 공백이 그대로 선형 공변량에 들어가면
# 그 값 하나가 회귀 계수를 왜곡할 수 있어 상한을 둔다 (3주 이상은 "충분히 쉼"으로 취급).
MAX_REST_DAYS = 21

_MatchRecord = tuple[pd.Timestamp, bool, int, int]  # (date, is_home, goals_for, goals_against)


def _recent_points(history: list[_MatchRecord], window: int) -> float:
    """직전 window경기(홈/원정 무관) 승점 합. 표본 부족 시 NaN."""
    if len(history) < window:
        return float("nan")
    recent = history[-window:]
    points = 0
    for _, _is_home, gf, ga in recent:
        if gf > ga:
            points += 3
        elif gf == ga:
            points += 1
    return float(points)


def _recent_venue_goal_diff(history: list[_MatchRecord], window: int, is_home: bool) -> float:
    """같은 장소(홈팀 -> 홈경기만, 원정팀 -> 원정경기만)에서의 직전 window경기 득실차 합.
    표본 부족 시 NaN.
    """
    venue_matches = [m for m in history if m[1] == is_home]
    if len(venue_matches) < window:
        return float("nan")
    recent = venue_matches[-window:]
    return float(sum(gf - ga for _, _, gf, ga in recent))


def _rest_days(history: list[_MatchRecord], current_date: pd.Timestamp) -> float:
    """직전 경기 이후 경과일. 첫 경기(이력 없음)는 NaN."""
    if not history:
        return float("nan")
    last_date = history[-1][0]
    days = (current_date - last_date).days
    return float(min(days, MAX_REST_DAYS))


def add_rolling_features(matches: pd.DataFrame, form_window: int = 5) -> pd.DataFrame:
    """matches(날짜순 정렬 필요)에 다음 컬럼을 추가해 반환한다.

    - home_form / away_form: 최근 form_window경기 승점 합
    - home_venue_form / away_venue_form: 같은 장소에서의 최근 form_window경기 득실차 합
    - home_rest_days / away_rest_days: 직전 경기 이후 경과일 (상한 MAX_REST_DAYS)
    """
    matches = matches.sort_values("Date").reset_index(drop=True)
    history: dict[str, list[_MatchRecord]] = {}

    home_form, away_form = [], []
    home_venue_form, away_venue_form = [], []
    home_rest, away_rest = [], []

    for row in matches.itertuples():
        home, away = row.HomeTeam, row.AwayTeam
        home_hist = history.get(home, [])
        away_hist = history.get(away, [])

        home_form.append(_recent_points(home_hist, form_window))
        away_form.append(_recent_points(away_hist, form_window))
        home_venue_form.append(_recent_venue_goal_diff(home_hist, form_window, is_home=True))
        away_venue_form.append(_recent_venue_goal_diff(away_hist, form_window, is_home=False))
        home_rest.append(_rest_days(home_hist, row.Date))
        away_rest.append(_rest_days(away_hist, row.Date))

        history.setdefault(home, []).append((row.Date, True, row.FTHG, row.FTAG))
        history.setdefault(away, []).append((row.Date, False, row.FTAG, row.FTHG))

    out = matches.copy()
    out["home_form"] = home_form
    out["away_form"] = away_form
    out["home_venue_form"] = home_venue_form
    out["away_venue_form"] = away_venue_form
    out["home_rest_days"] = home_rest
    out["away_rest_days"] = away_rest
    return out


def latest_team_state(matches: pd.DataFrame, form_window: int = 5) -> dict[str, dict]:
    """아직 열리지 않은 다음 경기의 피처를 계산하기 위해, 보유한 마지막 경기
    시점까지의 각 팀 최신 상태(폼/홈-원정 편차/마지막 경기 날짜)를 반환한다.

    add_rolling_features는 "이미 있는 경기 행"의 피처만 계산하므로, 예정된
    경기처럼 아직 행 자체가 없는 경우에는 이 함수로 각 팀의 최신 상태를 구한 뒤
    호출 측에서 예정 경기 날짜와 조합해 rest_days 등을 계산해야 한다.
    """
    matches = matches.sort_values("Date").reset_index(drop=True)
    history: dict[str, list[_MatchRecord]] = {}
    for row in matches.itertuples():
        home, away = row.HomeTeam, row.AwayTeam
        history.setdefault(home, []).append((row.Date, True, row.FTHG, row.FTAG))
        history.setdefault(away, []).append((row.Date, False, row.FTAG, row.FTHG))

    state = {}
    for team, hist in history.items():
        state[team] = {
            "form": _recent_points(hist, form_window),
            "home_venue_form": _recent_venue_goal_diff(hist, form_window, is_home=True),
            "away_venue_form": _recent_venue_goal_diff(hist, form_window, is_home=False),
            "last_match_date": hist[-1][0],
        }
    return state
