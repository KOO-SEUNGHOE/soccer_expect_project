import math

import pandas as pd

from features.rolling_features import add_rolling_features


def _matches(rows):
    df = pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def test_first_appearance_has_no_form_or_rest():
    matches = _matches([("2024-01-01", "A", "B", 1, 0)])
    out = add_rolling_features(matches, form_window=5)
    row = out.iloc[0]
    assert math.isnan(row["home_form"])
    assert math.isnan(row["away_form"])
    assert math.isnan(row["home_rest_days"])
    assert math.isnan(row["away_rest_days"])


def test_form_uses_only_past_matches_not_future():
    # A는 5번의 경기를 치른 뒤 6번째 경기를 갖는다. 6번째 경기의 form은 앞선 5경기로만
    # 계산되어야 하고, 이후(7번째) 결과가 6번째 행의 값에 영향을 주면 안 된다 (누수 검증).
    rows = [
        ("2024-01-01", "A", "X", 3, 0),  # A 승 (3점)
        ("2024-01-08", "A", "X", 3, 0),
        ("2024-01-15", "A", "X", 3, 0),
        ("2024-01-22", "A", "X", 0, 0),  # 무 (1점)
        ("2024-01-29", "A", "X", 0, 3),  # 패 (0점)
        ("2024-02-05", "A", "B", 1, 1),  # 6번째: form은 앞 5경기 = 3+3+3+1+0 = 10
        ("2024-02-12", "A", "B", 0, 5),  # 7번째: 이 대패가 6번째 행에 영향을 주면 안 됨
    ]
    matches = _matches(rows)
    out = add_rolling_features(matches, form_window=5)

    sixth = out.iloc[5]
    assert sixth["home_form"] == 10.0

    # 위 5경기 결과만 바뀌지 않는 한, 7번째 경기 결과를 바꿔도 6번째 행은 그대로여야 한다.
    rows_alt = rows.copy()
    rows_alt[6] = ("2024-02-12", "A", "B", 5, 0)  # 7번째 결과를 정반대로 변경
    out_alt = add_rolling_features(_matches(rows_alt), form_window=5)
    assert out_alt.iloc[5]["home_form"] == sixth["home_form"]


def test_rest_days_computed_from_previous_match():
    rows = [
        ("2024-01-01", "A", "B", 1, 0),
        ("2024-01-08", "A", "C", 1, 0),  # A는 직전 경기(1/1)로부터 7일 후
    ]
    out = add_rolling_features(_matches(rows), form_window=1)
    assert out.iloc[1]["home_rest_days"] == 7.0


def test_venue_form_only_counts_same_venue():
    # A의 홈 경기만 3번 있고 원정 경기가 섞여 있을 때, home_venue_form은 홈 경기만 집계해야 한다.
    rows = [
        ("2024-01-01", "A", "X", 2, 0),  # A 홈, 득실차 +2
        ("2024-01-08", "Y", "A", 0, 0),  # A 원정 (홈 집계에서 제외되어야 함)
        ("2024-01-15", "A", "X", 1, 0),  # A 홈, 득실차 +1
        ("2024-01-22", "A", "X", 0, 1),  # A 홈, 득실차 -1
        ("2024-01-29", "A", "X", 3, 0),  # 4번째 홈 경기 (직전 3개 홈 경기 합 계산용)
    ]
    matches = _matches(rows)
    out = add_rolling_features(matches, form_window=3)
    fourth_home_row = out.iloc[4]
    # 직전 3개 "홈" 경기(원정 경기 제외)의 득실차 합: (+2) + (+1) + (-1) = 2
    assert fourth_home_row["home_venue_form"] == 2.0
