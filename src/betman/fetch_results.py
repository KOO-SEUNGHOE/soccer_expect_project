"""배트맨 '경기정보 > 축구 > 일정/결과' 페이지에서 종료된 EPL 경기 스코어를 가져와
logs/predictions.db(우리 모델 예측)와 logs/betman_odds.db(배트맨 배당)의 미채점
경기를 채점한다.

왜 필요한가: 기존 채점(collect_results.py, score_odds.py)은 football-data.co.uk
CSV에만 의존하는데, 이 사이트가 라운드 종료 후 갱신이 하루~이틀 늦거나 접속이
아예 막히는 경우가 반복됐다. 배트맨 일정/결과 페이지는 경기가 끝나면 바로 스코어가
나와서 대체 소스로 쓸 수 있다 (2026-09-13 수동으로 검증).

두 DB는 날짜 기준이 다르다 — predictions.db는 football-data.org의 UTC 기준
날짜, betman_odds.db는 배트맨의 KST 기준 날짜다(예: KST 일요일 01:30 킥오프는
UTC로는 토요일). 그래서 팀 이름으로 매칭하되 날짜는 "KST 날짜 또는 그 전날"
범위에서 찾는다.

사용법:
    python src/betman/fetch_results.py
"""
from __future__ import annotations

import datetime as dt
import pathlib
import re
import sys
from collections.abc import Callable
from typing import NamedTuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from playwright.sync_api import Page, sync_playwright

from betman import db as betman_db
from pipeline import db as pred_db

SCHEDULE_URL = "https://www.betman.co.kr/main/mainPage/gameinfo/dataOfSoccerSchedule.do"
EPL_TAB_SELECTOR = "a[href='#tabs-3']"
EPL_PANEL_SELECTOR = "#tabs-3"

# 월초(1~이 일수)에는 지난달 말 경기가 현재 월 목록에서 빠져 있어 이전 달도 같이 본다.
PREVIOUS_MONTH_DAYS = 10

# 일정/결과 페이지는 팀 이름을 축약형으로 쓴다 (프로토 승부식 페이지의 정식 명칭과 다름).
# 매핑에 없는 팀은 그 경기만 건너뛰고 경고한다 — 승격/강등으로 팀이 바뀌면 추가할 것.
SHORT_KR_TO_FDCOUK_NAME: dict[str, str] = {
    "아스널": "Arsenal",
    "A빌라": "Aston Villa",
    "본머스": "Bournemouth",
    "브렌트퍼": "Brentford",
    "브라이턴": "Brighton",
    "첼시": "Chelsea",
    "코번트리": "Coventry",
    "크리스털": "Crystal Palace",
    "에버턴": "Everton",
    "풀럼": "Fulham",
    "헐시티": "Hull",
    "입스위치": "Ipswich",
    "리즈U": "Leeds",
    "리버풀": "Liverpool",
    "맨체스C": "Man City",
    "맨체스U": "Man United",
    "뉴캐슬U": "Newcastle",
    "노팅엄F": "Nott'm Forest",
    "선덜랜드": "Sunderland",
    "토트넘": "Tottenham",
    "웨스트햄": "West Ham",
    "울버햄": "Wolves",
    "번리": "Burnley",
}

_WHEN_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{2})")

_ROWS_JS = r"""() => Array.from(document.querySelectorAll('#tabs-3 tr[data-day]'))
    .filter(r => r.querySelectorAll('a.team').length === 2)
    .map(r => ({
        when: r.children[0].innerText.trim(),
        teams: Array.from(r.querySelectorAll('a.team')).map(a => a.innerText.trim()),
        scores: Array.from(r.querySelectorAll('strong.score')).map(s => s.innerText.trim()),
        badge: r.querySelector('.badge') ? r.querySelector('.badge').innerText.trim() : '',
    }))"""


class FinishedMatch(NamedTuple):
    kst_date: dt.date
    home_team: str
    away_team: str
    fthg: int
    ftag: int


def parse_kst_date(when: str) -> dt.date:
    """'26.10.10(토) 23:00' -> date(2026, 10, 10)."""
    m = _WHEN_RE.match(when.strip())
    if not m:
        raise ValueError(f"날짜 형식을 해석하지 못함: {when!r}")
    yy, mm, dd = (int(g) for g in m.groups())
    return dt.date(2000 + yy, mm, dd)


def parse_finished_matches(raw_rows: list[dict]) -> list[FinishedMatch]:
    """페이지에서 뽑은 행 목록에서 '종료'이고 스코어가 있는 EPL 경기만 골라낸다 (순수 함수).

    매핑에 없는 팀이 낀 경기는 stderr에 경고하고 건너뛴다.
    """
    matches = []
    for row in raw_rows:
        if row["badge"] != "종료" or len(row["scores"]) != 2 or "" in row["scores"]:
            continue
        home_kr, away_kr = row["teams"]
        unknown = {t for t in (home_kr, away_kr) if t not in SHORT_KR_TO_FDCOUK_NAME}
        if unknown:
            print(f"[betman results] 팀 이름 매핑이 없어 건너뜁니다: {unknown}", file=sys.stderr)
            continue
        matches.append(FinishedMatch(
            kst_date=parse_kst_date(row["when"]),
            home_team=SHORT_KR_TO_FDCOUK_NAME[home_kr],
            away_team=SHORT_KR_TO_FDCOUK_NAME[away_kr],
            fthg=int(row["scores"][0]),
            ftag=int(row["scores"][1]),
        ))
    return matches


def result_code(fthg: int, ftag: int) -> str:
    return "H" if fthg > ftag else "A" if fthg < ftag else "D"


def find_stored_date(unscored_rows: list, match: FinishedMatch) -> str | None:
    """미채점 행 중 같은 팀 조합이면서 날짜가 'KST 날짜 또는 그 전날'인 행의 저장된 날짜를 반환한다."""
    candidates = {match.kst_date.isoformat(), (match.kst_date - dt.timedelta(days=1)).isoformat()}
    for row in unscored_rows:
        if (
            row["home_team"] == match.home_team
            and row["away_team"] == match.away_team
            and row["match_date"] in candidates
        ):
            return row["match_date"]
    return None


def score_matches(conn, fetch_unscored: Callable, record_result: Callable, matches: list[FinishedMatch]) -> int:
    """종료된 경기들을 해당 DB의 미채점 행에 채점하고, 채점한 건수를 반환한다."""
    unscored = fetch_unscored(conn)
    scored_at = dt.datetime.now(dt.timezone.utc).isoformat()
    count = 0
    for m in matches:
        stored_date = find_stored_date(unscored, m)
        if stored_date is None:
            continue
        record_result(
            conn, stored_date, m.home_team, m.away_team, m.fthg, m.ftag, result_code(m.fthg, m.ftag), scored_at
        )
        count += 1
    return count


def scrape_finished_matches(page: Page, include_previous_month: bool) -> list[FinishedMatch]:
    page.goto(SCHEDULE_URL, wait_until="networkidle")
    page.click(EPL_TAB_SELECTOR)
    page.wait_for_load_state("networkidle")
    page.wait_for_selector(f"{EPL_PANEL_SELECTOR} tr[data-day]", timeout=15000)

    raw_rows = page.evaluate(_ROWS_JS)
    if include_previous_month:
        page.get_by_role("button", name="이전 달").click()
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)  # 월 전환 후 목록이 다시 그려질 시간
        raw_rows = page.evaluate(_ROWS_JS) + raw_rows
    return parse_finished_matches(raw_rows)


def main() -> None:
    today_kst = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9)).date()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        # 사이트가 킥오프 시각을 브라우저 타임존 기준으로 렌더링한다. UTC 서버(GitHub Actions)에서는
        # KST 00:30 경기가 전날 15:30으로 보여 날짜가 어긋나므로 항상 KST로 고정한다.
        context = browser.new_context(timezone_id="Asia/Seoul", locale="ko-KR")
        page = context.new_page()
        finished = scrape_finished_matches(page, include_previous_month=today_kst.day <= PREVIOUS_MONTH_DAYS)
        browser.close()

    pred_conn = pred_db.connect()
    bet_conn = betman_db.connect()
    n_pred = score_matches(pred_conn, pred_db.fetch_unscored, pred_db.record_result, finished)
    n_bet = score_matches(bet_conn, betman_db.fetch_unscored, betman_db.record_result, finished)
    print(
        f"[betman results] 종료 경기 {len(finished)}건 확인 -> "
        f"예측 {n_pred}건, 배트맨 배당 {n_bet}건 채점했습니다."
    )


if __name__ == "__main__":
    main()
