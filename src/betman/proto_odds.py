"""배트맨(공식 스포츠토토, betman.co.kr) '프로토 승부식'에서 EPL 경기의
실제 고정 배당(승/무/패)을 가져온다.

이 프로젝트의 메인 예측 파이프라인(football-data.co.uk 배당 기반)과는
완전히 별개의 섹션이다 — 대화 중 확인한 내용:
  - 배트맨 "승무패"(토토)는 배당이 아니라 투표율(%)만 제공해 이 프로젝트가
    쓰는 "배당 마진 제거 → 시장 확률" 개념과 성격이 다르다.
  - "프로토 승부식"은 실제 소수점 고정 배당(예: 1.90/3.30/3.40)을 쓰고
    실시간으로 오르내려서(배당률 상승/하락 표시) 유럽 북메이커와 같은
    성격의 데이터다. 이걸 쓴다.
  - 배당 데이터는 서버가 처음 주는 HTML에는 없고 페이지 로드 후 내부 API
    호출로 채워지므로, plain requests로는 못 읽고 Playwright로 실제 렌더링을
    거쳐야 한다.
  - GitHub Actions 클라우드 IP가 이런 사이트의 봇 차단에 걸리기 쉬워서
    자동화(주간 파이프라인)에는 포함하지 않고 로컬/수동 실행 전용으로 둔다.

사용법:
    playwright install chromium   # 최초 1회만
    python src/betman/proto_odds.py
"""
from __future__ import annotations

import datetime as dt
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from playwright.sync_api import Page, sync_playwright

from betman.db import connect, insert_odds

BASE_URL = "https://betman.co.kr"

# 배트맨이 쓰는 한글 팀 명칭 -> football-data.co.uk 축약 명칭.
# fixtures.py의 FDORG_TO_FDCOUK_NAME과 같은 이유로 시즌이 바뀌면 완전한
# 목록일 수 없다 — 매핑에 없는 팀은 건너뛰고 경고를 남긴다(전체 실패 방지).
# 2026-09-11 확인: 2026-27 시즌 승격팀(헐 시티/입스위치 타운/코번트리 시티)
# 포함 전체 20개 팀 매핑이 실제 화면 스크래핑으로 검증 완료됨(입스위치도
# 이번에 실전 매칭에 성공해 추정이 아니라 확정으로 바뀜).
KR_TO_FDCOUK_NAME: dict[str, str] = {
    "아스널": "Arsenal",
    "애스턴 빌라": "Aston Villa",
    "AFC본머스": "Bournemouth",
    "브렌트퍼드": "Brentford",
    "브라이턴&호브 앨비언": "Brighton",
    "첼시": "Chelsea",
    "코번트리 시티": "Coventry",
    "크리스털 팰리스": "Crystal Palace",
    "에버턴": "Everton",
    "풀럼": "Fulham",
    "헐 시티": "Hull",
    "입스위치 타운": "Ipswich",  # 미검증(추정)
    "리즈 유나이티드": "Leeds",
    "리버풀": "Liverpool",  # 미검증(추정)
    "맨체스터 시티": "Man City",
    "맨체스터 유나이티드": "Man United",
    "뉴캐슬 유나이티드": "Newcastle",
    "노팅엄 포리스트": "Nott'm Forest",
    "선덜랜드": "Sunderland",
    "토트넘 홋스퍼": "Tottenham",
}

_ODDS_VALUE_RE = re.compile(r"^\s*([\d.]+)")


def implied_probabilities_from_odds(odds_h: float, odds_d: float, odds_a: float) -> tuple[float, float, float]:
    """배당에서 마진을 제거한 확률을 계산한다 (build_features.implied_probabilities와
    동일한 방법: 1/odds를 구한 뒤 합이 1이 되도록 정규화). 데이터 소스가 완전히
    다른 별도 섹션이라 의존성을 만들지 않고 여기서 독립적으로 계산한다.
    """
    inv_h, inv_d, inv_a = 1 / odds_h, 1 / odds_d, 1 / odds_a
    overround = inv_h + inv_d + inv_a
    return inv_h / overround, inv_d / overround, inv_a / overround


def _parse_odds_text(text: str) -> float:
    """'1.65배당률 상승' 같은 텍스트에서 숫자 배당만 뽑아낸다."""
    m = _ODDS_VALUE_RE.match(text)
    if not m:
        raise ValueError(f"배당 텍스트를 해석하지 못함: {text!r}")
    return float(m.group(1))


def find_current_proto_round(page: Page) -> tuple[str, str, str]:
    """홈 화면에서 현재 판매 중인 '프로토 승부식' 링크를 찾아 (gmId, year, gmTs)를 반환한다.

    실제 링크는 `javascript:leftA.checkLeftSlipData(0,'/main/.../gameSlip.do?
    frameType=typeA&gmId=G101&gmTs=260105')`처럼 gmId/gmTs만 있고 year는 없다
    (gmTs 앞 2자리가 연도를 뜻함, 예: "26" -> 2026). gameSlip.do 페이지 자체는
    year 파라미터 없이도 정상 동작하는 것으로 확인함.
    """
    page.goto(BASE_URL, wait_until="networkidle")
    href = page.eval_on_selector(
        "a[href*='gameSlip.do'][href*='gmId=G101']",
        "el => el.getAttribute('href')",
    )
    if not href:
        raise RuntimeError("프로토 승부식 링크를 찾지 못했습니다 (홈 화면 구조가 바뀌었을 수 있음).")

    gm_id_match = re.search(r"gmId=(\w+)", href)
    gm_ts_match = re.search(r"gmTs=(\d+)", href)
    if not gm_id_match or not gm_ts_match:
        raise RuntimeError(f"프로토 승부식 링크 형식을 해석하지 못했습니다: {href}")

    gm_id, gm_ts = gm_id_match.group(1), gm_ts_match.group(1)
    year = f"20{gm_ts[:2]}"
    return gm_id, year, gm_ts


def scrape_epl_odds(page: Page) -> list[dict]:
    """현재 판매 중인 프로토 승부식 회차에서 EPL '축구 승무패' 배당을 모두 가져온다.

    매핑에 없는 팀이 낀 경기, 또는 승무패 마켓 자체가 없는 경기(핸디캡/
    언더오버만 있는 경우)는 건너뛴다.
    """
    gm_id, year, gm_ts = find_current_proto_round(page)
    page.goto(
        f"{BASE_URL}/main/mainPage/gamebuy/gameSlip.do?gmId={gm_id}&year={year}&gmTs={gm_ts}",
        wait_until="networkidle",
    )
    page.wait_for_selector("button.btn-more[data-leaguename]", timeout=15000)

    scraped_at = dt.datetime.now(dt.timezone.utc).isoformat()
    records = []
    for btn in page.query_selector_all("button.btn-more[data-leaguename='EPL']"):
        gamekey = btn.get_attribute("data-gamekey")
        home_kr = btn.get_attribute("data-homename")
        away_kr = btn.get_attribute("data-awayname")
        relmchdate = btn.get_attribute("data-relmchdate")  # 예: "20260905230000"

        unknown = {t for t in (home_kr, away_kr) if t not in KR_TO_FDCOUK_NAME}
        if unknown:
            print(f"[betman] 팀 이름 매핑이 없어 건너뜁니다: {unknown}", file=sys.stderr)
            continue

        win_row = page.query_selector(f"li[data-matchseq='{gamekey}']:has(b.game:text-is('축구 승무패'))")
        if win_row is None:
            continue  # 이 경기는 승무패 마켓이 없음

        odds_cells = win_row.query_selector_all(".btnChk .db")
        if len(odds_cells) != 3:
            print(f"[betman] 배당 3개를 못 찾아 건너뜁니다: {home_kr} vs {away_kr}", file=sys.stderr)
            continue
        odds_h, odds_d, odds_a = (_parse_odds_text(c.inner_text()) for c in odds_cells)
        implied_h, implied_d, implied_a = implied_probabilities_from_odds(odds_h, odds_d, odds_a)

        kickoff = dt.datetime.strptime(relmchdate, "%Y%m%d%H%M%S")  # noqa: DTZ007 (KST 로컬시각, 우리 데이터의 naive Date와 맞춤)
        records.append({
            "match_date": kickoff.date().isoformat(),
            "kickoff": kickoff.isoformat(),
            "home_team": KR_TO_FDCOUK_NAME[home_kr],
            "away_team": KR_TO_FDCOUK_NAME[away_kr],
            "odds_h": odds_h, "odds_d": odds_d, "odds_a": odds_a,
            "implied_h": implied_h, "implied_d": implied_d, "implied_a": implied_a,
            "scraped_at": scraped_at,
        })
    return records


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        records = scrape_epl_odds(page)
        browser.close()

    conn = connect()
    for record in records:
        insert_odds(conn, record)

    print(f"[betman] EPL 경기 {len(records)}건의 프로토 배당을 저장했습니다.")


if __name__ == "__main__":
    main()
