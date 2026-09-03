"""football-data.org API에서 EPL 예정 경기 일정을 가져온다 (무료 티어).

API 키가 필요하다: https://www.football-data.org/client/register 에서 무료로
발급받아 환경변수 FOOTBALL_DATA_API_KEY로 전달한다 (GitHub Actions에서는
secrets.FOOTBALL_DATA_API_KEY로 주입).

주의(검증 필요): football-data.org는 팀 이름을 "Manchester United FC"처럼
정식 명칭으로 주는 반면, football-data.co.uk(과거 결과/배당 데이터)는
"Man United"처럼 축약 표기를 쓴다. 아래 FDORG_TO_FDCOUK_NAME은 공개적으로
알려진 명칭 차이를 바탕으로 미리 작성한 매핑이며, 실제 API 키로 응답을 받아본 뒤
한 번 검증이 필요하다 (이 개발 환경에는 API 키가 없어 실제 호출로 확인하지 못했음).
매핑에 없는 팀 이름은 조용히 넘어가지 않고 예외를 던진다 — 이름이 어긋난 채로
엉뚱한 팀에 대한 예측이 만들어지는 것을 막기 위함이다.
"""
from __future__ import annotations

import os

import requests

API_URL = "https://api.football-data.org/v4/competitions/PL/matches"

FDORG_TO_FDCOUK_NAME: dict[str, str] = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Leeds United FC": "Leeds",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
}


def to_fdcouk_name(fdorg_name: str) -> str:
    """football-data.org 팀 명칭을 football-data.co.uk 축약 명칭으로 변환한다."""
    if fdorg_name not in FDORG_TO_FDCOUK_NAME:
        raise KeyError(
            f"알 수 없는 팀 이름: {fdorg_name!r}. "
            "FDORG_TO_FDCOUK_NAME 매핑에 추가해야 합니다 (src/ingest/fixtures.py)."
        )
    return FDORG_TO_FDCOUK_NAME[fdorg_name]


def fetch_upcoming_fixtures(api_key: str | None = None, status: str = "SCHEDULED") -> list[dict]:
    """예정된 EPL 경기 목록을 [{"date": "YYYY-MM-DD", "home_team": ..., "away_team": ...}] 형태로 반환한다.

    팀 이름은 to_fdcouk_name으로 football-data.co.uk 표기에 맞춰 변환된 상태로 나온다.
    """
    api_key = api_key or os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FOOTBALL_DATA_API_KEY가 설정되지 않았습니다. "
            "https://www.football-data.org/client/register 에서 무료 키를 발급받아 "
            "환경변수(로컬 실행) 또는 GitHub Actions secret(자동화)으로 설정하세요."
        )

    resp = requests.get(
        API_URL,
        headers={"X-Auth-Token": api_key},
        params={"status": status},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    fixtures = []
    for match in data.get("matches", []):
        fixtures.append({
            "date": match["utcDate"][:10],
            "home_team": to_fdcouk_name(match["homeTeam"]["name"]),
            "away_team": to_fdcouk_name(match["awayTeam"]["name"]),
        })
    return fixtures
