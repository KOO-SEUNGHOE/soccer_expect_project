import pytest

from ingest.fixtures import fetch_upcoming_fixtures, map_matches_json, to_fdcouk_name


def test_to_fdcouk_name_maps_known_team():
    assert to_fdcouk_name("Manchester United FC") == "Man United"
    # 아래 세 팀은 2026-27 시즌 승격팀 — 실제 GitHub Actions 실행에서 하나씩 발견됨
    assert to_fdcouk_name("Ipswich Town FC") == "Ipswich"
    assert to_fdcouk_name("Coventry City FC") == "Coventry"
    assert to_fdcouk_name("Hull City AFC") == "Hull"


def test_to_fdcouk_name_raises_for_unknown_team():
    with pytest.raises(KeyError):
        to_fdcouk_name("Some Unmapped FC")


def test_fetch_upcoming_fixtures_requires_api_key(monkeypatch):
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        fetch_upcoming_fixtures(api_key=None)


def _match(home: str, away: str, date: str = "2026-09-10T14:00:00Z") -> dict:
    return {"utcDate": date, "homeTeam": {"name": home}, "awayTeam": {"name": away}}


def test_map_matches_json_maps_known_teams():
    fixtures = map_matches_json([_match("Arsenal FC", "Chelsea FC")])
    assert fixtures == [{"date": "2026-09-10", "home_team": "Arsenal", "away_team": "Chelsea"}]


def test_map_matches_json_skips_fixture_with_unmapped_team():
    # 매핑에 없는 팀이 낀 경기 하나가 나머지 정상 경기까지 막아서는 안 된다.
    matches = [
        _match("Some New FC", "Chelsea FC"),
        _match("Arsenal FC", "Chelsea FC"),
    ]
    fixtures = map_matches_json(matches)
    assert len(fixtures) == 1
    assert fixtures[0]["home_team"] == "Arsenal"
