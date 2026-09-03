import pytest

from ingest.fixtures import fetch_upcoming_fixtures, to_fdcouk_name


def test_to_fdcouk_name_maps_known_team():
    assert to_fdcouk_name("Manchester United FC") == "Man United"


def test_to_fdcouk_name_raises_for_unknown_team():
    with pytest.raises(KeyError):
        to_fdcouk_name("Some Unmapped FC")


def test_fetch_upcoming_fixtures_requires_api_key(monkeypatch):
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        fetch_upcoming_fixtures(api_key=None)
