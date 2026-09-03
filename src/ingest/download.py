"""football-data.co.uk에서 EPL(E0) 시즌별 CSV를 내려받아 data/raw/에 저장한다.

사용법:
    python src/ingest/download.py --seasons 2223 2324 2425 2526

football-data.co.uk URL 패턴: https://www.football-data.co.uk/mmz4281/{season}/E0.csv
  - season은 "2324"처럼 두 시즌 연도를 이어붙인 4자리 문자열 (2023-24 시즌 -> "2324")
"""
import argparse
import pathlib
import sys
import time

import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
RAW_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"


def download_season(season: str, out_dir: pathlib.Path = RAW_DIR, sleep_sec: float = 1.0) -> pathlib.Path:
    """한 시즌의 EPL CSV를 내려받아 저장하고 저장 경로를 반환한다."""
    url = BASE_URL.format(season=season)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"E0_{season}.csv"

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)

    time.sleep(sleep_sec)  # 서버에 부담 주지 않도록 매 요청 사이 딜레이
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="football-data.co.uk EPL 시즌 데이터 다운로드")
    parser.add_argument(
        "--seasons",
        nargs="+",
        required=True,
        help="예: 2223 2324 2425 2526 (2023-24 시즌 -> 2324)",
    )
    args = parser.parse_args()

    for season in args.seasons:
        try:
            path = download_season(season)
            print(f"[OK] {season} -> {path}")
        except requests.HTTPError as e:
            print(f"[FAIL] {season}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
