"""logs/predictions.db에 누적된 실전 예측 채점 결과로 README의 성능 배지를 갱신한다.

값을 사람이 직접 README에 써넣지 않고(하드코딩 금지, CLAUDE.md 참고) 이 스크립트가
DB에서 계산한 값으로 README의 <!-- BADGE:START -->...<!-- BADGE:END --> 사이
블록을 덮어쓰는 방식으로 자동 갱신한다.

사용법:
    python src/evaluate/generate_badge.py
"""
from __future__ import annotations

import pathlib
import sys
import urllib.parse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from evaluate.backtest import brier_score
from pipeline.db import connect, fetch_scored

README_PATH = pathlib.Path(__file__).resolve().parents[2] / "README.md"
BADGE_START = "<!-- BADGE:START -->"
BADGE_END = "<!-- BADGE:END -->"


def compute_stats(scored_rows: list) -> dict | None:
    """채점 완료된 예측들로 실전 누적 Brier score를 계산한다 (순수 함수)."""
    if not scored_rows:
        return None
    probs = [{"H": r["model_h"], "D": r["model_d"], "A": r["model_a"]} for r in scored_rows]
    outcomes = [r["actual_result"] for r in scored_rows]
    return {"n": len(scored_rows), "brier": brier_score(probs, outcomes)}


def render_badge_markdown(stats: dict | None) -> str:
    if stats is None:
        return f"{BADGE_START}\n_(아직 실전 예측 채점 결과가 없습니다 — 주간 파이프라인이 몇 라운드 돌아간 뒤 표시됩니다)_\n{BADGE_END}"

    label = urllib.parse.quote(f"실전 예측 {stats['n']}경기 Brier")
    value = urllib.parse.quote(f"{stats['brier']:.3f}")
    badge_url = f"https://img.shields.io/badge/{label}-{value}-blue"
    return f"{BADGE_START}\n![실전 예측 성능]({badge_url})\n{BADGE_END}"


def update_readme(badge_markdown: str, readme_path: pathlib.Path = README_PATH) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if BADGE_START not in text or BADGE_END not in text:
        raise RuntimeError(f"README에 {BADGE_START}/{BADGE_END} 마커가 없습니다. 먼저 추가하세요.")
    start = text.index(BADGE_START)
    end = text.index(BADGE_END) + len(BADGE_END)
    readme_path.write_text(text[:start] + badge_markdown + text[end:], encoding="utf-8")


def main() -> None:
    conn = connect()
    stats = compute_stats(fetch_scored(conn))
    update_readme(render_badge_markdown(stats))
    print("[generate_badge]", stats)


if __name__ == "__main__":
    main()
