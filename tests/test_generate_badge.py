from evaluate.generate_badge import (
    BADGE_END,
    BADGE_START,
    compute_stats,
    render_badge_markdown,
    update_readme,
)


def test_compute_stats_empty_returns_none():
    assert compute_stats([]) is None


def test_compute_stats_computes_brier():
    rows = [
        {"model_h": 1.0, "model_d": 0.0, "model_a": 0.0, "actual_result": "H"},  # 완벽한 예측
    ]
    stats = compute_stats(rows)
    assert stats == {"n": 1, "brier": 0.0}


def test_render_badge_markdown_handles_no_data():
    md = render_badge_markdown(None)
    assert md.startswith(BADGE_START)
    assert md.endswith(BADGE_END)


def test_render_badge_markdown_includes_brier_value():
    md = render_badge_markdown({"n": 42, "brier": 0.512})
    assert "0.512" in md.replace("%2E", ".")  # URL 인코딩 여부와 무관하게 값이 들어갔는지 확인
    assert "img.shields.io" in md


def test_update_readme_replaces_only_marker_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        f"# title\n\n{BADGE_START}\nold\n{BADGE_END}\n\n다른 내용은 그대로.\n",
        encoding="utf-8",
    )
    update_readme(f"{BADGE_START}\nnew badge\n{BADGE_END}", readme_path=readme)

    text = readme.read_text(encoding="utf-8")
    assert "new badge" in text
    assert "old" not in text
    assert "다른 내용은 그대로." in text
