# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from ssr_abtest.render.html import capture_screenshot, html_to_text, load_variant_text

ROOT = Path(__file__).parents[1]


def test_html_drops_script_and_style_and_keeps_visible_copy():
    text = html_to_text((ROOT / "examples/pages/lp_a.html").read_text(encoding="utf-8"))
    assert "今週末、電車で行ける2日間のルート" in text
    assert "console.log" not in text
    assert "font-family" not in text
    assert "このルートを見る" in text


def test_plain_text_and_truncation(tmp_path: Path):
    path = tmp_path / "page.txt"
    path.write_text("  hello   world  ", encoding="utf-8")
    text, truncated = load_variant_text(path, None, max_chars=100)
    assert text == "hello   world"
    assert truncated is False
    short, was = load_variant_text(None, "abcdef", max_chars=3)
    assert short.startswith("abc")
    assert was is True


def test_screenshot_explains_the_missing_extra(tmp_path: Path):
    def boom():
        raise ImportError("missing")

    page = tmp_path / "page.html"
    page.write_text("<html><body><p>Hi</p></body></html>", encoding="utf-8")
    import ssr_abtest.render.html as html

    original = html._load_playwright
    html._load_playwright = boom
    try:
        with pytest.raises(RuntimeError, match="screenshot"):
            capture_screenshot(page, tmp_path / "shot.png")
    finally:
        html._load_playwright = original
