# SPDX-License-Identifier: Apache-2.0
"""HTML to visible text, plus an optional screenshot capture."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path


class _VisibleText(HTMLParser):
    _SKIP = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
            return
        if lowered in {"p", "div", "section", "article", "h1", "h2", "h3", "li", "br", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self.parts.append(text)
            self.parts.append(" ")


def html_to_text(html: str) -> str:
    parser = _VisibleText()
    parser.feed(html)
    parser.close()
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def load_variant_text(path: Path | None, inline: str | None, *, max_chars: int) -> tuple[str, bool]:
    """Return visible text and whether it was truncated."""
    if inline is not None:
        raw = inline
    elif path is not None:
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".html", ".htm"}:
            raw = html_to_text(raw)
    else:
        raise ValueError("a variant needs a path or inline text")
    stripped = raw.strip()
    if max_chars > 0 and len(stripped) > max_chars:
        return stripped[:max_chars].rstrip() + "\n[page text truncated]", True
    return stripped, False


def _load_playwright():
    from playwright.sync_api import sync_playwright

    return sync_playwright


def capture_screenshot(html_path: Path, dest: Path) -> Path:
    """Render a local HTML file. Requires the optional screenshot extra.

    The image is stored for inspection. This version does not send it to
    the model; the prompt uses the extracted text.
    """
    try:
        sync_playwright = _load_playwright()
    except ImportError as exc:
        raise RuntimeError(
            "Screenshot rendering needs the optional extra: "
            'pip install "ssr-abtest-sim[screenshot]" && playwright install chromium'
        ) from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(html_path.resolve().as_uri())
            page.screenshot(path=str(dest), full_page=True)
        finally:
            browser.close()
    return dest
