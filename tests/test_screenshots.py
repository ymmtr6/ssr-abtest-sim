# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from ssr_abtest.config import load_experiment
from ssr_abtest.render.images import read_image_size, write_placeholder_png
from ssr_abtest.render.screenshots import resolve_images

ROOT = Path(__file__).parents[1]


class _Page:
    def __init__(self, seen: dict):
        self.seen = seen

    def goto(self, url: str, wait_until: str = "load") -> None:
        self.seen["url"] = url
        self.seen.setdefault("urls", []).append(url)
        self.seen["wait_until"] = wait_until

    def screenshot(self, path: str, full_page: bool) -> None:
        self.seen["full_page"] = full_page
        write_placeholder_png(Path(path))

    def content(self) -> str:
        return "<html><body><h1>Fetched page</h1><script>nope()</script></body></html>"


class _Browser:
    def __init__(self, seen: dict):
        self.seen = seen

    def new_page(self, viewport: dict) -> _Page:
        self.seen["viewport"] = viewport
        return _Page(self.seen)

    def close(self) -> None:
        self.seen["closed"] = True


class _Playwright:
    def __init__(self, seen: dict):
        self.seen = seen
        self.chromium = self

    def launch(self) -> _Browser:
        self.seen["launched"] = self.seen.get("launched", 0) + 1
        return _Browser(self.seen)

    def __enter__(self) -> "_Playwright":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _factory(seen: dict):
    def sync_playwright():
        return _Playwright(seen)

    def load():
        return sync_playwright

    return load


def test_png_round_trip_and_jpeg_header(tmp_path: Path):
    png = tmp_path / "one.png"
    write_placeholder_png(png)
    assert read_image_size(png) == (1, 1)
    jpeg = tmp_path / "tiny.jpg"
    jpeg.write_bytes(bytes.fromhex("ffd8ffc0000b080001000101000000ffd9"))
    assert read_image_size(jpeg) == (1, 1)


def test_capture_records_viewport_and_reuses_the_cache(tmp_path: Path):
    config = load_experiment(ROOT / "configs/experiments/example_lp.yaml")
    config.screenshot.viewport = "mobile"
    config.screenshot.width = 390
    config.screenshot.height = 844
    config.screenshot.capture = "full_page"
    seen: dict = {}
    cache = tmp_path / "cache"
    output = tmp_path / "out"
    images, _text = resolve_images(
        config,
        capture=True,
        output_dir=output,
        cache_dir=cache,
        playwright_factory=_factory(seen),
    )
    assert seen["viewport"] == {"width": 390, "height": 844}
    assert seen["full_page"] is True
    assert seen["url"].startswith("file:")
    assert images["A"].width == 1
    assert images["A"].placeholder is False
    assert images["A"].path is not None and images["A"].path.is_file()
    resolve_images(
        config,
        capture=True,
        output_dir=output,
        cache_dir=cache,
        playwright_factory=_factory(seen),
    )
    assert seen["launched"] == 2


def test_text_mode_skips_images_and_a_url_capture_keeps_visible_text(tmp_path: Path):
    config = load_experiment(ROOT / "configs/experiments/example_lp.yaml")
    config.input_mode = "text"
    images, extra = resolve_images(config, capture=False, output_dir=None, cache_dir=None)
    assert images == {} and extra == {}

    config.input_mode = "both"
    config.variants["A"].kind = "url"
    config.variants["A"].url = "https://example.com/a"
    config.variants["A"].path = None
    seen: dict = {}
    _images, text = resolve_images(
        config,
        capture=True,
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        playwright_factory=_factory(seen),
    )
    assert text["A"] == "Fetched page"
    assert "nope" not in text["A"]
    assert "https://example.com/a" in seen["urls"]


def test_dry_run_plans_the_viewport_without_a_browser(tmp_path: Path):
    config = load_experiment(ROOT / "configs/experiments/example_lp.yaml")

    def boom():
        raise AssertionError("dry-run launched a browser")

    images, _text = resolve_images(
        config,
        capture=False,
        output_dir=tmp_path,
        cache_dir=None,
        playwright_factory=boom,
    )
    assert images["A"].placeholder is True
    assert (images["A"].width, images["A"].height) == (1280, 800)
    assert images["A"].tokens == 1200
    assert images["A"].path is not None and read_image_size(images["A"].path) == (1, 1)
