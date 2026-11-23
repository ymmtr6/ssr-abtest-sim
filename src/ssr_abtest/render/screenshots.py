# SPDX-License-Identifier: Apache-2.0
"""Resolve one screenshot per variant, using a file cache for real captures."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from ssr_abtest.config import ExperimentConfig, ScreenshotSettings, Variant
from ssr_abtest.render.html import capture_screenshot, html_to_text
from ssr_abtest.render.image_tokens import ImageTokenError, image_input_tokens
from ssr_abtest.render.images import read_image_size, write_placeholder_png

_URL_PLACEHOLDER = {
    "ja": "ドライランではこのURLの本文を取得していません。添付画像は寸法の見積もり用です。",
    "en": (
        "Dry-run did not fetch this URL. The attached image is a size placeholder "
        "for the cost estimate."
    ),
}


@dataclass
class VariantImage:
    variant_id: str
    path: Path | None
    width: int
    height: int
    detail: str
    capture: str
    viewport: str
    placeholder: bool
    assumed_dimensions: bool
    tokens: int | None
    token_error: str | None
    source: str


def planned_dimensions(
    settings: ScreenshotSettings,
    *,
    source: str,
    image_path: Path | None,
) -> tuple[int, int, bool]:
    """Return width, height, and whether the height is an assumption.

    An existing image file uses its real pixel size. A fold capture uses
    the viewport. A full-page capture of HTML or a URL is not known until
    a browser runs, so the planner uses ``full_page_height_factor``.
    """
    if source == "image":
        if image_path is None:
            raise ValueError("an image variant needs a file")
        width, height = read_image_size(image_path)
        return width, height, False
    if settings.capture == "fold":
        return settings.width, settings.height, False
    height = max(settings.height, int(round(settings.height * settings.full_page_height_factor)))
    return settings.width, height, True


def _fingerprint(variant: Variant, settings: ScreenshotSettings) -> str:
    if variant.kind == "url":
        identity = f"url:{variant.url}"
    elif variant.kind == "image" and variant.image is not None:
        stat = variant.image.stat()
        identity = f"image:{variant.image.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    elif variant.path is not None:
        stat = variant.path.stat()
        identity = f"file:{variant.path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    else:
        identity = f"text:{variant.id}"
    raw = f"{identity}|{settings.width}|{settings.height}|{settings.capture}|{settings.detail}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _price(model: str, width: int, height: int, detail: str) -> tuple[int | None, str | None]:
    try:
        return image_input_tokens(model, width, height, detail), None
    except ImageTokenError as exc:
        return None, str(exc)


def resolve_images(
    config: ExperimentConfig,
    *,
    capture: bool,
    output_dir: Path | None,
    cache_dir: Path | None,
    playwright_factory=None,
) -> tuple[dict[str, VariantImage], dict[str, str]]:
    """Return images by variant id, and extra text extracted from URLs.

    ``capture`` False never launches a browser. HTML and URL variants then
    use the planned pixel size. A live run caches the PNG under
    ``cache_dir`` and copies it into ``output_dir``.
    """
    if config.input_mode == "text":
        return {}, {}
    images: dict[str, VariantImage] = {}
    extra_text: dict[str, str] = {}
    for variant in config.variants.values():
        if variant.kind == "text":
            raise ValueError(
                f"variant {variant.id} is plain text, which cannot be screenshotted. "
                "Use input_mode text, or point the variant at HTML, a URL, or an image."
            )
        source_path = variant.image if variant.kind == "image" else variant.path
        width, height, assumed = planned_dimensions(
            config.screenshot,
            source=variant.kind,
            image_path=source_path if variant.kind == "image" else None,
        )
        placeholder = not capture
        written: Path | None = None
        if variant.kind == "image" and variant.image is not None:
            placeholder = False
            assumed = False
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                written = output_dir / f"{variant.id}{variant.image.suffix.lower()}"
                if not written.exists() or written.stat().st_mtime < variant.image.stat().st_mtime:
                    shutil.copyfile(variant.image, written)
            else:
                written = variant.image
        elif capture:
            if variant.kind not in {"html", "url"}:
                raise ValueError(f"variant {variant.id} cannot be captured")
            target: str | Path = variant.url if variant.kind == "url" and variant.url else variant.path  # type: ignore[assignment]
            if target is None:
                raise ValueError(f"variant {variant.id} has nothing to capture")
            cache_path = _cache_path(cache_dir, variant, config.screenshot)
            text_sidecar = None if cache_path is None else cache_path.with_suffix(".txt")
            cached_text = None
            if (
                cache_path is not None
                and cache_path.is_file()
                and (variant.kind != "url" or config.input_mode != "both" or (
                    text_sidecar is not None and text_sidecar.is_file()
                ))
            ):
                shot = cache_path
                if text_sidecar is not None and text_sidecar.is_file():
                    cached_text = text_sidecar.read_text(encoding="utf-8")
            else:
                shot = cache_path or (output_dir or Path(".")) / f"{variant.id}.png"
                page_html = capture_screenshot(
                    target,
                    shot,
                    width=config.screenshot.width,
                    height=config.screenshot.height,
                    full_page=config.screenshot.capture == "full_page",
                    playwright_factory=playwright_factory,
                )
                if page_html and variant.kind == "url":
                    cached_text = html_to_text(page_html)
                    if text_sidecar is not None and cached_text:
                        text_sidecar.parent.mkdir(parents=True, exist_ok=True)
                        text_sidecar.write_text(cached_text, encoding="utf-8")
            measured_w, measured_h = read_image_size(shot)
            width, height, assumed = measured_w, measured_h, False
            placeholder = False
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                written = output_dir / f"{variant.id}.png"
                if written.resolve() != shot.resolve():
                    shutil.copyfile(shot, written)
            else:
                written = shot
            if cached_text:
                extra_text[variant.id] = cached_text
        elif output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            written = output_dir / f"{variant.id}.png"
            write_placeholder_png(written)
        if variant.kind == "url" and variant.id not in extra_text and not capture:
            extra_text[variant.id] = _URL_PLACEHOLDER.get(config.language, _URL_PLACEHOLDER["en"])
        tokens, error = _price(config.models.generate, width, height, config.screenshot.detail)
        images[variant.id] = VariantImage(
            variant_id=variant.id,
            path=written,
            width=width,
            height=height,
            detail=config.screenshot.detail,
            capture=config.screenshot.capture,
            viewport=config.screenshot.viewport,
            placeholder=placeholder,
            assumed_dimensions=assumed,
            tokens=tokens,
            token_error=error,
            source=variant.kind,
        )
    return images, extra_text


def _cache_path(cache_dir: Path | None, variant: Variant, settings: ScreenshotSettings) -> Path | None:
    if cache_dir is None:
        return None
    digest = _fingerprint(variant, settings)
    return cache_dir / digest[:2] / f"{digest}.png"
