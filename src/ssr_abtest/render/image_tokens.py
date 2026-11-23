# SPDX-License-Identifier: Apache-2.0
"""Billable image-input tokens for the Responses API.

Rules are taken from the OpenAI vision guide, checked on 2026-11-23:
https://developers.openai.com/api/docs/guides/images-vision

Patch models cover the image with 32×32 patches. The image is first fit
inside the detail level's maximum side, then scaled down again when a
patch budget applies. Billable tokens are ``ceil(patches * multiplier)``.
The guide's worked examples for ``gpt-6-astra`` at ``detail: high`` are:

* 1024×1024 → 1229 tokens
* 2048×2048 → 3000 tokens
* 4096×512 → 2458 tokens

``gpt-5.4``, ``gpt-5.4-mini``, and ``gpt-5.4-nano`` use the same 1.2
multiplier. On that family ``auto`` follows ``high``. ``low`` allows a
larger patch budget than ``high``, so it can cost more.

Tile models (``gpt-4o``, ``gpt-4.1``, ``gpt-4o-mini``, ``gpt-5.1``) use a
base token count. ``low`` is that base alone. ``high`` and ``auto`` scale
into a 2048 square, shorten the short side to 768 when needed, then add
512px tiles.

A count can differ from the API by one token because of rounding. More
than 30,000 patches after resizing is rejected rather than scaled further.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PATCH = 32
MAX_PATCHES = 30_000

_PATCH_DETAILS = dict[str, tuple[int, int | None]]


@dataclass(frozen=True)
class PatchProfile:
    multiplier: float
    details: _PATCH_DETAILS


@dataclass(frozen=True)
class TileProfile:
    base_tokens: int
    tile_tokens: int


def _patch(multiplier: float, **details: tuple[int, int | None]) -> PatchProfile:
    return PatchProfile(multiplier=multiplier, details=dict(details))


# detail -> (max side, patch budget or None when the image is not resized to a budget)
_PROFILES: dict[str, PatchProfile | TileProfile] = {
    "gpt-6-astra": _patch(
        1.2,
        low=(512, None),
        high=(65_535, 2_500),
        original=(65_535, None),
        auto=(65_535, None),
    ),
    "gpt-5.6-sol": _patch(
        1.2,
        low=(512, None),
        high=(2048, 2_500),
        original=(65_535, None),
        auto=(65_535, None),
    ),
    "gpt-5.6-terra": _patch(
        1.2,
        low=(512, None),
        high=(2048, 2_500),
        original=(65_535, None),
        auto=(65_535, None),
    ),
    "gpt-5.6-luna": _patch(
        1.2,
        low=(512, None),
        high=(2048, 2_500),
        original=(65_535, None),
        auto=(65_535, None),
    ),
    "gpt-5.5": _patch(
        1.2,
        low=(512, None),
        high=(2048, 2_500),
        original=(6000, 10_000),
        auto=(6000, 10_000),
    ),
    "gpt-5.4-mini": _patch(
        1.2,
        low=(2048, 6_144),
        high=(2048, 2_500),
        original=(6000, 10_000),
        auto=(2048, 2_500),
    ),
    "gpt-5.4-nano": _patch(
        1.2,
        low=(2048, 6_144),
        high=(2048, 2_500),
        original=(6000, 10_000),
        auto=(2048, 2_500),
    ),
    "gpt-5.4": _patch(
        1.2,
        low=(2048, 6_144),
        high=(2048, 2_500),
        original=(6000, 10_000),
        auto=(2048, 2_500),
    ),
    "gpt-5.2": _patch(
        1.2,
        low=(2048, 6_144),
        high=(2048, 6_144),
        auto=(2048, 6_144),
    ),
    "gpt-4.1-mini": _patch(
        1.62,
        low=(2048, 6_144),
        high=(2048, 6_144),
        auto=(2048, 6_144),
    ),
    "gpt-5.1": TileProfile(70, 140),
    "gpt-4.1": TileProfile(85, 170),
    "gpt-4o-mini": TileProfile(2833, 5667),
    "gpt-4o": TileProfile(85, 170),
}


class ImageTokenError(ValueError):
    """The image cannot be priced, or the API would reject it."""


def profile_for(model: str) -> PatchProfile | TileProfile | None:
    """Match a model id or a dated snapshot such as ``gpt-5.4-mini-2026-03-17``."""
    name = model.strip().lower()
    for key in sorted(_PROFILES, key=len, reverse=True):
        if name == key or name.startswith(key + "-"):
            return _PROFILES[key]
    return None


def _fit_max_side(width: int, height: int, max_side: int) -> tuple[int, int]:
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    scale = max_side / longest
    return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))


def _patch_count(width: int, height: int) -> int:
    return math.ceil(width / PATCH) * math.ceil(height / PATCH)


def _fit_patch_budget(width: int, height: int, budget: int) -> tuple[int, int]:
    if _patch_count(width, height) <= budget:
        return width, height
    shrink = math.sqrt((PATCH * PATCH * budget) / (width * height))

    def edge_ratio(side: int) -> float:
        scaled = side * shrink
        return math.floor(scaled / PATCH) / (scaled / PATCH)

    adjusted = shrink * min(edge_ratio(width), edge_ratio(height))
    return max(1, math.floor(width * adjusted)), max(1, math.floor(height * adjusted))


def _patch_tokens(width: int, height: int, profile: PatchProfile, detail: str) -> int:
    if detail not in profile.details:
        allowed = ", ".join(sorted(profile.details))
        raise ImageTokenError(f"detail {detail!r} is not supported for this model ({allowed})")
    max_side, budget = profile.details[detail]
    fitted_w, fitted_h = _fit_max_side(width, height, max_side)
    if budget is not None:
        fitted_w, fitted_h = _fit_patch_budget(fitted_w, fitted_h, budget)
    patches = _patch_count(fitted_w, fitted_h)
    if patches > MAX_PATCHES:
        raise ImageTokenError(
            f"image needs {patches} patches after resizing, above the {MAX_PATCHES} limit"
        )
    return math.ceil(patches * profile.multiplier)


def _tile_tokens(width: int, height: int, profile: TileProfile, detail: str) -> int:
    if detail == "original":
        raise ImageTokenError("this model does not support detail 'original'")
    if detail not in {"low", "high", "auto"}:
        raise ImageTokenError(f"detail {detail!r} is not supported for this model")
    if detail == "low":
        return profile.base_tokens
    fitted_w, fitted_h = _fit_max_side(width, height, 2048)
    short, long = sorted((fitted_w, fitted_h))
    if short > 768:
        scale = 768 / short
        short = 768
        long = max(1, math.floor(long * scale))
    if fitted_w <= fitted_h:
        fitted_w, fitted_h = short, long
    else:
        fitted_w, fitted_h = long, short
    tiles = math.ceil(fitted_w / 512) * math.ceil(fitted_h / 512)
    return profile.base_tokens + tiles * profile.tile_tokens


def image_input_tokens(model: str, width: int, height: int, detail: str = "high") -> int:
    """Return billable input tokens for one image.

    ``detail`` is the Responses API value: ``low``, ``high``, ``auto``, or
    ``original``. Unknown models raise ``ImageTokenError``.
    """
    if width < 1 or height < 1:
        raise ImageTokenError("image dimensions must be positive")
    profile = profile_for(model)
    if profile is None:
        raise ImageTokenError(
            f"no published image-token rule for model {model!r}. "
            "The estimate omits image tokens until a rule is added."
        )
    if isinstance(profile, PatchProfile):
        return _patch_tokens(width, height, profile, detail)
    return _tile_tokens(width, height, profile, detail)
