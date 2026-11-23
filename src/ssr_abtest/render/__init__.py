# SPDX-License-Identifier: Apache-2.0
"""Turn HTML or plain-text variants into prompt text."""

from ssr_abtest.render.html import capture_screenshot, html_to_text, load_variant_text
from ssr_abtest.render.image_tokens import image_input_tokens
from ssr_abtest.render.screenshots import resolve_images

__all__ = [
    "capture_screenshot",
    "html_to_text",
    "image_input_tokens",
    "load_variant_text",
    "resolve_images",
]
