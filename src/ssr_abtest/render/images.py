# SPDX-License-Identifier: Apache-2.0
"""Read image dimensions and write a tiny PNG placeholder.

Dry-run does not launch a browser. A placeholder file marks where the
screenshot will be, and the cost estimate uses the planned pixel size.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_SUFFIXES


def read_image_size(path: Path) -> tuple[int, int]:
    """Return width and height for PNG, JPEG, or WebP."""
    data = path.read_bytes()
    if data.startswith(_PNG_SIGNATURE) and len(data) >= 24 and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    if data.startswith(b"\xff\xd8"):
        return _jpeg_size(data)
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return _webp_size(data)
    raise ValueError(f"cannot read image dimensions from {path.name}; use PNG, JPEG, or WebP")


def _jpeg_size(data: bytes) -> tuple[int, int]:
    index = 2
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while index + 8 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > len(data):
            break
        length = struct.unpack(">H", data[index : index + 2])[0]
        if marker in sof and index + 7 <= len(data):
            height, width = struct.unpack(">HH", data[index + 3 : index + 7])
            return int(width), int(height)
        index += length
    raise ValueError("JPEG file has no size marker")


def _webp_size(data: bytes) -> tuple[int, int]:
    index = 12
    while index + 8 <= len(data):
        chunk = data[index : index + 4]
        size = struct.unpack("<I", data[index + 4 : index + 8])[0]
        payload = data[index + 8 : index + 8 + size]
        if chunk == b"VP8X" and len(payload) >= 10:
            width = 1 + int.from_bytes(payload[4:7], "little")
            height = 1 + int.from_bytes(payload[7:10], "little")
            return width, height
        if chunk == b"VP8 " and len(payload) >= 10:
            width = int.from_bytes(payload[6:8], "little") & 0x3FFF
            height = int.from_bytes(payload[8:10], "little") & 0x3FFF
            return width, height
        index += 8 + size + (size % 2)
    raise ValueError("WebP file has no size chunk")


def write_placeholder_png(path: Path) -> None:
    """Write a 1×1 PNG. The billable size is stored beside it, not in the pixels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"\x00\xff\xff\xff"
    compressed = zlib.compress(raw)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = _PNG_SIGNATURE
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", compressed)
    png += chunk(b"IEND", b"")
    path.write_bytes(png)
