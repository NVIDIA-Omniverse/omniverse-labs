#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RGBA image verdicts for comparative and golden validation."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any


MAX_CHANNEL_DELTA = 16
MAX_CHANGED_PIXEL_RATIO = 0.025
MAX_MEAN_ABSOLUTE_ERROR = 3.5
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ImageError(ValueError):
    """An image cannot qualify as RGBA8 visual evidence."""


def compare(
    control_path: Path,
    contender_path: Path,
    *,
    expected_width: int,
    expected_height: int,
    expected_presentation: str,
    control_presentation: str,
    contender_presentation: str,
) -> dict[str, Any]:
    """Compare two rendered images under the fixed visual contract."""

    try:
        control = _read_png_rgba8(control_path)
    except (OSError, ImageError, zlib.error) as error:
        return _result("unavailable", [f"invalid control image: {error}"])
    if not expected_presentation:
        return _result("unavailable", ["expected color presentation is missing"])
    control_failure = _absolute_failure(
        control,
        expected_width,
        expected_height,
        control_presentation,
        expected_presentation,
    )
    if control_failure:
        return _result("unavailable", [f"invalid control image: {control_failure}"])

    try:
        contender = _read_png_rgba8(contender_path)
    except (OSError, ImageError, zlib.error) as error:
        return _result("regression", [f"invalid contender image: {error}"])
    contender_failure = _absolute_failure(
        contender,
        expected_width,
        expected_height,
        contender_presentation,
        expected_presentation,
    )
    if contender_failure:
        return _result("regression", [f"invalid contender image: {contender_failure}"])

    metrics = _metrics(control[2], contender[2], expected_width, expected_height)
    failures: list[str] = []
    if metrics["alpha_mismatches"]:
        failures.append(f"alpha mismatch in {metrics['alpha_mismatches']} pixels")
    if metrics["changed_pixel_ratio"] > MAX_CHANGED_PIXEL_RATIO:
        failures.append(
            f"changed pixel ratio {metrics['changed_pixel_ratio']:.6f} exceeds "
            f"maximum {MAX_CHANGED_PIXEL_RATIO:.6f}"
        )
    if metrics["mean_absolute_error"] > MAX_MEAN_ABSOLUTE_ERROR:
        failures.append(
            f"mean absolute error {metrics['mean_absolute_error']:.6f} exceeds "
            f"maximum {MAX_MEAN_ABSOLUTE_ERROR:.6f}"
        )
    return _result("regression" if failures else "pass", failures, metrics)


def _result(
    outcome: str,
    reasons: list[str],
    metrics: dict[str, int | float] | None = None,
) -> dict[str, Any]:
    return {"outcome": outcome, "reasons": reasons, "metrics": metrics}


def _absolute_failure(
    image: tuple[int, int, bytes],
    expected_width: int,
    expected_height: int,
    presentation: str,
    expected_presentation: str,
) -> str:
    if image[:2] != (expected_width, expected_height):
        return (
            f"dimensions {image[0]}x{image[1]} do not match "
            f"{expected_width}x{expected_height}"
        )
    if not _nonblank(image[2]):
        return "image is blank"
    if presentation != expected_presentation:
        return (
            f"color presentation {presentation or 'missing'} does not match "
            f"{expected_presentation}"
        )
    return ""


def _nonblank(rgba: bytes) -> bool:
    return any(
        rgba[index + 3] and any(rgba[index + offset] for offset in range(3))
        for index in range(0, len(rgba), 4)
    )


def _metrics(
    control: bytes,
    contender: bytes,
    width: int,
    height: int,
) -> dict[str, int | float]:
    changed_pixels = 0
    total_abs = 0
    max_observed_delta = 0
    alpha_mismatches = 0
    for index in range(0, len(control), 4):
        pixel_changed = False
        for offset in range(4):
            delta = abs(control[index + offset] - contender[index + offset])
            total_abs += delta
            max_observed_delta = max(max_observed_delta, delta)
            if offset == 3 and delta:
                alpha_mismatches += 1
            if delta > MAX_CHANNEL_DELTA:
                pixel_changed = True
        if pixel_changed:
            changed_pixels += 1
    pixel_count = width * height
    return {
        "width": width,
        "height": height,
        "pixel_count": pixel_count,
        "changed_pixels": changed_pixels,
        "changed_pixel_ratio": changed_pixels / pixel_count,
        "mean_absolute_error": total_abs / len(control),
        "max_observed_channel_delta": max_observed_delta,
        "alpha_mismatches": alpha_mismatches,
    }


def _read_png_rgba8(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ImageError(f"not a PNG file: {path}")

    offset = len(PNG_SIGNATURE)
    width: int | None = None
    height: int | None = None
    idat_parts: list[bytes] = []
    saw_iend = False
    while offset < len(data):
        if offset + 8 > len(data):
            raise ImageError(f"truncated PNG chunk header: {path}")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        offset += 8
        if offset + length + 4 > len(data):
            raise ImageError(f"truncated PNG chunk data: {path}")
        chunk_data = data[offset : offset + length]
        expected_crc = struct.unpack(">I", data[offset + length : offset + length + 4])[0]
        offset += length + 4
        crc = zlib.crc32(chunk_type)
        crc = zlib.crc32(chunk_data, crc) & 0xFFFFFFFF
        if crc != expected_crc:
            raise ImageError(
                f"PNG chunk {chunk_type.decode('ascii', 'replace')} CRC mismatch: {path}"
            )
        if chunk_type == b"IHDR":
            if width is not None or len(chunk_data) != 13:
                raise ImageError(f"invalid PNG IHDR chunk: {path}")
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            if width <= 0 or height <= 0:
                raise ImageError(f"invalid PNG dimensions: {path}")
            if bit_depth != 8 or color_type != 6:
                raise ImageError(f"only RGBA8 PNG images are supported: {path}")
            if compression != 0 or filter_method != 0 or interlace != 0:
                raise ImageError(f"only standard non-interlaced PNG images are supported: {path}")
        elif chunk_type == b"IDAT":
            if width is None:
                raise ImageError(f"PNG IDAT precedes IHDR: {path}")
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            saw_iend = True
            break
    if width is None or height is None or not idat_parts or not saw_iend:
        raise ImageError(f"invalid PNG structure: {path}")
    return width, height, _unfilter_rgba8(zlib.decompress(b"".join(idat_parts)), width, height)


def _unfilter_rgba8(raw: bytes, width: int, height: int) -> bytes:
    bytes_per_pixel = 4
    row_bytes = width * bytes_per_pixel
    expected = height * (row_bytes + 1)
    if len(raw) != expected:
        raise ImageError(f"decoded PNG length {len(raw)} does not match expected {expected}")
    output = bytearray()
    previous = bytes(row_bytes)
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        row = raw[cursor : cursor + row_bytes]
        cursor += row_bytes
        recon = bytearray(row_bytes)
        for index, value in enumerate(row):
            left = recon[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            up = previous[index]
            up_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            if filter_type == 0:
                recon[index] = value
            elif filter_type == 1:
                recon[index] = (value + left) & 0xFF
            elif filter_type == 2:
                recon[index] = (value + up) & 0xFF
            elif filter_type == 3:
                recon[index] = (value + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                recon[index] = (value + _paeth(left, up, up_left)) & 0xFF
            else:
                raise ImageError(f"unsupported PNG filter type {filter_type}")
        output.extend(recon)
        previous = bytes(recon)
    return bytes(output)


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    distances = (abs(estimate - left), abs(estimate - up), abs(estimate - up_left))
    if distances[0] <= distances[1] and distances[0] <= distances[2]:
        return left
    return up if distances[1] <= distances[2] else up_left
