# SPDX-License-Identifier: Apache-2.0
"""Resolution supersampling for edge anti-aliasing (cycles-parity n-05-03).

The OVRTX RealTimePathTracing worker renders silhouette and shadow edges
essentially aliased: ``omni:rtx:post:aa:op``, TAA/FXAA, the RTPT camera jitter
and the per-pixel filter attributes are inert on the RenderProduct, sample
accumulation does not add sub-pixel coverage (1 and 512 samples give the same
edge profile), and DLSS at the native scale adds almost none. The renderer's own
settings therefore cannot anti-alias; rendering above the output resolution and
averaging down does. This module owns the factor policy and the box
downsample of a finished :class:`RenderResult`.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

SUPERSAMPLE_ITEMS = (
    ("1", "1x (Off)", "Render at the output resolution; edges stay aliased"),
    ("2", "2x2", "Render at twice the width and height and average down (4x pixels)"),
    ("3", "3x3", "Render at three times the width and height and average down (9x pixels)"),
)
SUPERSAMPLE_MAX_FACTOR = 3
#: Largest rendered frame (pixels) a supersampled request may ask for; the SDK
#: bridge refuses frames above 67,108,864 pixels, stay well inside.
SUPERSAMPLE_MAX_PIXELS = 36_000_000
SUPERSAMPLE_MAX_DIMENSION = 16384


def parse_factor(value: Any) -> int:
    try:
        factor = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, min(SUPERSAMPLE_MAX_FACTOR, factor))


def effective_factor(width: int, height: int, requested: Any) -> int:
    """The largest factor <= requested whose frame fits the pixel/dimension budget."""
    factor = parse_factor(requested)
    width, height = max(1, int(width)), max(1, int(height))
    while factor > 1 and (
        width * factor > SUPERSAMPLE_MAX_DIMENSION
        or height * factor > SUPERSAMPLE_MAX_DIMENSION
        or width * height * factor * factor > SUPERSAMPLE_MAX_PIXELS
    ):
        factor -= 1
    return factor


def _box_mean(array: Any, factor: int) -> Any:
    height, width, channels = array.shape
    return array.reshape(height // factor, factor, width // factor, factor, channels).mean(axis=(1, 3))


def downsample_result(result: Any, factor: int) -> Any:
    """Box-average ``result`` by ``factor`` in each axis, returning a new result.

    Scene-linear HDR is averaged directly. Display-encoded LDR is averaged in
    linear light (sRGB decode/encode) so partially covered edge pixels get the
    coverage-correct value. Frames whose size is not divisible by ``factor``
    are returned unchanged.
    """
    factor = int(factor)
    if factor <= 1 or result.width % factor or result.height % factor:
        return result
    import numpy as np

    width, height = result.width // factor, result.height // factor
    changes: dict[str, Any] = {"width": width, "height": height, "presentation_rgba32f": b"", "semantic_capture": None}
    if result.linear_rgba16f:
        source = np.frombuffer(result.linear_rgba16f, dtype=np.float16).reshape(result.height, result.width, 4)
        mean = _box_mean(source.astype(np.float32), factor)
        changes["linear_rgba16f"] = mean.astype(np.float16).tobytes()
    if result.rgba8:
        source = np.frombuffer(result.rgba8, dtype=np.uint8).reshape(result.height, result.width, 4)
        rgb = source[..., :3].astype(np.float32) * np.float32(1.0 / 255.0)
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        alpha = source[..., 3:].astype(np.float32) * np.float32(1.0 / 255.0)
        mean = _box_mean(np.concatenate([linear, alpha], axis=-1), factor)
        lin = np.clip(mean[..., :3], 0.0, 1.0)
        encoded = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
        out = np.concatenate([encoded, np.clip(mean[..., 3:], 0.0, 1.0)], axis=-1)
        changes["rgba8"] = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8).tobytes()
    return replace(result, **changes)
