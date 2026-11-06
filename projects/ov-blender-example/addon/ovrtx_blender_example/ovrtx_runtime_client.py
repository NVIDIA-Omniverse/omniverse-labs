# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared render results and errors for the external OVStream runtime."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping
from . import color_presentation
from .semantic_output import SemanticCapture
SIMULATION_ID_PREFIX = "ovrtx-blender-"
DEFAULT_RENDER_SIMULATION_STEP_NS = 16_666_667

class RenderClientError(RuntimeError):
    """Raised when the render client cannot produce a render result."""


class ValueUpdateRecoveryRequired(RenderClientError):
    """An uncertain write epoch needs explicit simulation recovery; never retry it."""

    recovery_required = True


class ValueUpdateValidationError(RenderClientError):
    """Invalid local input needs a corrected request, rather than an RPC retry."""

    retryable = False


class RuntimeServicesPreparingError(RenderClientError):
    """Runtime-services worker is (re)starting: a transient wait, not a failure.

    The viewport session start raises this while the runtime services respawn
    (e.g. a worker restart). The render loop treats it as a deferral -- hold the
    loading state and retry -- rather than a failure publication.
    """


@dataclass(frozen=True)
class RenderResult:
    width: int
    height: int
    rgba8: bytes | memoryview
    completed_samples: int
    session_completed_samples: int
    simulation_time_ns: int
    render_output_simulation_time_ns: int = 0
    frame_format: str = color_presentation.FRAME_FORMAT_RGBA8
    frame_color_mode: str = color_presentation.FRAME_COLOR_MODE_DISPLAY_LDR
    render_var: str = color_presentation.RENDER_VAR_LDR_COLOR
    linear_rgba16f: bytes | memoryview = b""
    native_timings: Mapping[str, Any] = field(default_factory=dict)
    rgba8_origin: Literal["bottom_left", "top_left"] = "bottom_left"
    # Optional GPU-prepared upload plane; never used for capture/export.
    presentation_rgba32f: bytes | memoryview = b""
    semantic_capture: "SemanticCapture | None" = None
    # Exact requested RenderProduct identity, retained after copy and ACK.
    product_id: str = ""

    def rgba8_bytes(self, *, origin: Literal["bottom_left", "top_left"] = "bottom_left") -> bytes:
        """Materialize canonical bytes only at capture/final-output boundaries."""
        if not self.rgba8 and self.linear_rgba16f:
            raise RenderClientError("HDR-only frames require capture of Blender's displayed viewport, not an LDR render plane")
        if origin not in {"bottom_left", "top_left"} or self.rgba8_origin not in {"bottom_left", "top_left"}:
            raise RenderClientError("Unsupported RGBA8 row origin")
        if origin == self.rgba8_origin:
            return bytes(self.rgba8)
        return _flip_payload_rows(self.rgba8, self.width, self.height, 4)

    def linear_rgba16f_bytes(self, *, origin: Literal["bottom_left", "top_left"] = "bottom_left") -> bytes:
        """Materialize linear pixels at final-render/export boundaries only."""
        if origin not in {"bottom_left", "top_left"} or self.rgba8_origin not in {"bottom_left", "top_left"}:
            raise RenderClientError("Unsupported HDR row origin")
        if origin == self.rgba8_origin:
            return bytes(self.linear_rgba16f)
        return _flip_payload_rows(self.linear_rgba16f, self.width, self.height, 8)


def _flip_payload_rows(payload: bytes | memoryview, width: int, height: int, bytes_per_pixel: int) -> bytes:
    row_size = width * bytes_per_pixel
    return b"".join(payload[row * row_size : (row + 1) * row_size] for row in range(height - 1, -1, -1))
