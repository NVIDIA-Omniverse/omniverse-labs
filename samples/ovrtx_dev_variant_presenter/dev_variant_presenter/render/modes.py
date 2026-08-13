# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map a QualitySpec to omni:rtx:* RenderProduct attributes.

The composer authors these on the RenderProduct for the initial mode; a runtime
mode change writes them via write_attribute on the existing RenderProduct + reset()
+ warm-up (no reload). RT2 token is "RealTimePathTracing".
"""
from __future__ import annotations

from dev_variant_presenter.models import QualitySpec


def rendermode_attrs(q: QualitySpec) -> dict[str, tuple[str, object]]:
    """Return {attr_name: (usd_type, value)} for the render mode + sub-settings."""
    return {
        "omni:rtx:rendermode": ("token", q.mode),
        "omni:rtx:pt:samplesPerPixel": ("int", q.samples_per_pixel),
        "omni:rtx:rtpt:maxBounces": ("int", q.max_bounces),
    }
