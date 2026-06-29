# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retired Vulkan viewer adapter.

The viewer-facing renderer API is OVRTX only. This module used to expose a
Newton-style procedural viewer backed directly by private Vulkan bindings; that
surface is intentionally no longer available as a public viewer path.
"""

from __future__ import annotations


class ViewerVulkan:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "nusd_renderer.viewer_vulkan was retired. Use ovrtx.Renderer or "
            "nanousdview._backend.OvrtxViewportRenderer; procedural mesh logging "
            "needs a real OVRTX authoring API before it can return."
        )
