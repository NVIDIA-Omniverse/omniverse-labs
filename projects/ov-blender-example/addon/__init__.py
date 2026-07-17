# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blender extension entry point for the OVRTX example add-on."""

from __future__ import annotations

from . import ovrtx_blender_example as _impl

bl_info = _impl.bl_info

BLENDER_AVAILABLE = _impl.BLENDER_AVAILABLE
ADDON_PREFERENCES_ID = _impl.ADDON_PREFERENCES_ID
OvrtxExamplePreferences = _impl.OvrtxExamplePreferences
get_addon_preferences = _impl.get_addon_preferences
preflight_preferences = _impl.preflight_preferences
write_viewport_session_outputs = _impl.write_viewport_session_outputs
register = _impl.register
unregister = _impl.unregister

__all__ = [
    "ADDON_PREFERENCES_ID",
    "BLENDER_AVAILABLE",
    "OvrtxExamplePreferences",
    "bl_info",
    "get_addon_preferences",
    "preflight_preferences",
    "register",
    "unregister",
    "write_viewport_session_outputs",
]
