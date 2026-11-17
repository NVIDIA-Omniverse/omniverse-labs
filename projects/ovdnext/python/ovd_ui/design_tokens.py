# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Soft Glass tokens transcribed from the supplied design's dark/light canvases.

OKLCH colors are Chromium's actual RGBA8 conversions. Font files are static
instances of the exact WOFF2 assets embedded in the supplied HTML bundle.
"""
from pathlib import Path

FONT_DIRECTORY = Path(__file__).resolve().parent / "assets" / "fonts"
FONTS = {
    "regular": FONT_DIRECTORY / "Geist-400.ttf",
    "medium": FONT_DIRECTORY / "Geist-500.ttf",
    "semibold": FONT_DIRECTORY / "Geist-600.ttf",
    "mono": FONT_DIRECTORY / "DMMono-400.ttf",
    "mono_medium": FONT_DIRECTORY / "DMMono-500.ttf",
}
# ImGui FreeType requests REAL_DIM (ascender - descender); CSS uses unitsPerEm.
# Compensate through the public font_size property, retaining original outlines.
FONT_METRIC_SCALES = {"regular": 1.3, "medium": 1.3, "semibold": 1.3,
                      "mono": 1.302, "mono_medium": 1.302}


def font_pixels(css_pixels, face="regular"):
    return float(css_pixels) * FONT_METRIC_SCALES[face]

# Dark, light. Alpha values preserve the reference's surface compositing.
COLORS = {
    "stage": ("#0F1012", "#F4F4F6"),
    "background": ("#141517", "#F4F4F6"),
    "stage_center": ("#1F2126", "#FFFFFF"),
    "stage_edge": ("#141517", "#EDEEF1"),
    "surface": ("#1C1D21", "#FFFFFF"),
    "island_source": ("#1C1D21DB", "#FFFFFFD1"),
    # Measured opaque composition of the source's blurred panel. Public OVUI
    # has no backdrop blur; preserving raw alpha would expose sharp grid lines.
    "island": ("#1B1C1F", "#FCFCFD"),
    "elevated": ("#26282D", "#ECEDEF"),
    "workspace_rail": ("#FFFFFF0D", "#0000000D"),
    "workspace_tab_active": ("#26282D", "#FFFFFF"),
    "workspace_detach": ("#FFFFFF14", "#FFFFFF"),
    "workspace_detach_border": ("#00000000", "#00000014"),
    "border": ("#FFFFFF0F", "#0000000F"),
    "outline": ("#FFFFFF1A", "#0000001A"),
    "text": ("#ECEDEF", "#1B1C1F"),
    "secondary": ("#C9CBD1", "#4B4F5A"),
    "muted": ("#8B8F98", "#6E7280"),
    "accent": ("#61CB7C", "#25984D"),
    "accent_surface": ("#61CB7C24", "#25984D1F"),
    "selected_text": ("#9CEEAD", "#005416"),
    "badge_text": ("#9CEEAD", "#005F21"),
    "link": ("#61CB7C", "#007F35"),
    "on_accent": ("#0F1A13", "#FFFFFF"),
    "field": ("#0F1012", "#F4F4F6"),
    "hover": ("#FFFFFF14", "#0000000D"),
    "subtle": ("#FFFFFF0A", "#F4F4F6"),
    "transparent": ("#00000000", "#00000000"),
    "shadow": ("#00000059", "#14161E14"),
    "disabled": ("#62656D", "#9A9EA8"),
    # Feedback colors are retained application semantics; the supplied canvas
    # contains no error/warning specimen from which to derive replacements.
    "warning": ("#EBAC63", "#8E5012"),
    "error": ("#F48D95", "#B12436"),
    "error_surface": ("#F48D951A", "#B1243614"),
    "warning_surface": ("#EBAC631A", "#8E501214"),
    "profile_range": ("#61CB7C29", "#25984D29"),
}

# Supplied CSS: radial-gradient(ellipse 70% 60% at 50% 55%, center 0%, edge 70%).
STAGE_GRADIENT = {"center": (.5, .55), "radii": (.7, .6), "outer_stop": .7}

# Source CSS grid layers: preserve exact authored alpha rather than RGBA8 rounding.
STAGE_GRID_RGBA = {"dark": (1., 1., 1., .05), "light": (0., 0., 0., .07)}

LAYOUT = {
    "spacing": 8, "row": 30, "toolbar": 30, "header": 70,
    "sidebar": 292, "inspector": 320, "radius": 9,
    "island_radius": 14, "field_radius": 10, "row_radius": 8,
    "tab_radius": 7, "font": 12, "small_font": 11,
    "workspace_rail_width": 275, "workspace_rail_height": 40,
    "workspace_rail_padding": 3, "workspace_rail_gap": 2,
    "workspace_rail_radius": 11, "workspace_tab_radius": 8,
    "heading_font": 20, "section_font": 13, "padding": 14,
    "outer_margin": 16, "control": 28, "field_height": 32,
    "timeline_height": 76, "timeline_bottom": 16,
    "dialog_width": 960, "dialog_height": 640,
    "border_width": 1, "scrollbar": 6, "button_width": 112,
    "button_padding": 10, "compact_padding": 4, "zero": 0,
    "shadow_thickness": 30, "shadow_offset_y": 8,
    "object_row": 30, "object_indent": 16, "object_branch": 18,
    "object_chevron": 7, "object_detail": 13, "object_title": 17,
    "object_action_width": 24, "object_action_icon": 14,
    # Search CSS content-box circle: 11px interior + two 1.5px borders.
    "search_inset": 12, "search_icon_size": 14, "search_icon_radius": 6.25,
    "search_icon_stroke": 1.5, "search_editor_height": 24,
    "slider_track": 3, "slider_thumb": 13, "slider_gap": 12, "slider_number": 48,
    "splitter_hit_width": 8, "splitter_rule_width": 2,
    "panel_min_width": 240, "panel_max_width": 520,
    "find_width": 560, "find_height": 278,
}


def rgba(role, theme="dark"):
    """Plain sRGB values for renderers that must not import OVUI."""
    value = COLORS[role][int(theme == "light")].lstrip("#")
    if len(value) == 6:
        value += "FF"
    return tuple(int(value[index:index + 2], 16) / 255. for index in (0, 2, 4, 6))
