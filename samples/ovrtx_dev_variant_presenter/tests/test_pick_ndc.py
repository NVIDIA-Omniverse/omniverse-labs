# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ovrtx 0.4 pick rectangles are NDC [0,1], not pixels."""
from dev_variant_presenter.render.runtime import RenderRuntime


def test_pick_ndc_rect_center_pixel_on_1280x720():
    left, top, right, bottom = RenderRuntime._pick_ndc_rect(0.5, 0.5, 1280, 720)
    # int(0.5 * 1280) == 640; int(0.5 * 720) == 360
    assert (left, top, right, bottom) == (640 / 1280, 360 / 720, 641 / 1280, 361 / 720)
    assert 0.0 <= left < right <= 1.0
    assert 0.0 <= top < bottom <= 1.0


def test_pick_ndc_rect_clamps_edges():
    left, top, right, bottom = RenderRuntime._pick_ndc_rect(0.0, 0.0, 100, 50)
    assert (left, top, right, bottom) == (0.0, 0.0, 0.01, 0.02)
    left, top, right, bottom = RenderRuntime._pick_ndc_rect(0.999, 0.999, 100, 50)
    assert left == 99 / 100
    assert top == 49 / 50
    assert right == 1.0
    assert bottom == 1.0
