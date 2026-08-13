# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dev_variant_presenter.models import VariantChoice, QualitySpec


def test_variant_choice_is_hashable_and_frozen():
    c = VariantChoice("/World/X", "Carpaint", "Noir")
    assert c.set_name == "Carpaint"
    assert {c, c} == {c}  # hashable, dedupes


def test_quality_defaults():
    q = QualitySpec()
    assert q.mode == "RealTimePathTracing"
    assert q.resolution == (1920, 1080)
