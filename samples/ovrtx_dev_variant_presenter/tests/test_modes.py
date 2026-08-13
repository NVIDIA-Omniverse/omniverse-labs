# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dev_variant_presenter.models import QualitySpec
from dev_variant_presenter.render.modes import rendermode_attrs


def test_pathtracing_attrs():
    attrs = rendermode_attrs(QualitySpec(mode="PathTracing", samples_per_pixel=128, max_bounces=6))
    assert attrs["omni:rtx:rendermode"] == ("token", "PathTracing")
    assert attrs["omni:rtx:pt:samplesPerPixel"] == ("int", 128)
    assert attrs["omni:rtx:rtpt:maxBounces"] == ("int", 6)


def test_rt2_token_is_camelcase():
    attrs = rendermode_attrs(QualitySpec())  # default = RealTimePathTracing
    assert attrs["omni:rtx:rendermode"] == ("token", "RealTimePathTracing")  # NOT "Real-Time Path-Tracing"
