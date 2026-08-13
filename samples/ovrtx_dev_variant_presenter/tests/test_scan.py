# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from dev_variant_presenter.scan.variants import scan_stage

SCENE = str(Path(__file__).resolve().parents[1] / "data" / "ConceptCar" / "product_configurator_base.usd")
pytestmark = pytest.mark.skipif(
    not Path(SCENE).exists(), reason="local ConceptCar mirror not present (data/ ships outside the repo)")


def test_scan_finds_known_sets_and_cameras():
    info = scan_stage(SCENE)
    names = {vs.set_name for vs in info.variant_sets}
    assert {"Carpaint", "Wheel_Colors", "Doors"} <= names
    cp = next(vs for vs in info.variant_sets if vs.set_name == "Carpaint")
    assert "Nvidia_Green_Black" in cp.variants
    assert info.default_prim == "World"
    assert len(info.cameras) >= 17
    assert info.up_axis == "Y"
