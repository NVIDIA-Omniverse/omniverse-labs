# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from dev_variant_presenter.scan.effects import (
    Write, _linear_to_srgb_hex, _swatches, classify_variants,
)
from dev_variant_presenter.scan.variants import scan_stage

SCENE = str(Path(__file__).resolve().parents[1] / "data" / "ConceptCar" / "product_configurator_base.usd")
pytestmark = pytest.mark.skipif(
    not Path(SCENE).exists(), reason="local ConceptCar mirror not present (data/ ships outside the repo)")


def test_classifies_concept_car_sets():
    info = scan_stage(SCENE)
    actions = classify_variants(SCENE, info)
    assert set(actions) == {vs.set_name for vs in info.variant_sets}
    assert actions["Carpaint"].kind == "shader-input"
    assert actions["Doors"].kind == "transform"
    assert actions["Backdrops"].kind == "visibility"
    # nothing on this stage is structural
    assert all(a.kind != "structural" for a in actions.values())


def test_records_replayable_writes_per_variant():
    info = scan_stage(SCENE)
    actions = classify_variants(SCENE, info)
    cp = actions["Carpaint"].per_variant["Sakura"]
    assert cp and any(w.attr.startswith("inputs:") for w in cp)
    doors = actions["Doors"].per_variant["All_Open"]
    assert any("xformOp" in w.attr for w in doors)


def test_linear_to_srgb_hex_applies_gamma():
    assert _linear_to_srgb_hex([0, 0, 0]) == "#000000"
    assert _linear_to_srgb_hex([1, 1, 1]) == "#ffffff"
    # gamma brightens linear 0.5 well above the 128 a naive linear->8bit mapping gives
    assert int(_linear_to_srgb_hex([0.5, 0.5, 0.5])[1:3], 16) > 180


def test_swatches_picks_highest_variance_color_attr():
    # `base` is constant across variants; `tint` carries the identity -> tint wins
    pv = {
        "A": [Write("/M", "inputs:base", "color3f", (0.5, 0.5, 0.5)),
              Write("/M", "inputs:tint", "color3f", (1.0, 0.0, 0.0)),
              Write("/M", "inputs:metalness", "float", 0.0)],
        "B": [Write("/M", "inputs:base", "color3f", (0.5, 0.5, 0.5)),
              Write("/M", "inputs:tint", "color3f", (0.0, 0.0, 1.0)),
              Write("/M", "inputs:metalness", "float", 1.0)],
    }
    sw = _swatches(pv)
    assert set(sw) == {"A", "B"}
    assert sw["A"] != sw["B"]                       # red vs blue, not the shared grey base
    assert all(h.startswith("#") and len(h) == 7 for h in sw.values())


def test_swatches_none_for_non_color_set():
    pv = {"x": [Write("/P", "xformOp:translate", "double3", (1, 2, 3))],
          "y": [Write("/P", "xformOp:translate", "double3", (4, 5, 6))]}
    assert _swatches(pv) is None


def test_concept_car_color_sets_have_distinct_swatches():
    info = scan_stage(SCENE)
    actions = classify_variants(SCENE, info)
    cp = actions["Carpaint"].swatches
    assert cp and len(set(cp.values())) > 3         # distinct paint colors
    assert all(h.startswith("#") and len(h) == 7 for h in cp.values())
    assert actions["Doors"].swatches is None        # transform set -> no swatch
