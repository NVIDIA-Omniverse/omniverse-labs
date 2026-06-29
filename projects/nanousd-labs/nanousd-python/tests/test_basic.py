# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from pathlib import Path

import nanousd
from pxr import Gf, Usd

FIXTURE = Path(__file__).with_name("fixtures") / "root.usda"


def test_native_stage_open():
    stage = nanousd.Stage.open(str(FIXTURE))
    assert stage.is_valid
    assert len(stage.used_layers) == 2
    prim = stage.get_prim_at_path("/Prim")
    assert prim is not None
    assert prim.path == "/Prim"
    assert prim.type_name == "Xform"


def test_pxr_compat_registration():
    stage = Usd.Stage.Open(str(FIXTURE))
    assert stage
    assert len(stage.GetUsedLayers()) == 2
    assert tuple(Gf.Vec3d(1, 2, 3)) == (1.0, 2.0, 3.0)


def test_pxr_compat_reload_does_not_duplicate_meta_path_finder():
    import nanousd.pxr_compat as pxr_compat

    def finder_count() -> int:
        return sum(getattr(finder, "_nanousd_pxr_submodule_finder", False) for finder in sys.meta_path)

    before = finder_count()
    importlib.reload(pxr_compat)
    after = finder_count()
    assert before >= 1
    assert after == before
