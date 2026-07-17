# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import world_dome_usd_prim as dome  # noqa: E402


class _FakePrim:
    def __init__(self, path: str, type_name: str, attributes=("inputs:intensity", "inputs:color")) -> None:
        self.path = path
        self.type_name = type_name
        self.attributes = attributes


def test_world_dome_resolution_uses_only_configured_path() -> None:
    index = dome._world_dome_prim_index_from_prims(
        (_FakePrim("/World/OtherDome", "DomeLight"), _FakePrim("/World/StudioDome", "DomeLight"))
    )
    result = dome.resolve_world_dome_usd_prim(index)

    assert result.value == dome.WorldDomeUsdPrim("/World/StudioDome", "DomeLight")


def test_world_dome_resolution_keeps_detailed_failures_as_reasons() -> None:
    missing = dome.resolve_world_dome_usd_prim(
        dome._world_dome_prim_index_from_prims((_FakePrim("/World/Only", "DomeLight"),))
    )
    wrong_type = dome.resolve_world_dome_usd_prim(
        dome._world_dome_prim_index_from_prims((_FakePrim("/World/StudioDome", "Scope"),))
    )
    missing_attribute = dome.resolve_world_dome_usd_prim(
        dome._world_dome_prim_index_from_prims(
            (_FakePrim("/World/StudioDome", "DomeLight", ("inputs:intensity",)),)
        )
    )

    assert missing.error_reason == dome.ERROR_NO_DOME_PRIM
    assert wrong_type.error_reason == dome.ERROR_WRONG_DOME_PRIM_TYPE
    assert missing_attribute.error_reason == dome.ERROR_MISSING_DOME_PRIM_ATTRIBUTE
    assert missing_attribute.diagnostics["missing_attributes"] == ("inputs:color",)
