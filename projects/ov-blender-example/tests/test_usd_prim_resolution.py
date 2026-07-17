# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import json
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.usd_prim_resolution import (  # noqa: E402
    UsdPrimResolution,
    UsdPrimResolutionStatus,
)


def test_success_requires_value_without_error_reason() -> None:
    value = object()
    result = UsdPrimResolution(UsdPrimResolutionStatus.OK, value, diagnostics={"match": "path"})

    assert result.value is value
    assert result.error_reason == ""
    assert dict(result.diagnostics) == {"match": "path"}
    with pytest.raises(ValueError, match="requires a value"):
        UsdPrimResolution(UsdPrimResolutionStatus.OK)
    with pytest.raises(ValueError, match="cannot have an error reason"):
        UsdPrimResolution(UsdPrimResolutionStatus.OK, value, "ambiguous")


def test_error_requires_reason_without_value() -> None:
    result = UsdPrimResolution[object](UsdPrimResolutionStatus.ERROR, error_reason="ambiguous")

    assert result.value is None
    assert result.error_reason == "ambiguous"
    with pytest.raises(ValueError, match="requires an error reason"):
        UsdPrimResolution[object](UsdPrimResolutionStatus.ERROR)
    with pytest.raises(ValueError, match="cannot have a value"):
        UsdPrimResolution(UsdPrimResolutionStatus.ERROR, object(), "ambiguous")


def test_diagnostics_are_frozen_at_construction() -> None:
    diagnostics = {"candidates": [{"path": "/World/A"}]}
    result = UsdPrimResolution(UsdPrimResolutionStatus.OK, object(), diagnostics=diagnostics)
    diagnostics["later"] = True
    diagnostics["candidates"][0]["path"] = "/World/B"

    assert "later" not in result.diagnostics
    assert result.diagnostics["candidates"][0]["path"] == "/World/A"
    with pytest.raises(TypeError):
        result.diagnostics["later"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        result.diagnostics["candidates"][0]["path"] = "/World/B"
    assert result.diagnostics_dict() == {"candidates": [{"path": "/World/A"}]}
    assert json.loads(json.dumps(result.diagnostics_dict())) == {"candidates": [{"path": "/World/A"}]}


@pytest.mark.parametrize("status", ["ok", "error", "unknown"])
def test_status_requires_enum(status: str) -> None:
    with pytest.raises(ValueError, match="unsupported USD prim resolution status"):
        UsdPrimResolution(status, object())  # type: ignore[arg-type]
