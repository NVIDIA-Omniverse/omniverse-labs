# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.ovrtx_value_updates import (  # noqa: E402
    OvrtxAttributeValue,
    OvrtxSessionUpdatePort,
    OvrtxTransformValue,
    OvrtxValueUpdateResult,
)


class _FakeRenderClient:
    def __init__(self) -> None:
        self.transform_calls: list[tuple[object, tuple[OvrtxTransformValue, ...]]] = []
        self.attribute_calls: list[tuple[object, tuple[OvrtxAttributeValue, ...]]] = []

    def update_transforms(
        self,
        simulation_id: object,
        values: tuple[OvrtxTransformValue, ...],
    ) -> OvrtxValueUpdateResult:
        self.transform_calls.append((simulation_id, values))
        return OvrtxValueUpdateResult(len(values), 20, {"builder_name": "semantic"})

    def update_attribute_values(
        self,
        simulation_id: object,
        values: tuple[OvrtxAttributeValue, ...],
    ) -> OvrtxValueUpdateResult:
        self.attribute_calls.append((simulation_id, values))
        return OvrtxValueUpdateResult(len(values), 20, {"builder_name": "semantic"})


def test_named_values_enforce_required_interface_fields() -> None:
    with pytest.raises(ValueError, match="prim path"):
        OvrtxTransformValue("", ())
    with pytest.raises(ValueError, match="attribute"):
        OvrtxAttributeValue("/World/Key", "", 1.0, "Float")
    with pytest.raises(ValueError, match="value type"):
        OvrtxAttributeValue("/World/Key", "inputs:intensity", 1.0, "")


def test_value_update_result_enforces_count_and_pending_time() -> None:
    assert OvrtxValueUpdateResult(0).pending_simulation_time_ns is None
    with pytest.raises(ValueError, match="cannot have a pending time"):
        OvrtxValueUpdateResult(0, 20)
    with pytest.raises(ValueError, match="requires a pending time"):
        OvrtxValueUpdateResult(1)


def test_session_port_applies_explicit_transform_values() -> None:
    client = _FakeRenderClient()
    port = OvrtxSessionUpdatePort(client, "sim")
    matrix = [[1.0, 0.0, 0.0, 0.0]]
    values = [OvrtxTransformValue("/World/Cube", matrix)]

    result = port.update_transforms(values)

    assert result == OvrtxValueUpdateResult(1, 20, {"builder_name": "semantic"})
    assert client.transform_calls == [("sim", tuple(values))]
    assert client.attribute_calls == []


def test_session_port_applies_explicit_attribute_values() -> None:
    client = _FakeRenderClient()
    port = OvrtxSessionUpdatePort(client, "sim")
    values = [
        OvrtxAttributeValue("/World/Key", "inputs:intensity", 900.0, "Float"),
        OvrtxAttributeValue("/World/Key", "inputs:color", (1.0, 0.5, 0.25), "Float3"),
    ]

    result = port.update_attribute_values(values)

    assert result.updated_count == 2
    assert client.attribute_calls == [("sim", tuple(values))]
    assert client.transform_calls == []


def test_empty_batches_are_no_ops_without_client_calls() -> None:
    client = _FakeRenderClient()
    port = OvrtxSessionUpdatePort(client, "sim")

    assert port.update_transforms(()) == OvrtxValueUpdateResult(0)
    assert port.update_attribute_values(()) == OvrtxValueUpdateResult(0)
    assert client.transform_calls == []
    assert client.attribute_calls == []


def test_session_port_rejects_partial_result_counts() -> None:
    class PartialClient(_FakeRenderClient):
        def update_transforms(
            self,
            simulation_id: object,
            values: tuple[OvrtxTransformValue, ...],
        ) -> OvrtxValueUpdateResult:
            self.transform_calls.append((simulation_id, values))
            return OvrtxValueUpdateResult(1, 20)

    client = PartialClient()
    port = OvrtxSessionUpdatePort(client, "sim")

    with pytest.raises(RuntimeError, match="updated 1 of 2 values"):
        port.update_transforms(
            (
                OvrtxTransformValue("/World/A", ()),
                OvrtxTransformValue("/World/B", ()),
            )
        )


def test_session_port_rejects_duplicate_targets_before_client_calls() -> None:
    client = _FakeRenderClient()
    port = OvrtxSessionUpdatePort(client, "sim")

    with pytest.raises(ValueError, match="duplicate OVRTX transform prim path"):
        port.update_transforms(
            (
                OvrtxTransformValue("/World/Cube", ()),
                OvrtxTransformValue("/World/Cube", ()),
            )
        )
    with pytest.raises(ValueError, match="duplicate OVRTX attribute target"):
        port.update_attribute_values(
            (
                OvrtxAttributeValue("/World/Key", "inputs:intensity", 1.0, "Float"),
                OvrtxAttributeValue("/World/Key", "inputs:intensity", 2.0, "Float"),
            )
        )

    assert client.transform_calls == []
    assert client.attribute_calls == []
