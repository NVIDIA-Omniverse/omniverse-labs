# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

import ovfmi
import ovfmi.api as api_module
from ovfmi import AttributeWrite, FmiHost


def _schema():
    return {
        "instances": {
            "/World/Fmi": {
                "enabled": True,
                "fmu": "fake.fmu",
                "path": "/World/Fmi",
                "connections": [
                    {
                        "enabled": True,
                        "targets": ["/World/Body"],
                        "mappings": [
                            {
                                "fmiAttributeName": "input",
                                "usdAttributeName": "sim:input",
                                "direction": "input",
                                "usdMapping": [1, 1],
                            },
                            {
                                "fmiAttributeName": "output",
                                "usdAttributeName": "sim:output",
                                "direction": "output",
                                "usdMapping": [0, 1],
                            },
                        ],
                    }
                ],
            }
        },
        "initial_values": {
            "/World/Body": {
                "sim:input": [1.0, 2.0, 3.0],
                "sim:output": [0.0, 0.0, 0.0],
            }
        },
    }


class _RuntimeInstance:
    def __init__(self, parser_instance):
        self._parser_instance = parser_instance
        self.start_values = {}

    def get_parser_instance(self):
        return self._parser_instance

    def set_start_values(self, values):
        self.start_values = dict(values)


class _Backend:
    def __init__(self, input_router, output_router):
        self.input_router = input_router
        self.output_router = output_router
        self.instances = []
        self.released = False

    def populate(self, instances):
        for parser_instance in instances.values():
            instance = _RuntimeInstance(parser_instance)
            self.input_router.cache_connections(instance)
            self.input_router.write_start_values(instance)
            self.output_router.cache_connections(instance)
            self.instances.append(instance)

    def step(self, target_time):
        for instance in self.instances:
            value = self.input_router.get_inputs_for(instance)["input"]
            result = np.array(
                [(target_time, value * 2.0)],
                dtype=[("time", np.float64), ("output", np.float64)],
            )[0]
            self.output_router.write_outputs(instance, ["output"], result)

    def reset(self):
        pass

    def release(self):
        self.released = True


def test_public_lifecycle_and_usd_space_data_plane(monkeypatch):
    monkeypatch.setattr(api_module, "parse_source", lambda _source: _schema())
    fmi = FmiHost(_backend_factory=_Backend)

    report = fmi.attach_ovstage(object(), source_asset="ignored.usda")
    assert [instance.prim_path for instance in report.instances] == ["/World/Fmi"]

    fmi.step_sync(0.25)
    assert fmi.time == pytest.approx(0.25)
    with fmi.read(attribute_names=["sim:output"]) as result:
        assert len(result.groups) == 1
        group = result.groups[0]
        assert group.prim_paths == ("/World/Body",)
        np.testing.assert_allclose(group.tensors[0], [[4.0, 0.0, 0.0]])

    operation = fmi.write([
        AttributeWrite(
            prim_paths=("/World/Body",),
            attribute_name="sim:input",
            values=np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32),
        )
    ])
    fmi.wait_op(operation)
    fmi.step_sync(0.25)
    with fmi.read() as result:
        np.testing.assert_allclose(result.groups[0].tensors[0], [[40.0, 0.0, 0.0]])

    fmi.release()
    fmi.release()


def test_attach_requires_source_asset():
    with pytest.raises(RuntimeError, match="source_asset is required"):
        FmiHost(_backend_factory=_Backend).attach_ovstage(object())


def test_read_result_rejects_access_after_close(monkeypatch):
    monkeypatch.setattr(api_module, "parse_source", lambda _source: _schema())
    fmi = FmiHost(_backend_factory=_Backend)
    fmi.attach_ovstage(object(), source_asset="ignored.usda")
    result = fmi.read()
    result.close()
    with pytest.raises(RuntimeError, match="closed"):
        _ = result.groups
    fmi.release()


def test_public_api_contract():
    assert ovfmi.__all__ == [
        "AttributeWrite",
        "FmiHost",
        "FmiHostConfig",
        "InstanceInfo",
        "MissingInputPolicy",
        "PopulationReport",
        "ReadGroup",
        "ReadResult",
    ]
    assert all(getattr(ovfmi, name).__doc__ for name in ovfmi.__all__)
    assert not hasattr(ovfmi, "OvFmi")
    assert not hasattr(ovfmi, "OvFmiConfig")
