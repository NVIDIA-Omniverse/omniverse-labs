# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEMOAPP = REPO_ROOT / "demoapp"
sys.path.insert(0, str(DEMOAPP))

import main as ov_fmi_main  # noqa: E402
from fmi_parser import (  # noqa: E402
    FmuDirection,
    FmuParserConnection,
    FmuParserInstance,
    FmuParserMapping,
)


class _RuntimeInstance:
    def __init__(self, parser_instance):
        self._parser_instance = parser_instance

    def get_parser_instance(self):
        return self._parser_instance


def test_drive_target_is_written_only_to_control_ordinal(monkeypatch):
    parser_instance = FmuParserInstance(
        enabled=True,
        fmu="unused.fmu",
        path="/World/FMI",
        connections=[
            FmuParserConnection(
                enabled=True,
                targets=["/World/Joint"],
                mappings=[
                    FmuParserMapping(
                        fmiAttributeName="targetVelocity",
                        usdAttributeName="drive:angular:physics:targetVelocity",
                        direction=FmuDirection.OUTPUT,
                        usdMapping=(0, 0),
                    )
                ],
            )
        ],
    )
    instance = _RuntimeInstance(parser_instance)
    input_head = ov_fmi_main.OvrtxInputHead({})
    output_tail = ov_fmi_main.OvstageOutputTail(object(), input_head)
    output_tail.cache_connections(instance)

    result = np.array([(math.pi,)], dtype=[("targetVelocity", np.float32)])[0]
    output_tail.write_outputs(instance, ["targetVelocity"], result)

    writes = []
    monkeypatch.setattr(
        ov_fmi_main,
        "_write_ovstage_attribute",
        lambda _renderer, paths, attr, values, ordinal, **_kwargs: writes.append(
            (paths, attr, values.copy(), ordinal)
        ),
    )

    assert output_tail.flush_to_ovstage(3) is False
    assert output_tail.flush_controls_to_ovstage(2) is True
    assert len(writes) == 1
    assert writes[0][0] == ["/World/Joint"]
    assert writes[0][1] == "drive:angular:physics:targetVelocity"
    np.testing.assert_allclose(writes[0][2], [[180.0]], rtol=1.0e-6)
    assert writes[0][3] == 2


def test_overlap_presence_is_published_as_stage_attribute(monkeypatch):
    class _PhysX:
        def __init__(self):
            self.calls = []

        def overlap(self, geometry_type, **kwargs):
            self.calls.append((geometry_type, kwargs))
            return [{"rigid_body": 1}]

    physx = _PhysX()
    writes = []
    monkeypatch.setattr(
        ov_fmi_main,
        "_write_ovstage_attribute",
        lambda _renderer, paths, attr, values, ordinal, **_kwargs: writes.append(
            (paths, attr, values.copy(), ordinal)
        ),
    )

    wrote = ov_fmi_main._publish_overlap_sensor_results(
        physx,
        object(),
        {
            "/World/Sensor": {
                "query_shape": "/World/Sensor/Sphere",
                "position": [0.0, 0.0, 0.0],
                "radius": 0.1,
            }
        },
        ordinal=3,
        shape_geometry_type="shape",
        sphere_geometry_type="sphere",
        query_mode="any",
    )

    assert wrote is True
    assert physx.calls == [
        ("shape", {"mode": "any", "prim_path": "/World/Sensor/Sphere"})
    ]
    assert writes[0][0] == ["/World/Sensor"]
    assert writes[0][1] == "sensor:presence"
    np.testing.assert_array_equal(writes[0][2], [[1.0]])
    assert writes[0][3] == 3


def test_stage_input_refresh_can_be_limited_to_sensor_presence(monkeypatch):
    head = ov_fmi_main.OvrtxInputHead(
        {
            "/World/Sensor": {"sensor:presence": [0.0]},
            "/World/Control": {"xformOp:translate": [16.0, 16.0, 0.0]},
        }
    )
    head._input_map = {
        "/World/FMI": {
            "presence": ("/World/Sensor", "sensor:presence", 0, 1),
            "speed": ("/World/Control", "xformOp:translate", 0, 1),
        }
    }
    reads = []

    def _read(_renderer, paths, attr, ordinal):
        reads.append((paths, attr, ordinal))
        return {"/World/Sensor": [1.0]}

    monkeypatch.setattr(ov_fmi_main, "_read_ovstage_attribute_values", _read)
    head.update_from_ovstage(object(), 3, {"sensor:presence"})

    assert reads == [(["/World/Sensor"], "sensor:presence", 3)]
    assert head.get_inputs_for(_RuntimeInstance(FmuParserInstance(
        enabled=True,
        fmu="unused.fmu",
        path="/World/FMI",
        connections=[],
    ))) == {"presence": 1.0, "speed": 16.0}
