# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from xml.etree import ElementTree as ET

from fmpy import read_model_description


REPO_ROOT = Path(__file__).resolve().parents[2]
DEMOAPP = REPO_ROOT / "demoapp"

sys.path.insert(0, str(DEMOAPP))

from ssp_runtime import _default_system_connector_value  # noqa: E402


def test_conveyor_fmi2_model_descriptions_validate():
    expected = {
        "presence_sensor": ("PresenceSensor", {"rawPresence"}, {"presence"}),
        "conveyor_controller": (
            "ConveyorController",
            {"operatorSpeed", "enable", "eStop", "sensorPresence", "rejectSpeed"},
            {"zone0Speed", "zone4Speed", "rejectActive"},
        ),
        "motor_drive": (
            "MotorDrive",
            {"speedCommand", "enable", "eStop"},
            {"targetVelocity", "actualVelocity", "fault"},
        ),
    }

    for folder, (model_id, inputs, outputs) in expected.items():
        md = read_model_description(
            str(DEMOAPP / "fmu" / "fmi2" / folder),
            validate=True,
        )
        variables = {v.name: v for v in md.modelVariables}
        assert md.fmiVersion == "2.0"
        assert md.coSimulation.modelIdentifier == model_id
        assert inputs <= variables.keys()
        assert outputs <= variables.keys()
        assert all(variables[name].causality == "input" for name in inputs)
        assert all(variables[name].causality == "output" for name in outputs)


def test_conveyor_ssp_source_declares_five_motor_instances():
    ns = {"ssd": "http://ssp-standard.org/SSP1/SystemStructureDescription"}
    root = ET.parse(
        DEMOAPP / "ssp" / "conveyor_demo" / "SystemStructure.ssd"
    ).getroot()
    system = root.find("ssd:System", ns)
    assert system is not None

    components = {
        c.get("name"): c.get("source")
        for c in system.findall("ssd:Elements/ssd:Component", ns)
    }
    assert components["PresenceSensor"] == "resources/PresenceSensor.fmu"
    assert components["ConveyorController"] == "resources/ConveyorController.fmu"
    assert {
        name for name in components
        if name.startswith("MotorDrive")
    } == {
        "MotorDrive0",
        "MotorDrive1",
        "MotorDrive2",
        "MotorDrive3",
        "MotorDrive4",
    }
    assert all(
        components[f"MotorDrive{i}"] == "resources/MotorDrive.fmu"
        for i in range(5)
    )


def test_ssp_runtime_defaults_enable_inputs_to_enabled():
    assert _default_system_connector_value("enable") == 1.0
    assert _default_system_connector_value("enabled") == 1.0
    assert _default_system_connector_value("eStop") == 0.0
    assert _default_system_connector_value("operatorSpeed") == 0.0


def test_conveyor_stage_routes_sensor_and_drives_through_ovstage():
    stage_text = (DEMOAPP / "usd" / "conveyor" / "ConveyorFMI.usda").read_text()

    assert 'custom float sensor:presence = 0' in stage_text
    assert 'token fmi:usdAttribute = "sensor:presence"' in stage_text
    assert "physx:overlap" not in stage_text
    assert stage_text.count(
        'token fmi:usdAttribute = "drive:angular:physics:targetVelocity"'
    ) == 5

    main_text = (DEMOAPP / "main.py").read_text()
    assert "ARTICULATION_DOF_VELOCITY_TARGET" not in main_text
    assert "physx.update_from_ovstage(control_ordinal, control_ordinal)" in main_text
