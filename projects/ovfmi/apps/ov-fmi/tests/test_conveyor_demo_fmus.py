from pathlib import Path
import sys
from xml.etree import ElementTree as ET

from fmpy import read_model_description


REPO_ROOT = Path(__file__).resolve().parents[3]
OV_FMI = REPO_ROOT / "apps" / "ov-fmi"

sys.path.insert(0, str(OV_FMI))

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
            str(REPO_ROOT / "fmu" / "fmi2" / folder),
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
    root = ET.parse(REPO_ROOT / "ssp" / "conveyor_demo" / "SystemStructure.ssd").getroot()
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
