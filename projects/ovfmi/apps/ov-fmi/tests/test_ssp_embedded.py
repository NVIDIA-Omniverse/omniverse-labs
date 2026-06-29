"""Tests for SSP embedding in USD.

Validates the FMI 2.0 FMUs, SSP archive structure, SSP simulation via fmpy,
custom SSP runtime, USD schema parsing, and math correctness.
"""

import math
import os
import sys
import json
import zipfile
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_OV_FMI = _HERE.parent           # apps/ov-fmi
_REPO_ROOT = _OV_FMI.parents[1]  # repo root
_DATA = _REPO_ROOT / "usd" / "ov-fmi"
_SSP_FILE = _DATA / "orbit_controller.ssp"
_USDA_FILE = _DATA / "ssp_orbit.usda"

# Ensure apps/ov-fmi is on path for local imports
sys.path.insert(0, str(_OV_FMI))


_PREPARED_SSP_FILE = None


def _prepared_ssp_file():
    global _PREPARED_SSP_FILE
    if _PREPARED_SSP_FILE is None:
        from ssp_runtime import prepare_ssp_archive_for_current_platform
        _PREPARED_SSP_FILE = Path(
            prepare_ssp_archive_for_current_platform(_SSP_FILE)
        )
    return _PREPARED_SSP_FILE


def _usd_python():
    env_python = os.environ.get("USD_PYTHON")
    if env_python and Path(env_python).exists():
        return env_python

    env_file = _OV_FMI / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("USD_PYTHON="):
                value = line.split("=", 1)[1].strip().strip('"')
                if Path(value).exists():
                    return value

    return sys.executable


# ===================================================================
# 1. SSP archive validation
# ===================================================================

class TestSspArchive:
    """Validate the .ssp archive structure and contents."""

    def test_ssp_file_exists(self):
        assert _SSP_FILE.exists(), f"SSP file not found: {_SSP_FILE}"

    def test_ssp_is_valid_zip(self):
        assert zipfile.is_zipfile(_SSP_FILE), "SSP file is not a valid ZIP archive"

    def test_ssp_contains_ssd(self):
        with zipfile.ZipFile(_SSP_FILE) as z:
            names = z.namelist()
            assert "SystemStructure.ssd" in names, \
                f"Missing SystemStructure.ssd, got: {names}"

    def test_ssp_contains_fmu_resources(self):
        with zipfile.ZipFile(_SSP_FILE) as z:
            names = z.namelist()
            assert "resources/TrajectoryGenerator.fmu" in names
            assert "resources/PDController.fmu" in names

    def test_ssd_xml_well_formed(self):
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(_SSP_FILE) as z:
            ssd_bytes = z.read("SystemStructure.ssd")
        root = ET.fromstring(ssd_bytes)
        assert root is not None

    def test_ssd_has_expected_components(self):
        import xml.etree.ElementTree as ET
        ns = {"ssd": "http://ssp-standard.org/SSP1/SystemStructureDescription"}
        with zipfile.ZipFile(_SSP_FILE) as z:
            ssd_bytes = z.read("SystemStructure.ssd")
        root = ET.fromstring(ssd_bytes)
        components = root.findall(".//ssd:Component", ns)
        names = {c.get("name") for c in components}
        assert "TrajectoryGenerator" in names
        assert "PDController" in names

    def test_ssd_has_internal_connections(self):
        import xml.etree.ElementTree as ET
        ns = {"ssd": "http://ssp-standard.org/SSP1/SystemStructureDescription"}
        with zipfile.ZipFile(_SSP_FILE) as z:
            ssd_bytes = z.read("SystemStructure.ssd")
        root = ET.fromstring(ssd_bytes)
        connections = root.findall(".//ssd:Connection", ns)
        # Should have: 3 internal (target xyz), 6 input pass-through, 3 output pass-through
        assert len(connections) >= 12, f"Expected ≥12 connections, got {len(connections)}"

    def test_ssd_system_level_connectors(self):
        import xml.etree.ElementTree as ET
        ns = {"ssd": "http://ssp-standard.org/SSP1/SystemStructureDescription"}
        with zipfile.ZipFile(_SSP_FILE) as z:
            ssd_bytes = z.read("SystemStructure.ssd")
        root = ET.fromstring(ssd_bytes)
        system = root.find("ssd:System", ns)
        connectors = system.find("ssd:Connectors", ns).findall("ssd:Connector", ns)
        input_names = {c.get("name") for c in connectors if c.get("kind") == "input"}
        output_names = {c.get("name") for c in connectors if c.get("kind") == "output"}
        assert input_names == {"pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"}
        assert output_names == {"force_x", "force_y", "force_z"}

    def test_internal_fmus_are_valid_zips(self):
        with zipfile.ZipFile(_SSP_FILE) as ssp:
            for fmu_name in ["resources/TrajectoryGenerator.fmu",
                             "resources/PDController.fmu"]:
                fmu_bytes = ssp.read(fmu_name)
                f = tempfile.NamedTemporaryFile(suffix=".fmu", delete=False)
                try:
                    f.write(fmu_bytes)
                    f.close()
                    assert zipfile.is_zipfile(f.name), f"{fmu_name} is not a valid ZIP"
                finally:
                    os.unlink(f.name)


# ===================================================================
# 2. FMI 2.0 FMU standalone tests (via fmpy)
# ===================================================================

class TestFmi2StandaloneFmus:
    """Test FMI 2.0 FMUs individually via fmpy.simulate_fmu."""

    @pytest.fixture
    def trajectory_fmu(self):
        """Extract TrajectoryGenerator.fmu to a temp dir."""
        with zipfile.ZipFile(_prepared_ssp_file()) as ssp:
            fmu_bytes = ssp.read("resources/TrajectoryGenerator.fmu")
        f = tempfile.NamedTemporaryFile(suffix=".fmu", delete=False)
        f.write(fmu_bytes)
        f.close()
        yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def pdcontroller_fmu(self):
        """Extract PDController.fmu to a temp dir."""
        with zipfile.ZipFile(_prepared_ssp_file()) as ssp:
            fmu_bytes = ssp.read("resources/PDController.fmu")
        f = tempfile.NamedTemporaryFile(suffix=".fmu", delete=False)
        f.write(fmu_bytes)
        f.close()
        yield f.name
        os.unlink(f.name)

    def test_trajectory_generator_outputs_orbit(self, trajectory_fmu):
        import fmpy
        result = fmpy.simulate_fmu(
            trajectory_fmu,
            stop_time=1.0,
            step_size=1/60,
            output=["target_x", "target_y", "target_z"],
        )
        assert len(result) > 0
        # At t≈0: target_x ≈ 1.5, target_z ≈ 0
        assert abs(result[0]["target_x"] - 1.5) < 0.01
        assert abs(result[0]["target_z"]) < 0.01
        # target_y should always be 2.0 (orbit height)
        for row in result:
            assert abs(row["target_y"] - 2.0) < 0.001

    def test_trajectory_generator_circular_orbit(self, trajectory_fmu):
        import fmpy
        result = fmpy.simulate_fmu(
            trajectory_fmu,
            stop_time=6.28,  # ~2π, one full orbit
            step_size=1/60,
            output=["target_x", "target_z"],
        )
        # Should return to start position after one orbit
        first = result[0]
        last = result[-1]
        assert abs(last["target_x"] - first["target_x"]) < 0.1
        assert abs(last["target_z"] - first["target_z"]) < 0.1

    def test_trajectory_generator_orbit_radius(self, trajectory_fmu):
        import fmpy
        result = fmpy.simulate_fmu(
            trajectory_fmu,
            stop_time=1.0,
            step_size=1/60,
            output=["target_x", "target_z"],
        )
        for row in result:
            r = math.sqrt(row["target_x"]**2 + row["target_z"]**2)
            assert abs(r - 1.5) < 0.01, f"Orbit radius {r} != 1.5"

    def test_pd_controller_gravity_compensation(self, pdcontroller_fmu):
        import fmpy
        # Body at target position, zero velocity → only gravity compensation
        result = fmpy.simulate_fmu(
            pdcontroller_fmu,
            stop_time=0.1,
            step_size=1/60,
            start_values={"pos_y": 2.0, "target_y": 2.0},
            output=["force_x", "force_y", "force_z"],
        )
        # force_y should be ~9.81 (gravity compensation), force_x/z ≈ 0
        last = result[-1]
        assert abs(last["force_y"] - 9.81) < 0.5
        assert abs(last["force_x"]) < 0.1
        assert abs(last["force_z"]) < 0.1

    def test_pd_controller_restoring_force(self, pdcontroller_fmu):
        import fmpy
        # Body displaced from target → should produce restoring force
        result = fmpy.simulate_fmu(
            pdcontroller_fmu,
            stop_time=0.1,
            step_size=1/60,
            start_values={"pos_x": 1.0, "target_x": 0.0},
            output=["force_x"],
        )
        # force_x should be negative (pushing back toward target at 0)
        assert result[-1]["force_x"] < -10.0


# ===================================================================
# 3. SSP simulation tests (via fmpy.ssp.simulation)
# ===================================================================

class TestSspFmpySimulation:
    """Test SSP end-to-end via fmpy's built-in SSP simulation."""

    def test_ssp_simulate_runs(self):
        from fmpy.ssp.simulation import simulate_ssp
        result = simulate_ssp(str(_prepared_ssp_file()), stop_time=1.0, step_size=1/60)
        assert len(result) > 0

    def test_ssp_simulate_has_force_outputs(self):
        from fmpy.ssp.simulation import simulate_ssp
        result = simulate_ssp(str(_prepared_ssp_file()), stop_time=1.0, step_size=1/60)
        names = result.dtype.names
        assert "force_x" in names
        assert "force_y" in names
        assert "force_z" in names

    def test_ssp_simulate_force_nonzero(self):
        from fmpy.ssp.simulation import simulate_ssp
        result = simulate_ssp(str(_prepared_ssp_file()), stop_time=1.0, step_size=1/60)
        # Forces should be non-zero (body at origin, target orbiting at r=1.5)
        fx_max = max(abs(row["force_x"]) for row in result)
        fy_max = max(abs(row["force_y"]) for row in result)
        fz_max = max(abs(row["force_z"]) for row in result)
        assert fx_max > 10.0, f"force_x max {fx_max} too small"
        assert fy_max > 9.0, f"force_y max {fy_max} too small"
        assert fz_max > 1.0, f"force_z max {fz_max} too small"

    def test_ssp_simulate_force_y_includes_gravity(self):
        from fmpy.ssp.simulation import simulate_ssp
        result = simulate_ssp(str(_prepared_ssp_file()), stop_time=1.0, step_size=1/60)
        # force_y should include gravity compensation (~9.81) + PD term
        # With body at y=0, target at y=2: force_y ≈ 50*2 + 9.81 = 109.81
        fy_values = [row["force_y"] for row in result[1:]]  # skip t=0
        assert all(fy > 9.0 for fy in fy_values), "force_y should always be positive (gravity comp)"


# ===================================================================
# 4. Custom SSP runtime tests
# ===================================================================

class TestSspRuntime:
    """Test the custom SspRuntimeInstance used by main.py."""

    @pytest.fixture
    def ssp_instance(self):
        from ssp_runtime import SspRuntimeInstance
        from fmi_parser import FmuParserInstance
        pi = FmuParserInstance(
            enabled=True,
            fmu=str(_SSP_FILE),
            path="/World/OrbitController",
            connections=[],
        )
        inst = SspRuntimeInstance(pi, str(_SSP_FILE))
        yield inst
        inst.destroy()

    def test_ssp_runtime_loads(self, ssp_instance):
        assert ssp_instance is not None

    def test_ssp_runtime_first_step(self, ssp_instance):
        result = ssp_instance.step(
            filename="",
            inputs={"pos_x": 0.0, "pos_y": 0.0, "pos_z": 0.0,
                    "vel_x": 0.0, "vel_y": 0.0, "vel_z": 0.0},
            outputs=["force_x", "force_y", "force_z"],
            time=1/60,
        )
        assert result is not None
        # At t=1/60 with body at origin: target≈(1.5, 2.0, 0.025)
        assert abs(result["force_x"] - 74.99) < 1.0
        assert abs(result["force_y"] - 109.81) < 1.0
        assert abs(result["force_z"]) < 5.0

    def test_ssp_runtime_multiple_steps(self, ssp_instance):
        dt = 1/60
        for i in range(60):
            t = (i + 1) * dt
            result = ssp_instance.step(
                filename="",
                inputs={"pos_x": 0.0, "pos_y": 0.0, "pos_z": 0.0,
                        "vel_x": 0.0, "vel_y": 0.0, "vel_z": 0.0},
                outputs=["force_x", "force_y", "force_z"],
                time=t,
            )
            assert result is not None

    def test_ssp_runtime_skip_same_time(self, ssp_instance):
        """Calling step with time <= last should return None."""
        ssp_instance.step("", {"pos_x": 0.0, "pos_y": 0.0, "pos_z": 0.0,
                                "vel_x": 0.0, "vel_y": 0.0, "vel_z": 0.0},
                          ["force_x"], 1/60)
        result = ssp_instance.step("", {"pos_x": 0.0, "pos_y": 0.0, "pos_z": 0.0,
                                         "vel_x": 0.0, "vel_y": 0.0, "vel_z": 0.0},
                                   ["force_x"], 1/60)
        assert result is None

    def test_ssp_runtime_force_oscillates(self, ssp_instance):
        """Force x should change sign as the orbit target circles."""
        dt = 1/60
        fx_values = []
        for i in range(180):  # 3 seconds
            t = (i + 1) * dt
            result = ssp_instance.step(
                filename="",
                inputs={"pos_x": 0.0, "pos_y": 0.0, "pos_z": 0.0,
                        "vel_x": 0.0, "vel_y": 0.0, "vel_z": 0.0},
                outputs=["force_x"],
                time=t,
            )
            fx_values.append(float(result["force_x"]))
        # force_x should have both positive and negative values (orbit goes around)
        assert any(fx > 10 for fx in fx_values), "force_x never positive"
        assert any(fx < -10 for fx in fx_values), "force_x never negative"


# ===================================================================
# 5. Pure-Python math verification
# ===================================================================

class TestSspMathVerification:
    """Compare SSP output against analytically computed PD controller forces."""

    def test_force_matches_analytical(self):
        """Verify SSP force output matches hand-computed PD forces."""
        from ssp_runtime import SspRuntimeInstance
        from fmi_parser import FmuParserInstance
        pi = FmuParserInstance(enabled=True, fmu=str(_SSP_FILE),
                               path="/Test", connections=[])
        ssp = SspRuntimeInstance(pi, str(_SSP_FILE))

        dt = 1/60
        kp, kd, mass, gravity = 50.0, 10.0, 1.0, -9.81
        radius, height, omega = 1.5, 2.0, 1.0

        for i in range(1, 31):
            t = i * dt
            result = ssp.step(
                filename="",
                inputs={"pos_x": 0.0, "pos_y": 0.0, "pos_z": 0.0,
                        "vel_x": 0.0, "vel_y": 0.0, "vel_z": 0.0},
                outputs=["force_x", "force_y", "force_z"],
                time=t,
            )
            # Analytical: trajectory generator computes target at time t
            target_x = radius * math.cos(omega * t)
            target_y = height
            target_z = radius * math.sin(omega * t)

            # PD force with body at origin, zero velocity
            expected_fx = kp * target_x  # + kd * 0
            expected_fy = kp * target_y + mass * (-gravity)
            expected_fz = kp * target_z

            assert abs(result["force_x"] - expected_fx) < 0.5, \
                f"t={t:.3f}: force_x={result['force_x']:.2f} expected={expected_fx:.2f}"
            assert abs(result["force_y"] - expected_fy) < 0.5, \
                f"t={t:.3f}: force_y={result['force_y']:.2f} expected={expected_fy:.2f}"
            assert abs(result["force_z"] - expected_fz) < 0.5, \
                f"t={t:.3f}: force_z={result['force_z']:.2f} expected={expected_fz:.2f}"

        ssp.destroy()

    def test_euler_tracking_matches_fmi3_demo(self):
        """Simple Euler integration should produce the same ~0.28m tracking error
        as the FMI 3.0 multi-FMU orbit demo."""
        from ssp_runtime import SspRuntimeInstance
        from fmi_parser import FmuParserInstance
        pi = FmuParserInstance(enabled=True, fmu=str(_SSP_FILE),
                               path="/Test", connections=[])
        ssp = SspRuntimeInstance(pi, str(_SSP_FILE))

        dt = 1/60
        pos = np.array([1.5, 2.0, 0.0])
        vel = np.array([0.0, 0.0, 0.0])
        gravity_vec = np.array([0.0, -9.81, 0.0])

        for i in range(300):  # 5 seconds
            t = (i + 1) * dt
            result = ssp.step(
                filename="",
                inputs={
                    "pos_x": float(pos[0]), "pos_y": float(pos[1]), "pos_z": float(pos[2]),
                    "vel_x": float(vel[0]), "vel_y": float(vel[1]), "vel_z": float(vel[2]),
                },
                outputs=["force_x", "force_y", "force_z"],
                time=t,
            )
            force = np.array([float(result["force_x"]),
                              float(result["force_y"]),
                              float(result["force_z"])])
            acc = force / 1.0 + gravity_vec
            vel += acc * dt
            pos += vel * dt

        # After settling, tracking error should be around 0.28m (same as FMI 3.0 demo)
        target_x = 1.5 * math.cos(5.0)
        target_z = 1.5 * math.sin(5.0)
        dist = math.sqrt((pos[0] - target_x)**2 + (pos[1] - 2.0)**2 + (pos[2] - target_z)**2)
        assert dist < 0.5, f"Tracking error {dist:.3f}m too large (expected ~0.28m)"
        assert dist > 0.1, f"Tracking error {dist:.3f}m suspiciously small"

        ssp.destroy()


# ===================================================================
# 6. USD schema parsing tests
# ===================================================================

class TestUsdSchemaParsing:
    """Test that parse_fmi_schema.py correctly handles SspInstance prims."""

    @pytest.fixture
    def parsed_schema(self):
        """Run parse_fmi_schema.py on the SSP embedded test scene."""
        script = _OV_FMI / "parse_fmi_schema.py"
        result = subprocess.run(
            [_usd_python(), str(script), str(_USDA_FILE)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"Parser failed: {result.stderr}"
        return json.loads(result.stdout)

    def test_ssp_instance_detected(self, parsed_schema):
        instances = parsed_schema["instances"]
        assert "/World/OrbitController" in instances

    def test_ssp_instance_has_ssp_path(self, parsed_schema):
        inst = parsed_schema["instances"]["/World/OrbitController"]
        assert inst["ssp"] is not None
        assert inst["ssp"].endswith("orbit_controller.ssp")

    def test_ssp_instance_fmu_is_null(self, parsed_schema):
        inst = parsed_schema["instances"]["/World/OrbitController"]
        assert inst["fmu"] is None

    def test_ssp_instance_connections(self, parsed_schema):
        inst = parsed_schema["instances"]["/World/OrbitController"]
        assert len(inst["connections"]) == 1
        conn = inst["connections"][0]
        assert "/World/Sphere" in conn["targets"]

    def test_ssp_instance_input_mappings(self, parsed_schema):
        conn = parsed_schema["instances"]["/World/OrbitController"]["connections"][0]
        input_names = {m["fmiAttributeName"] for m in conn["mappings"]
                       if m["direction"] == "input"}
        assert input_names == {"pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"}

    def test_ssp_instance_output_mappings(self, parsed_schema):
        conn = parsed_schema["instances"]["/World/OrbitController"]["connections"][0]
        output_names = {m["fmiAttributeName"] for m in conn["mappings"]
                        if m["direction"] == "output"}
        assert output_names == {"force_x", "force_y", "force_z"}

    def test_rigid_body_auto_detected(self, parsed_schema):
        body_prims = parsed_schema.get("body_prims", [])
        assert "/World/Sphere" in body_prims

    def test_no_target_marker_in_scene(self, parsed_schema):
        """Verify there's no TargetMarker prim — it's internal to the SSP."""
        instances = parsed_schema["instances"]
        for path in instances:
            assert "TargetMarker" not in path


# ===================================================================
# 7. FMI 2.0 binary validation
# ===================================================================

class TestFmi2Binaries:
    """Validate that the FMI 2.0 shared libraries exist and load correctly."""

    def test_trajectory_generator_so_exists(self):
        """Check the .so is in the SSP archive."""
        with zipfile.ZipFile(_SSP_FILE) as ssp:
            fmu_bytes = ssp.read("resources/TrajectoryGenerator.fmu")
        f = tempfile.NamedTemporaryFile(suffix=".fmu", delete=False)
        try:
            f.write(fmu_bytes)
            f.close()
            with zipfile.ZipFile(f.name) as fmu:
                names = fmu.namelist()
                assert any("TrajectoryGenerator" in n for n in names), \
                    f"No TrajectoryGenerator binary found in: {names}"
        finally:
            os.unlink(f.name)

    def test_pd_controller_so_exists(self):
        with zipfile.ZipFile(_SSP_FILE) as ssp:
            fmu_bytes = ssp.read("resources/PDController.fmu")
        f = tempfile.NamedTemporaryFile(suffix=".fmu", delete=False)
        try:
            f.write(fmu_bytes)
            f.close()
            with zipfile.ZipFile(f.name) as fmu:
                names = fmu.namelist()
                assert any("PDController" in n for n in names), \
                    f"No PDController binary found in: {names}"
        finally:
            os.unlink(f.name)

    def test_model_description_fmi_version(self):
        """Both FMUs should report FMI version 2.0."""
        import fmpy
        with zipfile.ZipFile(_SSP_FILE) as ssp:
            for fmu_name in ["resources/TrajectoryGenerator.fmu",
                             "resources/PDController.fmu"]:
                fmu_bytes = ssp.read(fmu_name)
                f = tempfile.NamedTemporaryFile(suffix=".fmu", delete=False)
                try:
                    f.write(fmu_bytes)
                    f.close()
                    md = fmpy.read_model_description(f.name)
                    assert md.fmiVersion == "2.0", \
                        f"{fmu_name}: expected FMI 2.0, got {md.fmiVersion}"
                finally:
                    os.unlink(f.name)
