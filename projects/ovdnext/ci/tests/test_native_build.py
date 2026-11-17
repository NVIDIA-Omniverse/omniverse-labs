# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public build-driver contracts; the platform jobs run the real CMake/CTest work."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "ci/build_native.py"


class NativeBuildContracts(unittest.TestCase):
    def test_help_exposes_native_only_inputs(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--physics-root", result.stdout)
        self.assertNotIn("--qt-root", result.stdout)
        self.assertNotIn("--qwt-root", result.stdout)

    def load_driver(self):
        spec = importlib.util.spec_from_file_location("native_build", SCRIPT)
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_wrong_source_or_physics_identity_stops_before_configure(self):
        driver = self.load_driver()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = ["--source-root", str(root), "--build-root", str(root / "build"),
                "--physics-root", str(root), "--source-commit", "a"*40, "--physics-ref", "b"*40]
            for heads in [("c"*40,), ("a"*40, "c"*40)]:
                with self.subTest(heads=heads), patch.object(driver, "git_head", side_effect=heads), patch.object(driver, "run") as run:
                    with self.assertRaisesRegex(ValueError, "HEAD"):
                        driver.main(arguments)
                    run.assert_not_called()

    def test_native_commands_keep_inventory_gate(self):
        driver = self.load_driver()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);source = root / "source";physics = root / "physics";build = root / "build"
            for path in [source / "CMakeLists.txt", physics / "physx/pvddom/CMakeLists.txt",
                         physics / "physx/pvdruntime/compiler/cmake/PVDRuntime.cmake",
                         source / "ci/baselines/qt-free/ovdnext-ci-parity.json",
                         source / "ci/verify_test_inventory.py"]:
                path.parent.mkdir(parents=True, exist_ok=True);path.touch()
            commands = []
            def boundary(command, log=None):
                commands.append([str(value) for value in command])
                if log: Path(log).write_text("{}")
            arguments = ["--source-root", str(source), "--build-root", str(build),
                "--physics-root", str(physics), "--source-commit", "a"*40, "--physics-ref", "b"*40]
            with patch.object(driver, "git_head", side_effect=["a"*40, "b"*40]), patch.object(driver, "run", side_effect=boundary):
                driver.main(arguments)
            self.assertIn("-DCMAKE_DISABLE_FIND_PACKAGE_Qt5=ON", commands[0])
            self.assertIn("-DPVD3_REQUIRE_OVD_SUPPORT=ON", commands[0])
            self.assertIn("pvdcore_tests", commands[1])
            self.assertTrue(any("--no-tests=error" in command for command in commands))
            self.assertEqual(commands[-1][2], "verify")
            self.assertIn(str((source / "ci/baselines/qt-free/ovdnext-ci-parity.json").resolve()), commands[-1])
            self.assertIn("--scope", commands[-1]);self.assertIn("root", commands[-1])

    def test_subprocess_failure_is_not_swallowed(self):
        driver = self.load_driver()
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "failure.log"
            with self.assertRaises(subprocess.CalledProcessError):
                driver.run([sys.executable, "-c", "print('failure evidence');raise SystemExit(7)"], log)
            self.assertIn("failure evidence", log.read_text())


if __name__ == "__main__": unittest.main()
