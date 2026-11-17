# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import copy
import io
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from ci import verify_test_inventory as inventory
from ci import verify_ovdnext_data_dependencies as dependencies


DATA_TESTS = [
    "ovddata_tests",
    "ovddata_ingest_tests",
    "ovddata_omnipvd_tests",
    "ovdnext_data_probe_tests",
]

PLATFORM_EXCLUSIONS = {
    "windows-x86_64": [],
    "linux-x86_64": [
        {
            "name": "PvdArchiveSecurityTest.RejectsEntryNamesThatDoNotFitZipEntryBuffer",
            "reason": "_WIN32-guarded test exercises bundled minizip entry-name buffer handling linked only on Windows",
        },
        {
            "name": "PvdArchiveSecurityTest.RejectsTruncatedExtendedTimestampField",
            "reason": "_WIN32-guarded test exercises bundled minizip extended-timestamp parsing linked only on Windows",
        },
    ],
}

DISABLED_TESTS = [
    "OvdLoaderTest.DumpSnapshotToStdout",
    'RealWorldCaptures/OvdCaptureSmokeTest.DumpSnapshotToStdout/"2026_04_01_15_21_33_0000019f4952a142.ovd"',
    'RealWorldCaptures/OvdCaptureSmokeTest.DumpSnapshotToStdout/"2026_04_01_16_42_23_00000062b3443b1b.ovd"',
    'RealWorldCaptures/OvdCaptureSmokeTest.DumpSnapshotToStdout/"snippet_meshes.ovd"',
    'RealWorldCaptures/OvdCaptureSmokeTest.DumpSnapshotToStdout/"multi_geom.ovd"',
]

ALLOWED_SKIPS = [
    {
        "name": "OvdInputValidationTest.FileLoaderAcceptsLargestAvailableCorpusCapture",
        "reason": "largest local corpus capture is intentionally uncommitted",
    },
    {
        "name": "OvdMultiSceneTest.FactoryChildrenHaveNoOwningScene",
        "reason": "committed fixture may contain no PxMaterial",
    },
    {
        "name": "OvdMultiSceneDeformableTest.DeformablesPartitionAcrossScenes",
        "reason": "VolumeVolumeCollision capture is intentionally uncommitted",
    },
]

REQUIRED_IMPORTED_TESTS = ["OvdLoaderTest.ValidCapture"]

JOB_MAPPINGS = [
    {
        "legacy_id": "prepare:release",
        "category": "release_preparation",
        "github_ids": ["prepare-release"],
        "disposition": "preserved",
    },
    {
        "legacy_id": "verify:windows",
        "category": "windows_verification",
        "github_ids": ["verify-root-windows", "verify-ovdnext-data-windows"],
        "disposition": "expanded",
    },
    {
        "legacy_id": "verify:linux",
        "category": "linux_verification",
        "github_ids": ["verify-root-linux", "verify-ovdnext-data-linux"],
        "disposition": "expanded",
    },
    {
        "legacy_id": "package:windows",
        "category": "windows_packaging",
        "github_ids": ["package-windows"],
        "disposition": "preserved",
    },
    {
        "legacy_id": "package:linux",
        "category": "linux_packaging",
        "github_ids": ["package-linux"],
        "disposition": "preserved",
    },
    {
        "legacy_id": "release:merged-mr",
        "category": "release_publication",
        "github_ids": ["publish-release"],
        "disposition": "migrated",
    },
]


def discovery_entry(name, disabled=False):
    properties = []
    if disabled:
        properties.append({"name": "DISABLED", "value": True})
    return {"name": name, "command": [name], "properties": properties}


def baseline_entry(name, disabled=False):
    return {"name": name, "disabled": disabled}


def baseline_fixture():
    enabled_names = ["OvdLoaderTest.ValidCapture"]
    enabled_names.extend(entry["name"] for entry in ALLOWED_SKIPS)
    enabled_names.extend(entry["name"] for entry in PLATFORM_EXCLUSIONS["linux-x86_64"])
    enabled_names.extend("LegacySuite.Case%03d" % index for index in range(1, 229))
    legacy_tests = [baseline_entry(name) for name in enabled_names]
    legacy_tests.extend(baseline_entry(name, True) for name in DISABLED_TESTS)
    return {
        "schema_version": 1,
        "source_revisions": {
            "legacy_tests_commit": "857044a4ec51dc73b8931387cf6ff451d4506664",
            "ovdnext_data_foundation_commit": "630e1d071522a06ceac534a5dd7e8ab7e65efdd8",
            "combined_inventory_commit": "630e1d071522a06ceac534a5dd7e8ab7e65efdd8",
        },
        "required_configuration": "RelWithDebInfo",
        "platform_scope": ["windows-x86_64", "linux-x86_64"],
        "ctest": {
            "legacy": {
                "registered_count": 239,
                "enabled_count": 234,
                "disabled_count": 5,
                "tests": legacy_tests,
            },
            "ovdnext_data_foundation": {
                "registered_count": 4,
                "enabled_count": 4,
                "disabled_count": 0,
                "tests": [baseline_entry(name) for name in DATA_TESTS],
            },
            "combined": {
                "registered_count": 243,
                "enabled_count": 238,
                "disabled_count": 5,
            },
            "allowed_runtime_skips": copy.deepcopy(ALLOWED_SKIPS),
            "maximum_runtime_skip_count": 3,
            "platform_exclusions": copy.deepcopy(PLATFORM_EXCLUSIONS),
            "required_imported_test_names": copy.deepcopy(REQUIRED_IMPORTED_TESTS),
        },
        "pipeline": {
            "legacy_job_count": 6,
            "github_job_count": 8,
            "pull_request_blocking_job_count": 4,
            "job_mappings": copy.deepcopy(JOB_MAPPINGS),
            "pull_request_blocking_jobs": [
                "verify-root-windows",
                "verify-root-linux",
                "verify-ovdnext-data-windows",
                "verify-ovdnext-data-linux",
            ],
        },
    }


def policy_fixture(baseline):
    policy = copy.deepcopy(baseline)
    policy["ctest"]["legacy"].pop("tests")
    policy["ctest"]["ovdnext_data_foundation"].pop("tests")
    policy["ctest"]["disabled_tests"] = DISABLED_TESTS
    policy["ctest"]["ovdnext_data_test_names"] = DATA_TESTS
    return policy


def discovery_fixture(baseline):
    tests = baseline["ctest"]["legacy"]["tests"] + baseline["ctest"]["ovdnext_data_foundation"]["tests"]
    return {
        "kind": "ctestInfo",
        "version": {"major": 1, "minor": 0},
        "tests": [discovery_entry(test["name"], test["disabled"]) for test in tests],
    }


def write_junit(path, discovery, skipped=(), failed=(), errors=()):
    skipped = set(skipped)
    failed = set(failed)
    errors = set(errors)
    suite = ET.Element("testsuite", name="ovdnext")
    for test in discovery["tests"]:
        disabled = any(
            prop.get("name") == "DISABLED" and prop.get("value")
            for prop in test.get("properties", [])
        )
        if disabled and test["name"] not in skipped | failed | errors:
            continue
        case = ET.SubElement(suite, "testcase", name=test["name"])
        if test["name"] in skipped:
            ET.SubElement(case, "skipped")
        if test["name"] in failed:
            ET.SubElement(case, "failure", message="failed")
        if test["name"] in errors:
            ET.SubElement(case, "error", message="error")
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


class InventoryVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.baseline = baseline_fixture()
        self.policy = policy_fixture(self.baseline)
        self.discovery = discovery_fixture(self.baseline)

    def tearDown(self):
        self.temporary.cleanup()

    def write_inputs(self, skipped=(), failed=(), errors=()):
        baseline_path = self.root / "baseline.json"
        policy_path = self.root / "ovdnext-ci-parity-policy.json"
        discovery_path = self.root / "discovery.json"
        junit_path = self.root / "junit.xml"
        baseline_path.write_text(json.dumps(self.baseline), encoding="utf-8")
        policy_path.write_text(json.dumps(self.policy), encoding="utf-8")
        discovery_path.write_text(json.dumps(self.discovery), encoding="utf-8")
        write_junit(
            junit_path,
            self.discovery,
            skipped=skipped,
            failed=failed,
            errors=errors,
        )
        return baseline_path, discovery_path, junit_path

    def assert_inventory_failure(
        self,
        expected,
        skipped=(),
        failed=(),
        errors=(),
        platform="windows-x86_64",
    ):
        paths = self.write_inputs(skipped=skipped, failed=failed, errors=errors)
        with self.assertRaisesRegex(inventory.InventoryError, "^" + expected.replace(".", "\\.") + "$"):
            inventory.verify_inventory(*paths, scope="root", platform=platform)

    def assert_inventory_failure_contains(self, expected, platform="windows-x86_64"):
        paths = self.write_inputs()
        with self.assertRaises(inventory.InventoryError) as raised:
            inventory.verify_inventory(*paths, scope="root", platform=platform)
        self.assertIn(expected, str(raised.exception))

    def test_missing_baseline_test_is_rejected(self):
        self.discovery["tests"] = [
            test for test in self.discovery["tests"] if test["name"] != "OvdLoaderTest.ValidCapture"
        ]
        self.assert_inventory_failure("missing baseline test: OvdLoaderTest.ValidCapture")

    def test_required_imported_test_must_exist_in_baseline(self):
        missing = "ImportedSuite.RequiredCase"
        self.baseline["ctest"]["required_imported_test_names"] = [missing]
        self.policy["ctest"]["required_imported_test_names"] = [missing]
        self.assert_inventory_failure(
            "required imported test is missing from baseline: %s" % missing
        )

    def test_enabled_baseline_test_cannot_be_disabled(self):
        for test in self.discovery["tests"]:
            if test["name"] == "OvdLoaderTest.ValidCapture":
                test["properties"] = [{"name": "DISABLED", "value": True}]
        self.assert_inventory_failure("enabled baseline test became disabled: OvdLoaderTest.ValidCapture")

    def test_unexpected_runtime_skip_is_rejected(self):
        self.discovery["tests"].append(discovery_entry("NewSuite.NewSkip"))
        self.assert_inventory_failure("unexpected runtime skip: NewSuite.NewSkip", skipped=["NewSuite.NewSkip"])

    def test_runtime_skip_count_is_bounded(self):
        fourth = {"name": "LegacySuite.FourthAllowedSkip", "reason": "fixture-only fourth skip"}
        self.baseline["ctest"]["allowed_runtime_skips"].append(fourth)
        self.policy["ctest"]["allowed_runtime_skips"].append(copy.deepcopy(fourth))
        self.discovery["tests"].append(discovery_entry(fourth["name"]))
        skipped = [entry["name"] for entry in self.baseline["ctest"]["allowed_runtime_skips"]]
        self.assert_inventory_failure("runtime skip count 4 exceeds maximum 3", skipped=skipped)

    def test_current_policy_rejects_retired_frontend_skip_on_both_platforms(self):
        baseline_path = Path(__file__).resolve().parents[1] / "baselines/qt-free/ovdnext-ci-parity.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        tests = baseline["ctest"]["legacy"]["tests"] + baseline["ctest"]["ovdnext_data_foundation"]["tests"]
        discovery = {"kind": "ctestInfo", "version": {"major": 1, "minor": 0},
                     "tests": [discovery_entry(test["name"], test["disabled"]) for test in tests]}
        discovery_path, junit_path = self.root / "current.json", self.root / "current.xml"
        discovery_path.write_text(json.dumps(discovery), encoding="utf-8")
        skips = [entry["name"] for entry in baseline["ctest"]["allowed_runtime_skips"]]
        write_junit(junit_path, discovery, skipped=skips)
        for platform in inventory.SUPPORTED_PLATFORMS:
            inventory.verify_inventory(baseline_path, discovery_path, junit_path, scope="root", platform=platform)
        retired = "ClipSaveUiTest.AccentedTemporaryDirectoryStillOpensPickerAndSavesClip"
        discovery["tests"].append(discovery_entry(retired))
        discovery_path.write_text(json.dumps(discovery), encoding="utf-8")
        write_junit(junit_path, discovery, skipped=[retired])
        for platform in inventory.SUPPORTED_PLATFORMS:
            with self.subTest(platform=platform), self.assertRaisesRegex(inventory.InventoryError, "unexpected runtime skip"):
                inventory.verify_inventory(baseline_path, discovery_path, junit_path, scope="root", platform=platform)

    def test_disabled_junit_failure_is_rejected(self):
        name = DISABLED_TESTS[0]
        self.assert_inventory_failure(
            "runtime test failed: %s" % name,
            failed=[name],
        )

    def test_disabled_junit_error_is_rejected(self):
        name = DISABLED_TESTS[0]
        self.assert_inventory_failure(
            "runtime test error: %s" % name,
            errors=[name],
        )

    def test_combined_registered_count_is_recomputed(self):
        self.baseline["ctest"]["combined"]["registered_count"] = 242
        self.assert_inventory_failure("baseline combined registered_count does not equal 243")

    def test_complete_named_inventory_without_failures_succeeds(self):
        paths = self.write_inputs()
        inventory.verify_inventory(*paths, scope="root", platform="windows-x86_64")

    def test_verify_cli_emits_concise_success_summary(self):
        paths = self.write_inputs()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            inventory.main(
                [
                    "verify",
                    "--baseline",
                    str(paths[0]),
                    "--discovery",
                    str(paths[1]),
                    "--junit",
                    str(paths[2]),
                    "--scope",
                    "root",
                    "--platform",
                    "windows-x86_64",
                ]
            )
        self.assertRegex(
            output.getvalue(),
            r"^PASS: inventory verified scope=root platform=windows-x86_64 "
            r"discovered=243 runtime=238 skipped=0\n$",
        )

    def test_verify_cli_failure_exits_nonzero_without_success_summary(self):
        self.discovery["tests"] = [
            test
            for test in self.discovery["tests"]
            if test["name"] != "OvdLoaderTest.ValidCapture"
        ]
        paths = self.write_inputs()
        output = io.StringIO()
        diagnostics = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(diagnostics):
            with self.assertRaises(SystemExit) as raised:
                inventory.main(
                    [
                        "verify",
                        "--baseline",
                        str(paths[0]),
                        "--discovery",
                        str(paths[1]),
                        "--junit",
                        str(paths[2]),
                        "--scope",
                        "root",
                        "--platform",
                        "windows-x86_64",
                    ]
                )
        self.assertEqual(1, raised.exception.code)
        self.assertEqual("", output.getvalue())
        self.assertIn(
            "ERROR: missing baseline test: OvdLoaderTest.ValidCapture",
            diagnostics.getvalue(),
        )

    def test_linux_applicable_inventory_can_omit_windows_only_tests(self):
        excluded = {
            entry["name"] for entry in PLATFORM_EXCLUSIONS["linux-x86_64"]
        }
        self.discovery["tests"] = [
            test for test in self.discovery["tests"] if test["name"] not in excluded
        ]
        paths = self.write_inputs()
        inventory.verify_inventory(*paths, scope="root", platform="linux-x86_64")

    def test_windows_inventory_cannot_omit_either_windows_only_test(self):
        complete_discovery = copy.deepcopy(self.discovery["tests"])
        for entry in PLATFORM_EXCLUSIONS["linux-x86_64"]:
            name = entry["name"]
            with self.subTest(name=name):
                self.discovery["tests"] = [
                    test for test in complete_discovery if test["name"] != name
                ]
                self.assert_inventory_failure("missing baseline test: %s" % name)

    def test_linux_inventory_cannot_omit_unrelated_test(self):
        self.discovery["tests"] = [
            test
            for test in self.discovery["tests"]
            if test["name"] != "OvdLoaderTest.ValidCapture"
        ]
        self.assert_inventory_failure(
            "missing baseline test: OvdLoaderTest.ValidCapture",
            platform="linux-x86_64",
        )

    def test_linux_excluded_test_if_present_cannot_be_disabled(self):
        name = PLATFORM_EXCLUSIONS["linux-x86_64"][0]["name"]
        for test in self.discovery["tests"]:
            if test["name"] == name:
                test["properties"] = [{"name": "DISABLED", "value": True}]
        self.assert_inventory_failure(
            "enabled baseline test became disabled: %s" % name,
            platform="linux-x86_64",
        )

    def test_linux_excluded_test_if_present_failure_is_rejected(self):
        name = PLATFORM_EXCLUSIONS["linux-x86_64"][0]["name"]
        self.assert_inventory_failure(
            "runtime test failed: %s" % name,
            failed=[name],
            platform="linux-x86_64",
        )

    def test_standalone_linux_still_requires_every_data_test(self):
        self.discovery["tests"] = [
            discovery_entry(name) for name in DATA_TESTS[1:]
        ]
        paths = self.write_inputs()
        with self.assertRaisesRegex(
            inventory.InventoryError,
            "^missing baseline test: ovddata_tests$",
        ):
            inventory.verify_inventory(
                *paths,
                scope="ovdnext-data",
                platform="linux-x86_64",
            )

    def test_unknown_platform_is_rejected(self):
        self.assert_inventory_failure(
            "unknown verification platform: unsupported-x86_64",
            platform="unsupported-x86_64",
        )

    def test_platform_exclusion_keys_must_equal_platform_scope(self):
        del self.baseline["ctest"]["platform_exclusions"]["windows-x86_64"]
        del self.policy["ctest"]["platform_exclusions"]["windows-x86_64"]
        self.assert_inventory_failure_contains(
            "platform_exclusions keys must equal platform_scope"
        )

    def test_policy_unknown_platform_is_rejected(self):
        for document in (self.baseline, self.policy):
            document["platform_scope"].append("unsupported-x86_64")
            document["ctest"]["platform_exclusions"]["unsupported-x86_64"] = []
        self.assert_inventory_failure_contains(
            "unsupported platform in policy platform_scope: unsupported-x86_64"
        )

    def test_duplicate_platform_exclusion_name_is_rejected(self):
        duplicate = copy.deepcopy(PLATFORM_EXCLUSIONS["linux-x86_64"][0])
        self.baseline["ctest"]["platform_exclusions"]["windows-x86_64"].append(
            duplicate
        )
        self.policy["ctest"]["platform_exclusions"]["windows-x86_64"].append(
            copy.deepcopy(duplicate)
        )
        self.assert_inventory_failure_contains("duplicate platform exclusion")

    def test_nonbaseline_platform_exclusion_is_rejected(self):
        unknown = {
            "name": "LegacySuite.NotInBaseline",
            "reason": "fixture-only invalid exclusion",
        }
        self.baseline["ctest"]["platform_exclusions"]["linux-x86_64"].append(
            unknown
        )
        self.policy["ctest"]["platform_exclusions"]["linux-x86_64"].append(
            copy.deepcopy(unknown)
        )
        self.assert_inventory_failure_contains("non-baseline platform exclusion")

    def test_data_test_platform_exclusion_is_rejected(self):
        data_exclusion = {
            "name": DATA_TESTS[0],
            "reason": "fixture-only invalid data exclusion",
        }
        self.baseline["ctest"]["platform_exclusions"]["linux-x86_64"].append(
            data_exclusion
        )
        self.policy["ctest"]["platform_exclusions"]["linux-x86_64"].append(
            copy.deepcopy(data_exclusion)
        )
        self.assert_inventory_failure_contains("data test cannot be platform-excluded")

    def test_platform_exclusion_requires_nonempty_reason(self):
        self.baseline["ctest"]["platform_exclusions"]["linux-x86_64"][0][
            "reason"
        ] = ""
        self.policy["ctest"]["platform_exclusions"]["linux-x86_64"][0][
            "reason"
        ] = ""
        self.assert_inventory_failure_contains("has no reason")

    def test_baseline_platform_exclusion_must_match_policy(self):
        self.baseline["ctest"]["platform_exclusions"]["linux-x86_64"][0][
            "reason"
        ] = "different rationale"
        self.assert_inventory_failure(
            "generated baseline policy fields differ from checked-in policy"
        )


class InventoryCliTests(unittest.TestCase):
    def assert_cli_error(self, arguments, expected):
        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics):
            with self.assertRaises(SystemExit) as raised:
                inventory._parser().parse_args(arguments)
        self.assertEqual(2, raised.exception.code)
        self.assertIn(expected, diagnostics.getvalue())

    def test_verify_cli_requires_platform(self):
        self.assert_cli_error(
            [
                "verify",
                "--baseline",
                "baseline.json",
                "--discovery",
                "discovery.json",
                "--junit",
                "junit.xml",
                "--scope",
                "root",
            ],
            "--platform",
        )

    def test_verify_cli_rejects_unknown_platform(self):
        self.assert_cli_error(
            [
                "verify",
                "--baseline",
                "baseline.json",
                "--discovery",
                "discovery.json",
                "--junit",
                "junit.xml",
                "--scope",
                "root",
                "--platform",
                "unsupported-x86_64",
            ],
            "invalid choice",
        )


class InventoryCaptureTests(unittest.TestCase):
    def test_capture_writes_literal_categories_and_policy_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = baseline_fixture()
            policy = policy_fixture(baseline)
            discovery = discovery_fixture(baseline)
            policy_path = root / "policy.json"
            discovery_path = root / "discovery.json"
            output_path = root / "baseline.json"
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            discovery_path.write_text(json.dumps(discovery), encoding="utf-8")

            inventory.capture_inventory(policy_path, discovery_path, output_path)

            actual = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(239, len(actual["ctest"]["legacy"]["tests"]))
            self.assertEqual(DATA_TESTS, [test["name"] for test in actual["ctest"]["ovdnext_data_foundation"]["tests"]])
            self.assertEqual(
                PLATFORM_EXCLUSIONS,
                actual["ctest"]["platform_exclusions"],
            )
            self.assertEqual(JOB_MAPPINGS, actual["pipeline"]["job_mappings"])

    def test_capture_writes_lf_without_platform_newline_translation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = baseline_fixture()
            policy_path = root / "policy.json"
            discovery_path = root / "discovery.json"
            output_path = root / "baseline.json"
            policy_path.write_text(json.dumps(policy_fixture(baseline)), encoding="utf-8")
            discovery_path.write_text(json.dumps(discovery_fixture(baseline)), encoding="utf-8")

            inventory.capture_inventory(policy_path, discovery_path, output_path)

            output_bytes = output_path.read_bytes()
            self.assertIn(b"\n", output_bytes)
            self.assertNotIn(b"\r\n", output_bytes)
            self.assertTrue(output_bytes.endswith(b"\n"))


class DependencyBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.imports = self.root / "imports.txt"
        self.imports.write_text("KERNEL32.dll\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def write_graph(self, extra_label=None):
        labels = [
            "ovdnext_data_probe",
            "OvdDataOmniPvd",
            "OvdData",
            "PVDRuntime",
            "Ws2_32",
            "Threads::Threads",
        ]
        if extra_label:
            labels.append(extra_label)
        lines = ["digraph targets {"]
        for index, label in enumerate(labels):
            lines.append('  "node%d" [ label = "%s" ];' % (index, label))
        lines.extend([
            '  "node0" -> "node1";',
            '  "node1" -> "node2";',
            '  "node1" -> "node3";',
            '  "node3" -> "node4";',
            '  "node2" -> "node5";',
        ])
        if extra_label:
            lines.append('  "node1" -> "node6";')
        lines.append("}")
        graph = self.root / "targets.dot"
        graph.write_text("\n".join(lines), encoding="utf-8")
        return graph

    def assert_dependency_failure(self, forbidden):
        graph = self.write_graph(forbidden)
        with self.assertRaises(dependencies.DependencyError) as raised:
            dependencies.verify_dependency_boundary(graph, "ovdnext_data_probe", self.imports)
        self.assertIn(forbidden, str(raised.exception))

    def test_pvdcore_dependency_is_rejected(self):
        self.assert_dependency_failure("PVDCore")

    def test_pvddom_dependency_is_rejected(self):
        self.assert_dependency_failure("PvdDom")

    def test_qt_dependency_is_rejected(self):
        self.assert_dependency_failure("Qt5::Core")

    def test_approved_dependency_closure_succeeds(self):
        graph = self.write_graph()
        dependencies.verify_dependency_boundary(graph, "ovdnext_data_probe", self.imports)


if __name__ == "__main__":
    unittest.main()
