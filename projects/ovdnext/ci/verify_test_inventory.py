#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture and verify the auditable OvdNext/OVDChimera CTest inventory."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


class InventoryError(Exception):
    pass


SUPPORTED_PLATFORMS = ("windows-x86_64", "linux-x86_64")


def _read_json(path, description):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise InventoryError("unable to read %s: %s" % (description, error))
    if not isinstance(value, dict):
        raise InventoryError("%s must be a JSON object" % description)
    return value


def _require_dict(parent, key, description):
    value = parent.get(key)
    if not isinstance(value, dict):
        raise InventoryError("%s %s must be an object" % (description, key))
    return value


def _require_list(parent, key, description):
    value = parent.get(key)
    if not isinstance(value, list):
        raise InventoryError("%s %s must be an array" % (description, key))
    return value


def _require_count(parent, key, description):
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InventoryError("%s %s must be a nonnegative integer" % (description, key))
    return value


def _validate_named_strings(values, description):
    seen = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise InventoryError("%s contains an empty or non-string name" % description)
        if value in seen:
            raise InventoryError("%s contains duplicate name: %s" % (description, value))
        seen.add(value)
    return values


def _validate_platform_exclusions(
        ctest, platform_scope, data_names, baseline_legacy_names=None):
    exclusions = _require_dict(ctest, "platform_exclusions", "policy ctest")
    if set(exclusions) != set(platform_scope):
        raise InventoryError(
            "policy platform_exclusions keys must equal platform_scope")

    data_names = set(data_names)
    seen = set()
    for platform in platform_scope:
        entries = _require_list(
            exclusions, platform, "policy ctest platform_exclusions")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"name", "reason"}:
                raise InventoryError(
                    "policy platform exclusion for %s must contain only name and reason"
                    % platform)
            name = entry.get("name")
            reason = entry.get("reason")
            if not isinstance(name, str) or not name:
                raise InventoryError(
                    "policy platform exclusion for %s has an invalid name" % platform)
            if not isinstance(reason, str) or not reason.strip():
                raise InventoryError("policy platform exclusion %s has no reason" % name)
            if name in seen:
                raise InventoryError("duplicate platform exclusion: %s" % name)
            seen.add(name)
            if name in data_names:
                raise InventoryError(
                    "data test cannot be platform-excluded: %s" % name)
            if baseline_legacy_names is not None and name not in baseline_legacy_names:
                raise InventoryError(
                    "non-baseline platform exclusion: %s" % name)
    return exclusions


def _validate_policy(policy):
    if policy.get("schema_version") != 1:
        raise InventoryError("policy schema_version must equal 1")
    revisions = _require_dict(policy, "source_revisions", "policy")
    for key in ("legacy_tests_commit", "ovdnext_data_foundation_commit", "combined_inventory_commit"):
        value = revisions.get(key)
        if not isinstance(value, str) or len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise InventoryError("policy source_revisions %s must be a lowercase 40-character commit" % key)
    if not isinstance(policy.get("required_configuration"), str) or not policy["required_configuration"]:
        raise InventoryError("policy required_configuration must be a nonempty string")
    platform_scope = _validate_named_strings(
        _require_list(policy, "platform_scope", "policy"),
        "policy platform_scope")
    unsupported_platforms = [
        platform for platform in platform_scope
        if platform not in SUPPORTED_PLATFORMS
    ]
    if unsupported_platforms:
        raise InventoryError(
            "unsupported platform in policy platform_scope: %s"
            % unsupported_platforms[0])
    if set(platform_scope) != set(SUPPORTED_PLATFORMS):
        raise InventoryError(
            "policy platform_scope must equal supported verification platforms")

    ctest = _require_dict(policy, "ctest", "policy")
    for category_name in ("legacy", "ovdnext_data_foundation", "combined"):
        category = _require_dict(ctest, category_name, "policy ctest")
        registered = _require_count(category, "registered_count", "policy ctest %s" % category_name)
        enabled = _require_count(category, "enabled_count", "policy ctest %s" % category_name)
        disabled = _require_count(category, "disabled_count", "policy ctest %s" % category_name)
        if enabled + disabled != registered:
            raise InventoryError("policy ctest %s enabled_count plus disabled_count does not equal registered_count" % category_name)
    legacy = ctest["legacy"]
    data = ctest["ovdnext_data_foundation"]
    combined = ctest["combined"]
    for key in ("registered_count", "enabled_count", "disabled_count"):
        if combined[key] != legacy[key] + data[key]:
            raise InventoryError("policy combined %s does not equal legacy plus OvdNext Data" % key)

    disabled_tests = _validate_named_strings(
        _require_list(ctest, "disabled_tests", "policy ctest"), "policy disabled_tests")
    if len(disabled_tests) != legacy["disabled_count"]:
        raise InventoryError("policy disabled_tests count does not equal legacy disabled_count")
    data_names = _validate_named_strings(
        _require_list(ctest, "ovdnext_data_test_names", "policy ctest"),
        "policy ovdnext_data_test_names")
    if len(data_names) != data["registered_count"]:
        raise InventoryError("policy OvdNext Data names count does not equal registered_count")
    _validate_named_strings(
        _require_list(
            ctest,
            "required_imported_test_names",
            "policy ctest"),
        "policy required_imported_test_names")
    _validate_platform_exclusions(ctest, platform_scope, data_names)

    skips = _require_list(ctest, "allowed_runtime_skips", "policy ctest")
    skip_names = []
    for entry in skips:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"]:
            raise InventoryError("policy allowed_runtime_skips contains an invalid name")
        if not isinstance(entry.get("reason"), str) or not entry["reason"]:
            raise InventoryError("policy runtime skip %s has no reason" % entry["name"])
        skip_names.append(entry["name"])
    _validate_named_strings(skip_names, "policy allowed_runtime_skips")
    _require_count(ctest, "maximum_runtime_skip_count", "policy ctest")

    pipeline = _require_dict(policy, "pipeline", "policy")
    legacy_job_count = _require_count(pipeline, "legacy_job_count", "policy pipeline")
    github_job_count = _require_count(pipeline, "github_job_count", "policy pipeline")
    blocking_count = _require_count(
        pipeline, "pull_request_blocking_job_count", "policy pipeline")
    mappings = _require_list(pipeline, "job_mappings", "policy pipeline")
    if len(mappings) != legacy_job_count:
        raise InventoryError("policy job_mappings count does not equal legacy_job_count")
    github_ids = []
    legacy_ids = []
    for mapping in mappings:
        if not isinstance(mapping, dict):
            raise InventoryError("policy job mapping must be an object")
        legacy_id = mapping.get("legacy_id")
        category = mapping.get("category")
        disposition = mapping.get("disposition")
        if not isinstance(legacy_id, str) or not legacy_id:
            raise InventoryError("policy job mapping has an invalid legacy_id")
        if not isinstance(category, str) or not category:
            raise InventoryError("policy job mapping %s has an invalid category" % legacy_id)
        if disposition not in ("preserved", "expanded", "migrated"):
            raise InventoryError("invalid job mapping disposition: %s" % disposition)
        mapped_ids = _validate_named_strings(
            _require_list(mapping, "github_ids", "policy job mapping %s" % legacy_id),
            "policy job mapping %s github_ids" % legacy_id)
        if not mapped_ids:
            raise InventoryError("policy job mapping %s has no GitHub jobs" % legacy_id)
        legacy_ids.append(legacy_id)
        github_ids.extend(mapped_ids)
    _validate_named_strings(legacy_ids, "policy legacy job mappings")
    _validate_named_strings(github_ids, "policy GitHub job mappings")
    if len(github_ids) != github_job_count:
        raise InventoryError("policy mapped GitHub job count does not equal github_job_count")
    blocking_jobs = _validate_named_strings(
        _require_list(pipeline, "pull_request_blocking_jobs", "policy pipeline"),
        "policy pull_request_blocking_jobs")
    if len(blocking_jobs) != blocking_count:
        raise InventoryError("policy blocking job list does not equal pull_request_blocking_job_count")
    for job in blocking_jobs:
        if job not in github_ids:
            raise InventoryError("policy blocking job is not mapped: %s" % job)


def _discovery_tests(discovery):
    if discovery.get("kind") != "ctestInfo":
        raise InventoryError("CTest discovery kind must equal ctestInfo")
    raw_tests = _require_list(discovery, "tests", "CTest discovery")
    if not raw_tests:
        raise InventoryError("CTest discovery tests array must not be empty")
    result = []
    seen = set()
    for raw in raw_tests:
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str) or not raw["name"]:
            raise InventoryError("CTest discovery contains an invalid test name")
        name = raw["name"]
        if name in seen:
            raise InventoryError("CTest discovery contains duplicate test: %s" % name)
        seen.add(name)
        disabled = False
        properties = raw.get("properties", [])
        if not isinstance(properties, list):
            raise InventoryError("CTest discovery properties must be an array for %s" % name)
        for prop in properties:
            if not isinstance(prop, dict):
                raise InventoryError("CTest discovery property must be an object for %s" % name)
            if str(prop.get("name", "")).upper() == "DISABLED":
                value = prop.get("value")
                disabled = value is True or str(value).upper() in ("1", "ON", "TRUE", "YES")
        result.append({"name": name, "disabled": disabled})
    return result


def _baseline_test_list(category, category_name):
    tests = _require_list(category, "tests", "baseline ctest %s" % category_name)
    if not tests:
        raise InventoryError("baseline ctest %s tests array must not be empty" % category_name)
    result = []
    seen = set()
    for test in tests:
        if not isinstance(test, dict) or set(test) != {"name", "disabled"}:
            raise InventoryError("baseline ctest %s test entries must contain only name and disabled" % category_name)
        name = test.get("name")
        disabled = test.get("disabled")
        if not isinstance(name, str) or not name or not isinstance(disabled, bool):
            raise InventoryError("baseline ctest %s contains an invalid test entry" % category_name)
        if name in seen:
            raise InventoryError("baseline contains duplicate test: %s" % name)
        seen.add(name)
        result.append({"name": name, "disabled": disabled})
    return result


def _policy_projection(baseline):
    ctest = baseline["ctest"]
    return {
        "schema_version": baseline["schema_version"],
        "source_revisions": baseline["source_revisions"],
        "required_configuration": baseline["required_configuration"],
        "platform_scope": baseline["platform_scope"],
        "ctest": {
            "legacy": {key: ctest["legacy"][key] for key in ("registered_count", "enabled_count", "disabled_count")},
            "ovdnext_data_foundation": {
                key: ctest["ovdnext_data_foundation"][key]
                for key in ("registered_count", "enabled_count", "disabled_count")
            },
            "combined": {key: ctest["combined"][key] for key in ("registered_count", "enabled_count", "disabled_count")},
            "disabled_tests": [test["name"] for test in ctest["legacy"]["tests"] if test["disabled"]],
            "ovdnext_data_test_names": [test["name"] for test in ctest["ovdnext_data_foundation"]["tests"]],
            "required_imported_test_names": ctest["required_imported_test_names"],
            "allowed_runtime_skips": ctest["allowed_runtime_skips"],
            "maximum_runtime_skip_count": ctest["maximum_runtime_skip_count"],
            "platform_exclusions": ctest["platform_exclusions"],
        },
        "pipeline": baseline["pipeline"],
    }


def _validate_baseline(baseline, policy):
    baseline_ctest = _require_dict(baseline, "ctest", "baseline")
    legacy = _require_dict(baseline_ctest, "legacy", "baseline ctest")
    data = _require_dict(baseline_ctest, "ovdnext_data_foundation", "baseline ctest")
    combined = _require_dict(baseline_ctest, "combined", "baseline ctest")
    legacy_tests = _baseline_test_list(legacy, "legacy")
    data_tests = _baseline_test_list(data, "ovdnext_data_foundation")
    all_names = [test["name"] for test in legacy_tests + data_tests]
    _validate_named_strings(all_names, "baseline combined tests")
    required_imported_names = _validate_named_strings(
        _require_list(
            baseline_ctest,
            "required_imported_test_names",
            "baseline ctest"),
        "baseline required_imported_test_names")
    legacy_by_name = {test["name"]: test for test in legacy_tests}
    for name in required_imported_names:
        test = legacy_by_name.get(name)
        if test is None:
            raise InventoryError(
                "required imported test is missing from baseline: %s" % name)
        if test["disabled"]:
            raise InventoryError(
                "required imported test is disabled in baseline: %s" % name)
    _validate_platform_exclusions(
        baseline_ctest,
        baseline["platform_scope"],
        {test["name"] for test in data_tests},
        {test["name"] for test in legacy_tests})

    derived = {}
    for name, category, tests in (("legacy", legacy, legacy_tests), ("ovdnext_data_foundation", data, data_tests)):
        actual_registered = len(tests)
        actual_disabled = len([test for test in tests if test["disabled"]])
        actual_enabled = actual_registered - actual_disabled
        derived[name] = {
            "registered_count": actual_registered,
            "enabled_count": actual_enabled,
            "disabled_count": actual_disabled,
        }
        for key, actual in derived[name].items():
            stored = _require_count(category, key, "baseline ctest %s" % name)
            if stored != actual:
                raise InventoryError("baseline %s %s does not equal %d" % (name, key, actual))
    combined_derived = {
        key: derived["legacy"][key] + derived["ovdnext_data_foundation"][key]
        for key in ("registered_count", "enabled_count", "disabled_count")
    }
    for key, actual in combined_derived.items():
        stored = _require_count(combined, key, "baseline ctest combined")
        if stored != actual:
            raise InventoryError("baseline combined %s does not equal %d" % (key, actual))

    _require_list(baseline["ctest"], "allowed_runtime_skips", "baseline ctest")
    _require_count(baseline["ctest"], "maximum_runtime_skip_count", "baseline ctest")
    _require_dict(baseline, "pipeline", "baseline")
    projection = _policy_projection(baseline)
    _validate_policy(projection)
    if projection != policy:
        raise InventoryError("generated baseline policy fields differ from checked-in policy")
    return legacy_tests, data_tests


def _read_junit(path):
    try:
        root = ET.parse(str(path)).getroot()
    except (OSError, ET.ParseError) as error:
        raise InventoryError("unable to read CTest JUnit: %s" % error)
    results = {}
    for case in root.iter("testcase"):
        name = case.get("name")
        if not name:
            raise InventoryError("CTest JUnit contains a testcase without a name")
        if name in results:
            raise InventoryError("CTest JUnit contains duplicate testcase: %s" % name)
        status = "passed"
        if case.find("failure") is not None:
            status = "failed"
        elif case.find("error") is not None:
            status = "error"
        elif case.find("skipped") is not None:
            status = "skipped"
        results[name] = status
    return results


def capture_inventory(policy_path, discovery_path, output_path):
    policy = _read_json(policy_path, "inventory policy")
    _validate_policy(policy)
    discovered = _discovery_tests(_read_json(discovery_path, "CTest discovery"))
    ctest_policy = policy["ctest"]
    data_names = ctest_policy["ovdnext_data_test_names"]
    discovered_by_name = {test["name"]: test for test in discovered}
    for name in data_names:
        if name not in discovered_by_name:
            raise InventoryError("missing OvdNext Data foundation test: %s" % name)
    data_name_set = set(data_names)
    legacy_tests = [test for test in discovered if test["name"] not in data_name_set]
    data_tests = [discovered_by_name[name] for name in data_names]

    actual_disabled = [test["name"] for test in legacy_tests if test["disabled"]]
    if actual_disabled != ctest_policy["disabled_tests"]:
        raise InventoryError("discovered disabled tests differ from policy")
    for category_name, tests in (("legacy", legacy_tests), ("ovdnext_data_foundation", data_tests)):
        expected = ctest_policy[category_name]
        actual_registered = len(tests)
        actual_disabled_count = len([test for test in tests if test["disabled"]])
        actual_enabled = actual_registered - actual_disabled_count
        for key, actual in (
            ("registered_count", actual_registered),
            ("enabled_count", actual_enabled),
            ("disabled_count", actual_disabled_count),
        ):
            if expected[key] != actual:
                raise InventoryError("discovery %s %s does not equal policy" % (category_name, key))
    combined_actual = {
        "registered_count": len(discovered),
        "enabled_count": len([test for test in discovered if not test["disabled"]]),
        "disabled_count": len([test for test in discovered if test["disabled"]]),
    }
    if combined_actual != ctest_policy["combined"]:
        raise InventoryError("discovery combined counts do not equal policy")

    baseline = {
        "schema_version": policy["schema_version"],
        "source_revisions": policy["source_revisions"],
        "required_configuration": policy["required_configuration"],
        "platform_scope": policy["platform_scope"],
        "ctest": {
            "legacy": dict(ctest_policy["legacy"], tests=legacy_tests),
            "ovdnext_data_foundation": dict(ctest_policy["ovdnext_data_foundation"], tests=data_tests),
            "combined": ctest_policy["combined"],
            "allowed_runtime_skips": ctest_policy["allowed_runtime_skips"],
            "maximum_runtime_skip_count": ctest_policy["maximum_runtime_skip_count"],
            "platform_exclusions": ctest_policy["platform_exclusions"],
            "required_imported_test_names": ctest_policy["required_imported_test_names"],
        },
        "pipeline": policy["pipeline"],
    }
    _validate_baseline(baseline, policy)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(baseline, indent=2) + "\n").encode("utf-8"))


def verify_inventory(baseline_path, discovery_path, junit_path, scope, platform):
    baseline_path = Path(baseline_path)
    policy_path = baseline_path.parent / "ovdnext-ci-parity-policy.json"
    policy = _read_json(policy_path, "checked-in inventory policy")
    _validate_policy(policy)
    baseline = _read_json(baseline_path, "test inventory baseline")
    legacy_tests, data_tests = _validate_baseline(baseline, policy)
    if platform not in baseline["platform_scope"]:
        raise InventoryError("unknown verification platform: %s" % platform)
    discovered = _discovery_tests(_read_json(discovery_path, "CTest discovery"))
    discovered_by_name = {test["name"]: test for test in discovered}

    if scope == "root":
        excluded_names = {
            entry["name"]
            for entry in baseline["ctest"]["platform_exclusions"][platform]
        }
        required = legacy_tests + data_tests
        for test in required:
            actual = discovered_by_name.get(test["name"])
            if actual is None:
                if test["name"] in excluded_names:
                    continue
                raise InventoryError("missing baseline test: %s" % test["name"])
            if not test["disabled"] and actual["disabled"]:
                raise InventoryError("enabled baseline test became disabled: %s" % test["name"])
        required_names = {test["name"] for test in required}
        for test in discovered:
            if test["name"] not in required_names and test["disabled"]:
                raise InventoryError("new test is disabled: %s" % test["name"])
    elif scope == "ovdnext-data":
        for test in data_tests:
            actual = discovered_by_name.get(test["name"])
            if actual is None:
                raise InventoryError("missing baseline test: %s" % test["name"])
            if actual["disabled"]:
                raise InventoryError("enabled baseline test became disabled: %s" % test["name"])
        for test in discovered:
            if not (test["name"].startswith("ovddata") or test["name"].startswith("ovdnext_data")):
                raise InventoryError("unexpected standalone test: %s" % test["name"])
            if test["disabled"]:
                raise InventoryError("new test is disabled: %s" % test["name"])
    else:
        raise InventoryError("scope must be root or ovdnext-data")

    results = _read_junit(junit_path)
    for name, status in results.items():
        if name not in discovered_by_name:
            raise InventoryError("unexpected runtime result: %s" % name)
        if status in ("failed", "error"):
            raise InventoryError("runtime test %s: %s" % (status, name))
    for test in discovered:
        if test["disabled"]:
            continue
        status = results.get(test["name"])
        if status is None:
            raise InventoryError("missing runtime result: %s" % test["name"])

    skipped = [name for name, status in results.items() if status == "skipped"]
    allowed = {
        entry["name"] for entry in baseline["ctest"]["allowed_runtime_skips"]
    }
    for name in skipped:
        if name not in allowed:
            raise InventoryError("unexpected runtime skip: %s" % name)
    maximum = baseline["ctest"]["maximum_runtime_skip_count"]
    if len(skipped) > maximum:
        raise InventoryError(
            "runtime skip count %d exceeds maximum %d" % (len(skipped), maximum))
    return {
        "discovered": len(discovered),
        "runtime": len(results),
        "skipped": len(skipped),
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--policy", required=True)
    capture.add_argument("--discovery", required=True)
    capture.add_argument("--output", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--baseline", required=True)
    verify.add_argument("--discovery", required=True)
    verify.add_argument("--junit", required=True)
    verify.add_argument("--scope", required=True, choices=("root", "ovdnext-data"))
    verify.add_argument(
        "--platform",
        required=True,
        choices=SUPPORTED_PLATFORMS)
    return parser


def main(arguments=None):
    parser = _parser()
    args = parser.parse_args(arguments)
    try:
        if args.mode == "capture":
            capture_inventory(args.policy, args.discovery, args.output)
        else:
            summary = verify_inventory(
                args.baseline,
                args.discovery,
                args.junit,
                args.scope,
                args.platform)
            print(
                "PASS: inventory verified scope=%s platform=%s "
                "discovered=%d runtime=%d skipped=%d"
                % (
                    args.scope,
                    args.platform,
                    summary["discovered"],
                    summary["runtime"],
                    summary["skipped"],
                )
            )
    except InventoryError as error:
        parser.exit(1, "ERROR: %s\n" % error)


if __name__ == "__main__":
    main()
