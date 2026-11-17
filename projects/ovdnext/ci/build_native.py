#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and verify the native inspector without a desktop toolkit SDK."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

TARGETS = ("pvdcore_tests", "checked_output_tests", "ovd_capture_catalog_tests",
           "ovd_capture_control_tests", "ovddata_tests", "ovddata_ingest_tests",
           "ovddata_omnipvd_tests", "ovdnext_data_probe", "ovdnext_data_probe_tests")


def git_head(root):
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip().lower()


def run(command, log=None):
    command = [str(value) for value in command]
    if log is None:
        subprocess.run(command, check=True)
    else:
        with Path(log).open("w", encoding="utf-8") as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True, text=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "build-root", "physics-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--physics-ref", default="a12f79ce585ffb79e5ffa4848cb45efc3de2f632")
    parser.add_argument("--configuration", choices=["RelWithDebInfo"], default="RelWithDebInfo")
    parser.add_argument("--generator", default="Ninja")
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--ctest", default="ctest")
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args(argv)
    source, build, physics = (path.resolve() for path in (args.source_root, args.build_root, args.physics_root))
    for label, root, commit in (("Source", source, args.source_commit), ("Physics", physics, args.physics_ref)):
        if not re.fullmatch(r"[0-9a-fA-F]{40}", commit): raise ValueError(f"{label} commit must be an exact SHA")
        if git_head(root) != commit.lower(): raise ValueError(f"{label} HEAD does not match requested commit")
    runtime, dom = physics / "physx/pvdruntime", physics / "physx/pvddom"
    baseline = source / "ci/baselines/qt-free/ovdnext-ci-parity.json"
    verifier = source / "ci/verify_test_inventory.py"
    for path in (source / "CMakeLists.txt", runtime / "compiler/cmake/PVDRuntime.cmake",
                 dom / "CMakeLists.txt", baseline, verifier):
        if not path.is_file(): raise ValueError(f"Required input is missing: {path}")
    build.mkdir(parents=True, exist_ok=True)
    run([args.cmake, "-S", source, "-B", build, "-G", args.generator,
         f"-DCMAKE_BUILD_TYPE={args.configuration}", "-DPVD3_BUILD_PYTHON=OFF",
         "-DPVD3_BUILD_TESTS=ON", "-DCMAKE_DISABLE_FIND_PACKAGE_Qt5=ON",
         "-DPVD3_REQUIRE_OVD_SUPPORT=ON", f"-DPVDRUNTIME_ROOT={runtime}", f"-DPVDDOM_ROOT={dom}"], build / "configure.log")
    run([args.cmake, "--build", build, "--config", args.configuration, "--target", *TARGETS, "--parallel"])
    discovery, junit = build / "ctest-discovery.json", build / "ctest.junit.xml"
    ctest = [args.ctest, "--test-dir", build, "-C", args.configuration]
    run([*ctest, "--show-only=json-v1"], discovery)
    run([*ctest, "--no-tests=error", "--output-on-failure", "--output-junit", junit])
    platform = "windows-x86_64" if os.name == "nt" else "linux-x86_64"
    run([sys.executable, verifier, "verify", "--baseline", baseline, "--discovery", discovery,
         "--junit", junit, "--scope", "root", "--platform", platform], build / "inventory-verifier.log")
    if args.artifact_root:
        args.artifact_root.mkdir(parents=True, exist_ok=True)
        (args.artifact_root / "build-info.json").write_text(json.dumps({
            "source_commit": args.source_commit.lower(), "physics_ref": args.physics_ref.lower(),
            "configuration": args.configuration, "platform": platform, "targets": TARGETS,
            "scope": "native-verification"}, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    try: main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Native verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
