# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the genuine pinned PhysX TCP fixture in a task-owned source export."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PIN = "ad5c86f498025752261284367f0f1908d392a6b1"
REQUIRED = ("physx/compiler/public/CMakeLists.txt", "physx/include/PxPhysicsAPI.h",
            "physx/source/physx/src/omnipvd/NpOmniPvd.cpp", "physx/pvdruntime/include/OmniPvdDefines.h")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_inputs(source, revision, config, manifest):
    if revision != PIN:
        raise ValueError("Unexpected sender revision; a pin change requires an explicit decision")
    if config != "checked":
        raise ValueError("The genuine OVD fixture requires checked configuration")
    data = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if data["revision"] != revision:
        raise ValueError("Source manifest revision differs from requested revision")
    seen = set()
    for item in data["files"]:
        relative = Path(item["path"])
        target = (source / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(source.resolve()) or item["path"] in seen:
            raise ValueError("Invalid or duplicate source manifest path")
        if not target.is_file() or sha256(target) != item["sha256"]:
            raise ValueError("Missing or changed SDK source: " + item["path"])
        seen.add(item["path"])
    if not set(REQUIRED) <= seen:
        raise ValueError("Full SDK source export is required, not the sparse reader dependency")
    return {"revision": revision, "manifest_sha256": sha256(manifest), "verified_files": len(seen)}


def validate_cuda(toolkit, manifest, *, windows):
    data = json.loads(manifest.read_text(encoding="utf-8-sig"))
    expected = "12.8.0-5-" + ("windows-x86_64" if windows else "manylinux_2_35_x86_64") + "-release"
    if data["package"] != expected:
        raise ValueError("Toolkit must match the pinned SDK CUDA dependency")
    seen = set()
    for item in data["files"]:
        relative = Path(item["path"])
        path = (toolkit / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(toolkit.resolve()) or item["path"] in seen:
            raise ValueError("Invalid or duplicate toolkit manifest path")
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError("Missing or changed toolkit input: " + item["path"])
        seen.add(item["path"])
    required = {"include/cuda.h", "bin/nvcc.exe", "lib/x64/cuda.lib"} if windows else {"include/cuda.h", "bin/nvcc"}
    if not required <= seen:
        raise ValueError("Incomplete toolkit input manifest")
    return {"package": expected, "manifest_sha256": sha256(manifest), "verified_files": len(seen)}


def configure_command(source, work, *, windows, cuda=None):
    return ["cmake", "-S", str(ROOT / "examples/tcp_selection_sender"), "-B", str(work / "build"),
            "-G", "Visual Studio 17 2022" if windows else "Ninja",
            *(["-A", "x64", "-DCMAKE_SYSTEM_VERSION=10.0.26100.0"]
              if windows else ["-DCMAKE_BUILD_TYPE=checked"]),
            "-DPHYSX_ROOT_DIR=" + (source / "physx").as_posix(),
            "-DTARGET_BUILD_PLATFORM=" + ("windows" if windows else "linux"),
            "-DPX_OUTPUT_LIB_DIR=" + (work / "lib").as_posix(),
            "-DPX_OUTPUT_BIN_DIR=" + (work / "bin").as_posix(),
            "-DPX_BUILDPVDRUNTIME=ON", "-DPX_GENERATE_GPU_PROJECTS=" + ("ON" if cuda else "OFF"),
            "-DPX_GENERATE_STATIC_LIBRARIES=ON", "-DPX_BUILDSNIPPETS=OFF",
            "-DPX_BUILDUNITTESTS=OFF", "-DNV_USE_STATIC_WINCRT=OFF", "-DPX_CMAKE_SUPPRESS_REGENERATION=ON",
            *(["-DOVDNEXT_SENDER_GPU=ON", "-DCUDAToolkit_ROOT=" + cuda.as_posix(),
               "-DPX_GENERATE_GPU_REDUCED_ARCHITECTURES=ON", "-DCMAKE_CUDA_ARCHITECTURES=86",
               *(["-T", "cuda=" + cuda.as_posix()] if windows else
                 ["-DCMAKE_CUDA_COMPILER=" + (cuda / "bin/nvcc").as_posix()])] if cuda else [])]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-source", required=True, type=Path)
    parser.add_argument("--sdk-revision", required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--config", default="checked")
    parser.add_argument("--cuda-toolkit", type=Path, help="Opt in to a separate GPU-capable fixture build")
    parser.add_argument("--cuda-manifest", type=Path)
    args = parser.parse_args(argv)
    source, work = args.sdk_source.resolve(), args.work_root.resolve()
    manifest = args.source_manifest or source.parent / "source-manifest.json"
    evidence = validate_inputs(source, args.sdk_revision, args.config, manifest)
    version = subprocess.check_output(["cmake", "--version"], text=True).splitlines()[0]
    if version != "cmake version 3.25.2":
        raise ValueError("Use pinned CMake 3.25.2, found " + version)
    windows = os.name == "nt"
    cuda = args.cuda_toolkit.resolve() if args.cuda_toolkit else None
    if cuda:
        evidence["cuda"] = validate_cuda(cuda, args.cuda_manifest or cuda.parent / "input-report.json", windows=windows)
        evidence["nvcc"] = subprocess.check_output([str(cuda / "bin" / ("nvcc.exe" if windows else "nvcc")), "--version"], text=True)
    work.mkdir(parents=True, exist_ok=True)
    commands = [configure_command(source, work, windows=windows, cuda=cuda),
                ["cmake", "--build", str(work / "build"), "--config", "checked", "--parallel", "2" if cuda else "4",
                 "--target", "ovd_tcp_selection_sender"]]
    for label, command in zip(("configure", "build"), commands):
        with (work / (label + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    binary = work / "bin" / ("ovd_tcp_selection_sender.exe" if windows else "ovd_tcp_selection_sender")
    capability = json.loads(subprocess.check_output([str(binary), "--describe"], text=True))
    if capability["ovd_capable"] is not True or capability["wire"] != [0, 4, 0] or capability["integration"] != [3, 5]:
        raise ValueError("Built fixture does not match the required OVD capability/version")
    if cuda and not capability.get("gpu_capable"):
        raise ValueError("GPU fixture was not built GPU capable")
    evidence.update({"commands": commands, "cmake": version, "os": platform.platform(),
                     "capability": capability, "binary": str(binary), "binary_sha256": sha256(binary),
                     "generated_pxconfig_sha256": sha256(source / "physx/include/PxConfig.h"),
                     "cache_sha256": sha256(work / "build/CMakeCache.txt"),
                     "compiler_files": {str(p.relative_to(work)): {"sha256": sha256(p), "content": p.read_text()}
                                        for p in (work / "build/CMakeFiles").glob("*/CMake*Compiler.cmake")},
                     "dependent_binaries": {str(p.relative_to(work)): sha256(p) for p in (work / "bin").rglob("*PhysXGpu*")
                                            if p.suffix in (".dll", ".so")}})
    (work / "build-report.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"binary": str(binary), "binary_sha256": evidence["binary_sha256"]}))


if __name__ == "__main__":
    main()
