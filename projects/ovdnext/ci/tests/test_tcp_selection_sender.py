# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The real-SDK fixture recipe must reject unverifiable or OVD-disabled inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("sender_build", ROOT / "ci/build_tcp_selection_sender.py")
build = importlib.util.module_from_spec(SPEC)
if Path(SPEC.origin).exists():
    SPEC.loader.exec_module(build)
PIN = "ad5c86f498025752261284367f0f1908d392a6b1"


def source_fixture(tmp_path):
    root = tmp_path / "source"
    paths = ["physx/compiler/public/CMakeLists.txt", "physx/include/PxPhysicsAPI.h",
             "physx/source/physx/src/omnipvd/NpOmniPvd.cpp", "physx/pvdruntime/include/OmniPvdDefines.h"]
    files = []
    for path in paths:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture input\n")
        files.append({"path": path, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(json.dumps({"revision": PIN, "files": files}))
    return root, manifest


@pytest.mark.parametrize("revision,config,error", [("0" * 40, "checked", "revision"),
                                                   (PIN, "release", "checked")])
def test_disabled_or_wrong_revision_rejected_before_build(tmp_path, revision, config, error):
    assert hasattr(build, "validate_inputs"), "Pinned genuine sender recipe not implemented"
    root, manifest = source_fixture(tmp_path)
    with pytest.raises(ValueError, match=error):
        build.validate_inputs(root, revision, config, manifest)


@pytest.mark.parametrize("damage", ["changed", "missing", "manifest_revision", "escape"])
def test_unverifiable_source_rejected(tmp_path, damage):
    assert hasattr(build, "validate_inputs"), "Pinned genuine sender recipe not implemented"
    root, manifest = source_fixture(tmp_path)
    data = json.loads(manifest.read_text())
    if damage == "changed":
        (root / data["files"][0]["path"]).write_bytes(b"changed")
    elif damage == "missing":
        (root / data["files"][0]["path"]).unlink()
    elif damage == "manifest_revision":
        data["revision"] = "0" * 40
    else:
        data["files"][0]["path"] = "../outside"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        build.validate_inputs(root, PIN, "checked", manifest)


@pytest.mark.parametrize("windows", [True, False])
def test_cpu_recipe_enables_checked_ovd_and_isolates_outputs(tmp_path, windows):
    assert hasattr(build, "configure_command"), "Pinned genuine sender recipe not implemented"
    source, work = tmp_path / "sdk source", tmp_path / "build work"
    args = build.configure_command(source, work, windows=windows)
    for option in ("-DPX_BUILDPVDRUNTIME=ON", "-DPX_GENERATE_GPU_PROJECTS=OFF",
                   "-DPX_GENERATE_STATIC_LIBRARIES=ON", "-DPX_BUILDSNIPPETS=OFF",
                   "-DPX_BUILDUNITTESTS=OFF", "-DNV_USE_STATIC_WINCRT=OFF",
                   "-DPX_CMAKE_SUPPRESS_REGENERATION=ON"):
        assert option in args
    assert "-DPHYSX_ROOT_DIR=" + (source / "physx").as_posix() in args
    assert "-DPX_OUTPUT_LIB_DIR=" + (work / "lib").as_posix() in args
    assert "-DPX_OUTPUT_BIN_DIR=" + (work / "bin").as_posix() in args
    assert args[args.index("-G") + 1] == ("Visual Studio 17 2022" if windows else "Ninja")
    if not windows:
        assert "-DCMAKE_BUILD_TYPE=checked" in args


def test_verified_source_returns_manifest_identity(tmp_path):
    assert hasattr(build, "validate_inputs"), "Pinned genuine sender recipe not implemented"
    root, manifest = source_fixture(tmp_path)
    evidence = build.validate_inputs(root, PIN, "checked", manifest)
    assert evidence["revision"] == PIN
    assert evidence["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert evidence["verified_files"] == 4


@pytest.mark.parametrize("windows", [True, False])
def test_gpu_recipe_uses_explicit_task_toolkit_without_changing_cpu_defaults(tmp_path, windows):
    toolkit = tmp_path / "cuda input"
    args = build.configure_command(tmp_path / "sdk", tmp_path / "gpu work", windows=windows, cuda=toolkit)
    assert "-DPX_GENERATE_GPU_PROJECTS=ON" in args and "-DPX_GENERATE_GPU_PROJECTS=OFF" not in args
    assert "-DOVDNEXT_SENDER_GPU=ON" in args and "-DCUDAToolkit_ROOT=" + toolkit.as_posix() in args
    assert "-DPX_GENERATE_GPU_REDUCED_ARCHITECTURES=ON" in args
    if windows:
        assert args[args.index("-T") + 1] == "cuda=" + toolkit.as_posix()
    else:
        assert "-DCMAKE_CUDA_COMPILER=" + (toolkit / "bin/nvcc").as_posix() in args


@pytest.mark.parametrize("damage", [None, "changed", "missing", "version", "escape"])
def test_gpu_toolkit_manifest_is_validated_before_compilation(tmp_path, damage):
    toolkit = tmp_path / "cuda"
    files = []
    for name in ("bin/nvcc.exe", "include/cuda.h", "lib/x64/cuda.lib"):
        path = toolkit / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"verified fixture input")
        files.append({"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    data = {"package": "12.8.0-5-windows-x86_64-release", "files": files}
    if damage == "changed": (toolkit / "include/cuda.h").write_bytes(b"changed")
    if damage == "missing": (toolkit / "include/cuda.h").unlink()
    if damage == "version": data["package"] = "12.4.1-1-windows-x86_64-release"
    if damage == "escape": files[0]["path"] = "../outside"
    manifest = tmp_path / "input-report.json"
    manifest.write_text(json.dumps(data))
    if damage:
        with pytest.raises(ValueError): build.validate_cuda(toolkit, manifest, windows=True)
    else:
        result = build.validate_cuda(toolkit, manifest, windows=True)
        assert result["verified_files"] == 3 and result["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
