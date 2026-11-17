# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable packaging guards use real distribution trees and archive bytes."""
import hashlib
import importlib.util
from importlib.metadata import PathDistribution
import json
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from copy import deepcopy

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "package_ovui_portable.py"


def load_package():
    assert SCRIPT.is_file(), "Portable package implementation is missing"
    spec = importlib.util.spec_from_file_location("portable_package", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def distribution(tmp_path, version="1.2.3", member="sample/data.bin"):
    site = tmp_path / "site"
    info = site / "sample-1.2.3.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Name: sample\nVersion: {version}\n")
    payload = site / member
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(b"payload")
    (info / "LICENSE").write_text("sample license")
    files = [member, "sample-1.2.3.dist-info/METADATA", "sample-1.2.3.dist-info/LICENSE"]
    (info / "RECORD").write_text("".join(f"{f},,\n" for f in files))
    return PathDistribution(info)


def test_distribution_keeps_assets_metadata_and_license(tmp_path):
    package = load_package()
    package.copy_distribution(distribution(tmp_path), "1.2.3", tmp_path / "app")
    assert (tmp_path / "app/sample/data.bin").read_bytes() == b"payload"
    assert (tmp_path / "app/sample-1.2.3.dist-info/LICENSE").read_text() == "sample license"


def test_distribution_omits_ovui_agent_tools_and_keeps_viewer_runtime(tmp_path):
    package = load_package()
    dist = distribution(tmp_path, member="omni/ui/runtime.py")
    tools = ["ovui_agent_skills/skills/omniverse-ui-inspector/scripts/ovui-inspect.py",
             "ovui_agent_skills/skills/omniverse-ui-inspector/scripts/ovui-inspect",
             "ovui_agent_skills/skills/omniverse-ui-inspector/SKILL.md"]
    record = tmp_path / "site/sample-1.2.3.dist-info/RECORD"
    for member in tools:
        path = tmp_path / "site" / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"agent tooling")
    record.write_text(record.read_text() + "".join(f"{name},,\n" for name in tools))

    package.copy_distribution(dist, "1.2.3", tmp_path / "app")

    assert (tmp_path / "app/omni/ui/runtime.py").read_bytes() == b"payload"
    assert (tmp_path / "app/sample-1.2.3.dist-info/LICENSE").is_file()
    assert not (tmp_path / "app/ovui_agent_skills").exists()


def test_runtime_inventory_rejects_leftover_ovui_agent_tools(tmp_path):
    script = tmp_path / "app/ovui_agent_skills/skills/omniverse-ui-inspector/scripts/ovui-inspect.py"
    script.parent.mkdir(parents=True)
    script.write_bytes(b"unexpected agent tooling")

    with pytest.raises(ValueError, match="OvUI agent tooling"):
        load_package().validate_runtime_inventory(tmp_path, [])


def test_wrong_dependency_pin_rejected(tmp_path):
    package = load_package()
    with pytest.raises(ValueError, match="version"):
        package.copy_distribution(distribution(tmp_path, "9.9.9"), "1.2.3", tmp_path / "app")


@pytest.mark.parametrize("commit,valid", [("a81db103a4d99ff2d75368a444e167889536ff42", True), ("wrong", False)])
def test_installed_ovui_build_matches_source_pin(tmp_path, commit, valid):
    package = load_package()
    dist = distribution(tmp_path, "0.2.0", "omni/ui/_build_info.py")
    (tmp_path / "site/omni/ui/_build_info.py").write_text(f"__version__ = '0.2.0'\n__commit__ = '{commit}'\n")
    if valid:
        package.validate_ovui_build(dist)
    else:
        with pytest.raises(ValueError, match="OVUI.*pin"):
            package.validate_ovui_build(dist)


def test_escaped_distribution_file_rejected(tmp_path):
    package = load_package()
    with pytest.raises(ValueError, match="escapes"):
        package.copy_distribution(distribution(tmp_path, member="../outside.bin"), "1.2.3", tmp_path / "app")


def test_output_reuse_rejected_without_modification(tmp_path):
    package = load_package()
    (tmp_path / "keep").write_text("original")
    with pytest.raises(ValueError, match="new"):
        package.new_output(tmp_path, [])
    assert (tmp_path / "keep").read_text() == "original"


def test_output_inside_runtime_rejected(tmp_path):
    package = load_package()
    with pytest.raises(ValueError, match="source"):
        package.new_output(tmp_path / "new", [tmp_path])


def test_version_mismatch_rejected(tmp_path):
    package = load_package()
    (tmp_path / "python/ovd_ui").mkdir(parents=True)
    (tmp_path / "CHANGELOG.md").write_text("# Releases\n## v0.1.25 (2026-09-13)\n")
    (tmp_path / "python/ovd_ui/__init__.py").write_text('__version__ = "0.1.24"\n')
    with pytest.raises(ValueError, match="version"):
        package.validate_version(tmp_path)


@pytest.mark.parametrize("suffix", [".zip", ".tar.gz"])
def test_archive_reproducibility_checksum_sizes_and_modes(tmp_path, suffix):
    package = load_package()
    bundle = tmp_path / "OvdNext-linux-x86_64"
    bundle.mkdir()
    (bundle / "OvdNext.sh").write_bytes(b"#!/bin/sh\n")
    (bundle / "data").write_bytes(b"abc")
    first = tmp_path / ("one" + suffix)
    second = tmp_path / ("two" + suffix)
    package.make_archive(bundle, first, 1700000000)
    os.utime(bundle / "data", (1800000000, 1800000000))
    package.make_archive(bundle, second, 1700000000)
    assert first.read_bytes() == second.read_bytes()
    report = package.finish_report(bundle, first, {"version": "0.1.25"}, 160)
    assert report["expanded_bytes"] == 13
    assert report["archive_bytes"] == len(first.read_bytes())
    assert first.with_name(first.name + ".sha256").read_text().split()[0] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert json.loads((tmp_path / "package-report.json").read_text())["archive_bytes"] == len(first.read_bytes())
    if suffix == ".zip":
        with zipfile.ZipFile(first) as archive:
            assert archive.read(bundle.name + "/data") == b"abc"
            assert (archive.getinfo(bundle.name + "/OvdNext.sh").external_attr >> 16) & 0o777 == 0o755
    else:
        with tarfile.open(first) as archive:
            assert archive.extractfile(bundle.name + "/data").read() == b"abc"
            assert archive.getmember(bundle.name + "/OvdNext.sh").mode == 0o755
            assert all(m.uid == m.gid == 0 and m.mtime == 1700000000 for m in archive.getmembers())


def test_zip_report_counts_hardlinks_as_materialized_files_and_writes_qualified_copy(tmp_path):
    package = load_package()
    bundle = tmp_path / "OvdNext-linux-x86_64"
    bundle.mkdir()
    (bundle / "libpython.so").write_bytes(b"one payload")
    os.link(bundle / "libpython.so", bundle / "libpython.so.1")
    archive = tmp_path / "OvdNext-linux-x86_64-1.2.3.zip"
    package.make_archive(bundle, archive, 1700000000)

    report = package.finish_report(bundle, archive, {"version": "1.2.3"}, 160)

    assert report["expanded_bytes"] == 22
    assert report["unique_payload_bytes"] == 22
    compatibility = tmp_path / "package-report.json"
    qualified = tmp_path / "OvdNext-linux-x86_64-1.2.3.zip.package-report.json"
    assert qualified.read_bytes() == compatibility.read_bytes()
    assert json.loads(qualified.read_text()) == report


def test_archive_budget_fails(tmp_path):
    package = load_package()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"123")
    with pytest.raises(ValueError, match="budget"):
        package.finish_report(bundle, archive, {}, 0)


@pytest.mark.parametrize("missing", [None, "README.txt", "native", "assets/colors.json", "notices/OVUI-LICENSE.txt",
                                     "notices/OvdNext-LICENSE.txt", "notices/OvdNext-NOTICE.txt",
                                     "notices/Expat-LICENSE.txt", "corrupt_expat_license",
                                     "notices/HACL-LICENSE.txt", "corrupt_hacl_license"])
def test_bundle_inspection_rejects_missing_payload(tmp_path, missing):
    package = load_package()
    app = tmp_path / "app"
    (tmp_path / "README.txt").write_text("Portable application instructions")
    shutil.copytree(SCRIPT.parents[1] / "python/ovd_ui/assets", app / "ovd_ui/assets")
    (app / "ovd_ui/notices").mkdir()
    (app / "_pvd_native.cp312-win_amd64.pyd").write_bytes(b"native")
    for name in ("NOTICES.txt", "CHANGELOG.md", "OvdNext-LICENSE.txt", "OvdNext-NOTICE.txt",
                 "OVUI-LICENSE.txt", "OVUI-THIRD_PARTY_NOTICES.md",
                 "Physics-LICENSE.md", "PVD-source-notice.txt", "PxShared-source-notice.txt",
                 "pybind11-LICENSE.txt", "Python-LICENSE.txt", "Expat-LICENSE.txt", "HACL-LICENSE.txt"):
        (app / "ovd_ui/notices" / name).write_text("license")
    (app / "ovd_ui/notices/Expat-LICENSE.txt").write_bytes(
        (SCRIPT.parent / "ovui_package/Expat-LICENSE.txt").read_bytes())
    (app / "ovd_ui/notices/HACL-LICENSE.txt").write_bytes(
        (SCRIPT.parent / "ovui_package/HACL-LICENSE.txt").read_bytes())
    if missing is None:
        package.inspect_bundle(tmp_path)
        return
    if missing == "corrupt_expat_license":
        (app / "ovd_ui/notices/Expat-LICENSE.txt").write_bytes(b"incomplete notice")
        with pytest.raises(ValueError, match="Expat license.*hash"):
            package.inspect_bundle(tmp_path)
        return
    if missing == "corrupt_hacl_license":
        (app / "ovd_ui/notices/HACL-LICENSE.txt").write_bytes(b"incomplete notice")
        with pytest.raises(ValueError, match="HACL license.*hash"):
            package.inspect_bundle(tmp_path)
        return
    target = (tmp_path / "README.txt" if missing == "README.txt" else
              app / "_pvd_native.cp312-win_amd64.pyd" if missing == "native" else app / "ovd_ui" / missing)
    target.unlink()
    with pytest.raises(ValueError, match="[Mm]issing|omits"):
        package.inspect_bundle(tmp_path)


def test_repository_expat_notice_matches_pinned_source_license():
    sys.path.insert(0, str(SCRIPT.parent))
    from install_ovui_expat import LICENSE_HASHES

    notice = SCRIPT.parent / "ovui_package/Expat-LICENSE.txt"
    assert notice.is_file(), "Expat's source license must be retained for both platforms"
    assert hashlib.sha256(notice.read_bytes()).hexdigest() == LICENSE_HASHES["LICENSE.expat.txt"]


@pytest.mark.parametrize("windows", [True, False])
def test_stage_notices_retains_component_terms_for_both_platforms(tmp_path, monkeypatch, windows):
    from types import SimpleNamespace

    package = load_package()
    bundle = tmp_path / "bundle"
    (bundle / "app/ovd_ui").mkdir(parents=True)
    # Full packaging copies the Python package before staging notices. Conflicting
    # committed/root bytes must fail here just as they would in a release build.
    committed_notices = package.ROOT / "python/ovd_ui/notices"
    if committed_notices.exists():
        shutil.copytree(committed_notices, bundle / "app/ovd_ui/notices")
    ovui = tmp_path / "ovui"
    ovui.mkdir()
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        (ovui / name).write_text("OvUI fixture license")
    prefix = tmp_path / "python"
    license_path = prefix / ("LICENSE.txt" if windows else "lib/python3.12/LICENSE.txt")
    license_path.parent.mkdir(parents=True)
    license_path.write_text("CPython fixture license")
    pybind_license = tmp_path / "pybind-license"
    pybind_license.write_text("pybind11 fixture license")
    monkeypatch.setattr(package.metadata, "distribution", lambda name: SimpleNamespace(
        files=["licenses/LICENSE"], locate_file=lambda path: pybind_license))
    monkeypatch.setattr(package, "run", lambda *args: "Physics fixture license")

    package.stage_notices(bundle, ovui, tmp_path / "physics", prefix)

    notices = bundle / "app/ovd_ui/notices"
    assert (notices / "Expat-LICENSE.txt").read_bytes() == (SCRIPT.parent / "ovui_package/Expat-LICENSE.txt").read_bytes()
    assert "Expat-LICENSE.txt" in package.NOTICES
    assert "Expat (XML parser in the Python runtime)" in (notices / "NOTICES.txt").read_text()
    assert (notices / "HACL-LICENSE.txt").read_bytes() == (SCRIPT.parent / "ovui_package/HACL-LICENSE.txt").read_bytes()
    assert "HACL* and KaRaMeL (Python hash implementations): see HACL-LICENSE.txt." in (notices / "NOTICES.txt").read_text()
    assert "python/licenses/" in (notices / "NOTICES.txt").read_text()
    assert "internal preview" not in (notices / "NOTICES.txt").read_text().casefold()
    pvd = (notices / "PVD-source-notice.txt").read_text()
    assert "Copyright (c) 2008-2014 NVIDIA Corporation" in pvd
    assert "SPDX-License-Identifier: Apache-2.0" in pvd
    assert "confidential" not in pvd.casefold()
    pxshared = (notices / "PxShared-source-notice.txt").read_text()
    for retained in ("NVIDIA CORPORATION", "AGEIA Technologies", "NovodeX AG",
                     "Redistribution and use in source and binary forms",
                     "``AS IS''", "IN NO EVENT SHALL"):
        assert retained in pxshared
    assert "Apache-2.0" not in pxshared
    for source, target in (("LICENSE", "OvdNext-LICENSE.txt"), ("NOTICE", "OvdNext-NOTICE.txt")):
        assert (notices / target).read_bytes() == (package.ROOT / source).read_bytes()
        assert "CC-BY" not in (notices / target).read_text()
    index = (notices / "NOTICES.txt").read_text()
    assert "Apache-2.0" in index
    assert "OvdNext-LICENSE.txt" in index and "OvdNext-NOTICE.txt" in index
    assert (notices / "OVUI-LICENSE.txt").read_text() == "OvUI fixture license"
    assert (notices / "OVUI-THIRD_PARTY_NOTICES.md").read_text() == "OvUI fixture license"
    assert (notices / "Python-LICENSE.txt").read_text() == "CPython fixture license"
    assert (notices / "pybind11-LICENSE.txt").read_text() == "pybind11 fixture license"
    assert (notices / "Physics-LICENSE.md").read_text() == "Physics fixture license\n"


@pytest.mark.parametrize("windows,member", [(True, "bin/cudart64_12.dll"), (False, "lib/libcudart.so.12")])
def test_cuda_stages_only_runtime_and_vendor_metadata(tmp_path, windows, member):
    package = load_package()
    dist = distribution(tmp_path, "12.9.79", "nvidia/cuda_runtime/" + member)
    header = tmp_path / "site/nvidia/cuda_runtime/include/cuda.h"
    header.parent.mkdir()
    header.write_text("developer header")
    record = tmp_path / "site/sample-1.2.3.dist-info/RECORD"
    record.write_text(record.read_text() + "nvidia/cuda_runtime/include/cuda.h,,\n")
    package.copy_cuda(dist, tmp_path / "bundle", windows)
    assert (tmp_path / "bundle/lib" / Path(member).name).read_bytes() == b"payload"
    assert (tmp_path / "bundle/lib/notices/sample-1.2.3.dist-info/LICENSE").read_text() == "sample license"
    assert not list((tmp_path / "bundle").rglob("cuda.h"))


def test_linux_python_does_not_copy_unrelated_prefix_libraries(tmp_path):
    package = load_package()
    prefix = tmp_path / "prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib/python3.12").mkdir(parents=True)
    (prefix / "bin/python3.12").write_bytes(b"python")
    (prefix / "lib/libpython3.12.so.1.0").write_bytes(b"python shared library")
    (prefix / "lib/libgraphics.so").write_bytes(b"unrelated host library")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    package.copy_python(prefix, bundle, False)
    assert (bundle / "python/lib/libpython3.12.so.1.0").read_bytes() == b"python shared library"
    assert not (bundle / "python/lib/libgraphics.so").exists()


@pytest.mark.parametrize("platform", ["windows", "linux"])
def test_python_copy_retains_complete_verified_dependency_notices(tmp_path, platform):
    package = load_package()
    prefix = tmp_path / "prefix"
    windows = platform == "windows"
    files = (("python.exe", "python3.dll", "python312.dll", "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt")
             if windows else ("bin/python3.12", "lib/libpython3.12.so.1.0"))
    for name in files:
        path = prefix / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"runtime fixture")
    (prefix / ("Lib" if windows else "lib/python3.12")).mkdir(parents=True)
    if windows:
        (prefix / "DLLs").mkdir()
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    package.copy_python(prefix, bundle, windows)

    # Independently audited full-PBS/source bytes for every delivered consumer.
    expected = {
        "LICENSE.bzip2.txt": "c6dbbf828498be844a89eaa3b84adbab3199e342eb5cb2ed2f0d4ba7ec0f38a3",
        "LICENSE.cpython.txt": "86e61415828a8b5b06ec8d024e6f086ce155a8b85fd0c419c0ba4dc004e74fdd",
        "LICENSE.libedit.txt": "29cea33c32bbc9785142386377915612a2fa786482c46843383384aded2e09b1",
        "LICENSE.libffi.txt": "deaf3a42effb551a5b140fa9afefed183a27f1341c6d1bf430d106a5e6931fc0",
        "LICENSE.liblzma.txt": "9a4062de0a2c388a98cf35a35d348b62fa97c838a71c3c28ee1a2d7d0a565b02",
        "LICENSE.libuuid.txt": "122ee1f7e258f2c3c0e538a75c037684f420454bf3850ddc74ce750bbf5fe86b",
        "LICENSE.mpdecimal.txt": "669512af7219f58be03a398766d7c9da11a3b3df9d3f05cb74c5ceca25c8da3b",
        "LICENSE.ncurses.txt": "87a4c4442337b8968ef956031c406b74f9cb7149b7ba87311bdaba534816201c",
        "LICENSE.openssl-3.txt": "7d5450cb2d142651b8afa315b5f238efc805dad827d91ba367d8516bc9d49e7a",
        "LICENSE.sqlite.txt": "38bef3d28b24f145ea293bd3b6eb4b20396982abc8303128fb493986ea5bc719",
        "LICENSE.zlib.txt": "e32ff4e00d9d94930537635291da39e7e612703334bf6fde8c7f1686fe8a45a2",
    }
    if windows:
        expected = {
            "LICENSE.cpython.txt": "341832873fd316a37927e79385093fbbfd40a467428480835fe435a80cadf4e5",
            "LICENSE.libffi.txt": "a61d06e8f7be57928e71e800eb9273b05cb8868c484108afe41e4305bb320dde",
            "LICENSE.liblzma.txt": "0b01625d853911cd0e2e088dcfb743261034a091bb379246cb25a14cc4c74bf1",
            "LICENSE.openssl-3.txt": "7d5450cb2d142651b8afa315b5f238efc805dad827d91ba367d8516bc9d49e7a",
            "LICENSE.sqlite.txt": "49cf7bcb103eb3244150a23bb23d840370a4762914faf1b450072a4b03c14670",
            "LICENSE.zlib.txt": "e32ff4e00d9d94930537635291da39e7e612703334bf6fde8c7f1686fe8a45a2",
        }
    notices = bundle / "python/licenses"
    assert {path.name for path in notices.glob("*")} == set(expected)
    assert {name: hashlib.sha256((notices / name).read_bytes()).hexdigest()
            for name in expected} == expected


def test_python_notice_source_provenance_matches_committed_file_pins():
    manifest = json.loads((SCRIPT.parent / "ovui-python-3.12.16.json").read_text())
    resources = SCRIPT.parent / "ovui_package/python-runtime-notices"
    windows = manifest["platforms"]["windows"]["dependency_notices"]
    mapping = {"openssl": "LICENSE.openssl-3.txt", "libffi": "LICENSE.libffi.txt",
               "liblzma": "LICENSE.liblzma.txt", "sqlite": "LICENSE.sqlite.txt", "zlib": "LICENSE.zlib.txt"}
    pins = windows["files_sha256"]
    assert set(windows["dependency_sources"]) == set(mapping)
    assert set(pins) == {*mapping.values(), "LICENSE.cpython.txt"}
    sources = [(windows["dependency_sources"][name], member) for name, member in mapping.items()]
    sources.append((windows["cpython_source"], "LICENSE.cpython.txt"))
    for source, member in sources:
        assert source["notice_sha256"] == pins[member]
        assert hashlib.sha256((resources / "windows" / member).read_bytes()).hexdigest() == pins[member]
        assert isinstance(source["notice_member"], str) and source["notice_member"]
        assert re.fullmatch(r"[0-9a-f]{64}", source.get("archive_sha256", source.get("sha256", "")))

    linux = manifest["platforms"]["linux"]["dependency_notices"]
    for dependency in ("bzip2", "zlib"):
        source = linux[dependency + "_source"]
        member = "LICENSE." + dependency + ".txt"
        assert source["license_sha256"] == linux["files_sha256"][member]
        assert hashlib.sha256((resources / "linux" / member).read_bytes()).hexdigest() == source["license_sha256"]
        assert re.fullmatch(r"[0-9a-f]{64}", source["archive_sha256"])


def test_windows_python_copy_restores_verified_dependency_notices(tmp_path, monkeypatch):
    sys.path.insert(0, str(SCRIPT.parent))
    import install_ovui_python

    package = load_package()
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    for name in ("python.exe", "python3.dll", "python312.dll", "vcruntime140.dll",
                 "vcruntime140_1.dll", "LICENSE.txt"):
        (prefix / name).write_bytes(b"runtime fixture")
    (prefix / "Lib").mkdir()
    (prefix / "DLLs").mkdir()
    resources = tmp_path / "notice-resources"
    notices = resources / "windows"
    notices.mkdir(parents=True)
    notice = notices / "LICENSE.libffi.txt"
    notice.write_bytes(b"verified libffi notice\n")
    manifest = install_ovui_python.load_manifest()
    manifest["platforms"]["windows"]["dependency_notices"] = {
        "source_archive": {"sha256": "c" * 64},
        "files_sha256": {notice.name: hashlib.sha256(notice.read_bytes()).hexdigest()}}
    monkeypatch.setattr(install_ovui_python, "load_manifest", lambda: deepcopy(manifest))
    monkeypatch.setattr(package, "PYTHON_NOTICE_RESOURCES", resources)
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    package.copy_python(prefix, bundle, True)

    assert (bundle / "python/LICENSE.txt").read_bytes() == b"runtime fixture"
    assert (bundle / "python/licenses/LICENSE.libffi.txt").read_bytes() == b"verified libffi notice\n"


def notice_copy_fixture(tmp_path, monkeypatch, platform):
    """Keep real Python copying and notice validation with one pinned notice."""
    sys.path.insert(0, str(SCRIPT.parent))
    import install_ovui_python

    package = load_package()
    prefix = tmp_path / "prefix"
    files = (("python.exe", "python3.dll", "python312.dll", "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt")
             if platform == "windows" else ("bin/python3.12", "lib/libpython3.12.so.1.0"))
    for name in files:
        path = prefix / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"runtime fixture")
    (prefix / ("Lib" if platform == "windows" else "lib/python3.12")).mkdir(parents=True)
    if platform == "windows":
        (prefix / "DLLs").mkdir()
    resources = tmp_path / "notice-resources"
    notices = resources / platform
    notices.mkdir(parents=True)
    notice = notices / "LICENSE.zlib.txt"
    notice.write_bytes(b"verified zlib notice\n")
    manifest = install_ovui_python.load_manifest()
    manifest["platforms"][platform]["dependency_notices"] = {
        "source_archive": {"sha256": "c" * 64},
        "files_sha256": {notice.name: hashlib.sha256(notice.read_bytes()).hexdigest()}}
    monkeypatch.setattr(install_ovui_python, "load_manifest", lambda: deepcopy(manifest))
    monkeypatch.setattr(package, "PYTHON_NOTICE_RESOURCES", resources)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    return package, prefix, bundle, notice, manifest


@pytest.mark.parametrize("platform", ["windows", "linux"])
@pytest.mark.parametrize("pins", [{}, {"LICENSE.zlib.txt": "A" * 64}, {"../LICENSE.zlib.txt": "a" * 64},
                                  {"LICENSE.zlib.txt/../LICENSE.zlib.txt": "a" * 64},
                                  {"LICENSE.zlib.txt\n": "a" * 64},
                                  {"LICENSE.zlib.txt": 123}, {123: "a" * 64}, []])
def test_python_dependency_notices_require_safe_complete_sha256_pins(tmp_path, monkeypatch, platform, pins):
    package, _, _, notice, manifest = notice_copy_fixture(tmp_path, monkeypatch, platform)
    manifest["platforms"][platform]["dependency_notices"]["files_sha256"] = pins

    with pytest.raises(ValueError, match="complete filename/SHA-256 pins"):
        package.python_dependency_notice_records(notice.parent, platform)


@pytest.mark.parametrize("platform", ["windows", "linux"])
@pytest.mark.parametrize("corrupt", ["missing", "changed"])
def test_python_notice_source_rejected_before_runtime_copy(tmp_path, monkeypatch, platform, corrupt):
    package, prefix, bundle, notice, _ = notice_copy_fixture(tmp_path, monkeypatch, platform)
    if corrupt == "missing":
        notice.unlink()
    else:
        notice.write_bytes(b"unverified notice")

    with pytest.raises(ValueError, match="Python dependency notice"):
        package.copy_python(prefix, bundle, platform == "windows")

    assert not (bundle / "python").exists()


@pytest.mark.parametrize("platform", ["windows", "linux"])
def test_python_notice_symlink_is_rejected_even_with_matching_bytes(tmp_path, monkeypatch, platform):
    package, _, _, notice, _ = notice_copy_fixture(tmp_path, monkeypatch, platform)
    target = tmp_path / "same-notice.txt"
    target.write_bytes(notice.read_bytes())
    notice.unlink()
    try:
        notice.symlink_to(target)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege unavailable; exercised on Linux")
        raise

    with pytest.raises(ValueError, match="not a regular file"):
        package.python_dependency_notice_records(notice.parent, platform)


@pytest.mark.parametrize("platform", ["windows", "linux"])
def test_python_notice_copy_corruption_is_rejected(tmp_path, monkeypatch, platform):
    package, prefix, bundle, notice, _ = notice_copy_fixture(tmp_path, monkeypatch, platform)
    real_copy = package.copy_file

    def corrupt_notice_copy(source, target):
        real_copy(source, target)
        if source == notice:
            target.write_bytes(b"corrupted while staging")

    monkeypatch.setattr(package, "copy_file", corrupt_notice_copy)

    with pytest.raises(ValueError, match="Python dependency notice hash mismatch"):
        package.copy_python(prefix, bundle, platform == "windows")


def test_windows_python_omits_orphan_zlib_but_distribution_keeps_its_provider(tmp_path, monkeypatch):
    package, prefix, bundle, _, _ = notice_copy_fixture(tmp_path, monkeypatch, "windows")
    retained = {"_ssl.pyd", "libssl-3-x64.dll", "libcrypto-3-x64.dll", "_ctypes.pyd", "libffi-8.dll"}
    for name in retained | {"zlib1.dll"}:
        (prefix / "DLLs" / name).write_bytes(b"runtime DLL fixture")

    package.copy_python(prefix, bundle, True)

    assert {path.name for path in (bundle / "python/DLLs").iterdir()} == retained
    dist = distribution(tmp_path / "wheel", member="sample.libs/zlib1.dll")
    package.copy_distribution(dist, "1.2.3", bundle / "app")
    assert (bundle / "app/sample.libs/zlib1.dll").read_bytes() == b"payload"


@pytest.mark.parametrize("windows", [True, False])
def test_python_copy_preserves_verified_input_and_xml_build_records(tmp_path, windows):
    package = load_package()
    prefix = tmp_path / "prefix"
    files = ("python.exe", "python3.dll", "python312.dll", "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt") if windows else ("bin/python3.12", "lib/libpython3.12.so.1.0")
    for name in files:
        path = prefix / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"runtime fixture")
    (prefix / ("Lib" if windows else "lib/python3.12")).mkdir(parents=True)
    if windows:
        (prefix / "DLLs").mkdir()
    markers = {"runtime-input-provenance.json": b'{"input": "verified archive"}\n'}
    if windows:
        markers["ovdnext-expat-build.json"] = b'{"xml_modules": "verified rebuild"}\n'
        markers["LICENSE.expat.txt"] = b"Expat source license\n"
        markers["LICENSE.cpython-xml.txt"] = b"CPython XML source license\n"
    for name, data in markers.items():
        (prefix / name).write_bytes(data)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    package.copy_python(prefix, bundle, windows)
    for name, data in markers.items():
        assert (bundle / "python" / name).read_bytes() == data
    if not windows:
        assert not (bundle / "python/ovdnext-expat-build.json").exists()


def test_distribution_omits_tests_and_numpy_sdk_but_keeps_runtime(tmp_path):
    package = load_package()
    dist = distribution(tmp_path, member="numpy/_core/runtime.py")
    files = ["numpy/tests/test_example.py", "numpy/_core/include/numpy/arrayobject.h",
             "numpy/_core/_multiarray_tests.pyd", "numpy/_core/_umath_tests.pyi", "numpy/_core/_multiarray_umath.pyd"]
    record = tmp_path / "site/sample-1.2.3.dist-info/RECORD"
    for member in files:
        path = tmp_path / "site" / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    record.write_text(record.read_text() + "".join(f"{name},,\n" for name in files))
    package.copy_distribution(dist, "1.2.3", tmp_path / "app")
    assert (tmp_path / "app/numpy/_core/_multiarray_umath.pyd").is_file()
    assert not (tmp_path / "app/numpy/tests").exists()
    assert not (tmp_path / "app/numpy/_core/include").exists()
    assert not (tmp_path / "app/numpy/_core/_multiarray_tests.pyd").exists()
    assert not (tmp_path / "app/numpy/_core/_umath_tests.pyi").exists()


def test_python_omits_native_tests_and_tcl_but_keeps_runtime(tmp_path):
    package = load_package()
    source = tmp_path / "DLLs"
    source.mkdir()
    for name in ("_testcapi.pyd", "_testinternalcapi.pyd", "_ctypes_test.pyd", "tcl86t.dll", "tk86t.dll",
                 "_dbm.cpython-312-x86_64-linux-gnu.so", "_ssl.pyd"):
        (source / name).write_bytes(b"fixture")
    package.copy_tree(source, tmp_path / "staged")
    assert {p.name for p in (tmp_path / "staged").iterdir()} == {"_ssl.pyd"}


def test_linux_hardlink_archive_preserves_alias_without_duplicate_payload(tmp_path):
    package = load_package()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "libpython.so").write_bytes(b"one payload")
    os.link(bundle / "libpython.so", bundle / "libpython.so.1")
    archive = tmp_path / "bundle.tar.gz"
    package.make_archive(bundle, archive, 1700000000)
    with tarfile.open(archive) as packed:
        first, alias = packed.getmembers()
        assert first.isfile()
        assert alias.islnk() and alias.linkname == first.name and alias.size == 0
        packed.extractall(tmp_path / "extracted", filter="data")
    assert (tmp_path / "extracted/bundle/libpython.so.1").read_bytes() == b"one payload"
    report = package.finish_report(bundle, archive, {}, 160)
    assert report["expanded_bytes"] == 22
    assert report["unique_payload_bytes"] == 11


def test_probe_environment_removes_developer_search_paths(tmp_path):
    package = load_package()
    env = package.probe_environment(tmp_path, False, {"PATH": "/developer/bin", "PYTHONPATH": "/source", "LD_LIBRARY_PATH": "/developer/lib",
                                                     "GLCONTEXT_WIN_LIBGL": "/ci/mesa/opengl32.dll", "GALLIUM_DRIVER": "llvmpipe"})
    assert env["PATH"] == "/usr/bin:/bin"
    assert "PYTHONPATH" not in env
    assert "GLCONTEXT_WIN_LIBGL" not in env and "GALLIUM_DRIVER" not in env
    assert env["LD_LIBRARY_PATH"] == str(tmp_path / "lib") + ":" + str(tmp_path / "python/lib")


@pytest.mark.skipif(sys.platform != "linux", reason="Uses real Linux ELF and installed package metadata")
def test_linux_support_library_requires_notice_coverage(tmp_path):
    package = load_package()
    source = Path("/lib/x86_64-linux-gnu/libz.so.1")
    result = package.linux_library_notice(source, tmp_path)
    assert result["package"].startswith("zlib1g") and result["version"]
    assert (tmp_path / result["notice"]).stat().st_size > 100
    glib = package.linux_library_notice(Path("/usr/lib/x86_64-linux-gnu/libglib-2.0.so.0"), tmp_path)
    assert any("LGPL" in item["file"] for item in glib["licenses"])
    assert all((tmp_path / item["file"]).stat().st_size > 1000 for item in glib["licenses"])
    unknown = tmp_path / "unknown.so"
    unknown.write_bytes(b"unknown provider")
    with pytest.raises(ValueError, match="notice|package"):
        package.linux_library_notice(unknown, tmp_path)


@pytest.mark.skipif(sys.platform != "linux", reason="Uses real Linux compiler, objcopy and strip")
def test_linux_debug_split_and_aliases_touch_only_staged_copies(tmp_path):
    package = load_package()
    source = tmp_path / "original.c"
    source.write_text("int answer(void) { return 42; }\n")
    original = tmp_path / "original.so"
    subprocess.run(["cc", "-shared", "-g", "-o", original, source], check=True)
    original_bytes = original.read_bytes()
    bundle = tmp_path / "bundle"
    (bundle / "app").mkdir(parents=True)
    shutil.copyfile(original, bundle / "app/_pvd_native.test.so")
    shutil.copyfile(original, bundle / "library.so")
    shutil.copyfile(original, bundle / "library.so.1")
    package.prepare_linux_elf(bundle, tmp_path / "symbols")
    assert original.read_bytes() == original_bytes
    assert list((tmp_path / "symbols").rglob("*.debug"))
    assert ".debug_info" not in subprocess.check_output(["readelf", "-S", bundle / "app/_pvd_native.test.so"], text=True)
    assert (bundle / "library.so").read_bytes() == original_bytes
    assert (bundle / "library.so").stat().st_ino == (bundle / "library.so.1").stat().st_ino

@pytest.mark.parametrize("name", ["Qt5Core.dll", "libQt6Gui.so.6", "qwt.dll", "libqwt-qt5.so.6", "PySide6", "PyQt5"])
def test_rejects_qt_runtime_identities(name):
    with pytest.raises(ValueError, match="Qt"):
        load_package().reject_qt_runtime(name)

def test_optional_pillow_qt_adapter_is_not_a_runtime_dependency():
    load_package().reject_qt_runtime("PIL/ImageQt.py")
    load_package().reject_qt_runtime("Pillow")

def test_native_dependency_scan_rejects_qt_before_staging(tmp_path, monkeypatch):
    package = load_package()
    monkeypatch.setattr(package, "run", lambda *a, **k: "  NEEDED Shared library: [libQt5Core.so.5]")
    with pytest.raises(ValueError, match="Qt"):
        package.native_dependencies(tmp_path / "_pvd_native.so", False)

def test_no_qt_required_cli_argument():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], text=True, capture_output=True)
    assert result.returncode == 0
    assert "--qt-root" not in result.stdout

def test_rejects_distribution_metadata_for_qt_binding(tmp_path):
    dist = distribution(tmp_path)
    (tmp_path / "site/sample-1.2.3.dist-info/METADATA").write_text("Name: PySide6\nVersion: 1.2.3\n")
    with pytest.raises(ValueError, match="Qt"):
        load_package().copy_distribution(dist, "1.2.3", tmp_path / "app")

@pytest.mark.parametrize("entry", ["lib/Qt5Core.dll", "app/PySide6-6.0.dist-info/METADATA"])
def test_staged_inventory_rejects_qt_files_and_metadata(tmp_path, entry):
    path = tmp_path / entry
    path.parent.mkdir(parents=True)
    path.write_text("Name: PySide6\nVersion: 6.0\n")
    with pytest.raises(ValueError, match="Qt"):
        load_package().validate_runtime_inventory(tmp_path, [])


@pytest.mark.skipif(sys.platform == "win32", reason="Linux ldd staging uses POSIX library paths")
def test_linux_staging_keeps_host_compiler_runtime_out_of_bundle(tmp_path, monkeypatch):
    package = load_package()
    host = tmp_path / "host"
    host.mkdir()
    names = ("libstdc++.so.6", "libgcc_s.so.1", "libz.so.1")
    for name in names:
        (host / name).write_bytes(name.encode())
    bundle = tmp_path / "bundle"
    (bundle / "app").mkdir(parents=True)
    prefix = tmp_path / "python-prefix"
    prefix.mkdir()
    monkeypatch.setattr(package.sys, "base_prefix", str(prefix))
    monkeypatch.setattr(package.metadata, "distribution", lambda _: object())
    monkeypatch.setattr(package, "copy_cuda", lambda *_: None)
    monkeypatch.setattr(package, "linux_library_notice", lambda *_: {"provider": "test"})
    def native_tool(command, **kwargs):
        assert command[0] in {"ldd", "readelf"}
        if command[0] == "readelf":
            return "\n".join(f"NEEDED Shared library: [{name}]" for name in names)
        return "\n".join(f" {name} => {host / name} (0x1)" for name in names)
    monkeypatch.setattr(package, "run", native_tool)
    native = bundle / "app/_pvd_native.so"
    native.write_bytes(b"native")
    report = package.native_dependencies(native, False)
    assert "libstdc++.so.6" in report and "libgcc_s.so.1" in report
    notices = package.stage_libraries(native, bundle, False, None)
    assert sorted(path.name for path in (bundle / "lib").iterdir()) == ["libz.so.1"]
    assert (bundle / "lib/libz.so.1").read_bytes() == b"libz.so.1"
    assert set(notices) == {"lib/libz.so.1"}
    package.validate_runtime_inventory(bundle, [])


@pytest.mark.parametrize("entry", ["lib/libstdc++.so.6", "app/vendor/libstdc++.so.6.0.30",
                                  "lib/libgcc_s.so.1", "app/libgcc_s.so"])
def test_inventory_rejects_vendored_linux_compiler_runtime(tmp_path, entry):
    path = tmp_path / entry
    path.parent.mkdir(parents=True)
    path.write_bytes(b"unexpected builder runtime")
    with pytest.raises(ValueError, match="host.*runtime"):
        load_package().validate_runtime_inventory(tmp_path, [])


def secure_runtime_evidence():
    sys.path.insert(0, str(SCRIPT.parent))
    from install_ovui_pillow import DISABLED_CODECS, PILLOW_SOURCE, ZLIB_SOURCE

    return {"python": "3.12.16", "expat": "expat_2.9.0", "pillow": {
        "version": "13.0.0", "features": {"zlib": True, **dict.fromkeys(
            ("jpg", "jpg_2000", "libtiff", "freetype2", "webp", "littlecms2", "raqm", "libimagequant", "xcb", "avif"), False)},
        "png_decode": True, "image_color": True, "raw_rgba": True,
        "zlib_version": "1.3.1.zlib-ng", "zlib_ng_version": "2.3.3",
        "build_provenance": {
            "purpose": "OvdNext PNG icons, raw RGBA and ImageColor",
            "sources": deepcopy([PILLOW_SOURCE, ZLIB_SOURCE]), "build_python": "3.12.16",
            "disabled_codecs": list(DISABLED_CODECS), "zlib_linkage": "static",
            "build_settings": {"zlib": "enable", "platform-guessing": "disable"}},
        "vendor_license_sha256": "6c9f0d975b41afaa34d22f55bb8986ce69e5cb7ad327cb2b28820cd425edf5ee",
        "vendor_license_declared": True}}


@pytest.mark.parametrize("failure", ["missing_manifest", "wrong_source_hash", "missing_vendor_license"])
def test_package_security_guard_rejects_codec_disabled_unverified_pillow(failure):
    evidence = secure_runtime_evidence()
    if failure == "missing_manifest":
        evidence["pillow"].pop("build_provenance")
    if failure == "wrong_source_hash":
        evidence["pillow"]["build_provenance"]["sources"][1]["sha256"] = "0" * 64
    if failure == "missing_vendor_license":
        evidence["pillow"].pop("vendor_license_sha256")
    with pytest.raises(ValueError, match="provenance|license"):
        load_package().validate_security_runtime(evidence)


@pytest.mark.parametrize("windows,identity", [(True, "zlib1.dll"), (True, "zlib-ng.dll"),
                                             (True, "zlibwapi.dll"), (False, "libz.so.1"),
                                             (False, "libz-ng.so.2")])
def test_pillow_native_dependencies_reject_dynamic_zlib(tmp_path, monkeypatch, windows, identity):
    package = load_package()
    monkeypatch.setattr(package.shutil, "which", lambda _: "dumpbin")
    monkeypatch.setattr(package, "run", lambda *args, **kwargs:
                        "    " + identity if windows else "NEEDED Shared library: [" + identity + "]")
    with pytest.raises(ValueError, match="static.*zlib"):
        package.native_dependencies(tmp_path / "_imaging.pyd", windows, require_static_zlib=True)


def test_static_zlib_guard_does_not_mistake_zstd_for_zlib(tmp_path, monkeypatch):
    package = load_package()
    monkeypatch.setattr(package, "run", lambda *args, **kwargs: "NEEDED Shared library: [libzstd.so.1]")
    package.native_dependencies(tmp_path / "_imaging.so", False, require_static_zlib=True)


@pytest.mark.parametrize("python", ["3.12.13", "3.12.14", "3.13.0", None])
def test_package_security_guard_rejects_unqualified_python(python):
    evidence = secure_runtime_evidence()
    evidence["python"] = python
    with pytest.raises(ValueError, match="Python.*3.12.16"):
        load_package().validate_security_runtime(evidence)


@pytest.mark.parametrize("expat", ["expat_2.8.5", "expat_2.8.4", "2.9.0", None])
def test_package_security_guard_rejects_unqualified_expat(expat):
    evidence = secure_runtime_evidence()
    evidence["expat"] = expat
    with pytest.raises(ValueError, match="Expat.*expat_2.9.0"):
        load_package().validate_security_runtime(evidence)


def test_package_security_guard_requires_measured_expat():
    evidence = secure_runtime_evidence()
    evidence.pop("expat")
    with pytest.raises(ValueError, match="Expat.*expat_2.9.0"):
        load_package().validate_security_runtime(evidence)


@pytest.mark.parametrize("field", ["png_decode", "image_color", "raw_rgba", "zlib"])
def test_package_security_guard_requires_application_image_capabilities(field):
    evidence = secure_runtime_evidence()
    target = evidence["pillow"]["features"] if field == "zlib" else evidence["pillow"]
    target[field] = False
    with pytest.raises(ValueError, match="Pillow|PNG|RGBA|zlib|ImageColor"):
        load_package().validate_security_runtime(evidence)


def test_package_security_guard_rejects_enabled_jpeg2000():
    evidence = secure_runtime_evidence()
    evidence["pillow"]["features"]["jpg_2000"] = True
    with pytest.raises(ValueError, match="JPEG2000|OpenJPEG|jpg_2000"):
        load_package().validate_security_runtime(evidence)


def test_package_security_guard_accepts_qualified_runtime():
    load_package().validate_security_runtime(secure_runtime_evidence())


def test_packaging_rejects_old_python_before_creating_output(tmp_path, monkeypatch, capsys):
    module = load_package()
    monkeypatch.setattr(module.platform, "python_version", lambda: "3.12.13")
    monkeypatch.setattr(module.sys, "version_info", (3, 12, 13))
    with pytest.raises(SystemExit) as failure:
        module.main(["--output", str(tmp_path / "output"), "--native", str(tmp_path / "native"),
                     "--physics-root", str(tmp_path / "physics"), "--ovui-root", str(tmp_path / "ovui")])
    assert failure.value.code == 2
    assert "Packaged Python must be 3.12.16" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("entry", ["app/pillow.libs/libopenjp2-abc.so.2.5.4", "app/PIL/openjp2.dll"])
def test_inventory_rejects_openjpeg_even_when_codec_probe_is_disabled(tmp_path, entry):
    path = tmp_path / entry
    path.parent.mkdir(parents=True)
    path.write_bytes(b"unexpected JPEG2000 decoder")
    with pytest.raises(ValueError, match="OpenJPEG"):
        load_package().validate_runtime_inventory(tmp_path, [])
