# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail closed at downloaded-package, extracted-runtime, and process boundaries."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace
import warnings
import zipfile
from copy import deepcopy

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci"))
SPEC = importlib.util.spec_from_file_location("verify_ovui_package", ROOT / "ci/verify_ovui_package.py")
verify = importlib.util.module_from_spec(SPEC)
if SPEC.loader and Path(SPEC.origin).is_file():
    SPEC.loader.exec_module(verify)

ZLIB_LICENSE = b"""(C) 1995-2024 Jean-loup Gailly and Mark Adler

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.

2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.

3. This notice may not be removed or altered from any source distribution.
"""


def security_evidence():
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
        "vendor_license_sha256": hashlib.sha256(ZLIB_LICENSE).hexdigest(),
        "vendor_license_declared": True}}


def python_input_provenance(platform):
    from install_ovui_python import core_paths, input_provenance

    windows = platform == "windows"
    return {**input_provenance(windows), "runtime_files": {
        name: hashlib.sha256(b"python").hexdigest() for name in core_paths(windows)}}


def python_dependency_notice_fixture(platform):
    from install_ovui_python import load_manifest

    entry = load_manifest()["platforms"][platform]
    spec = entry["dependency_notices"]
    files = [("python/licenses/" + name,
              (ROOT / "ci/ovui_package/python-runtime-notices" / platform / name).read_bytes())
             for name in spec["files_sha256"]]
    return {"input_archive_sha256": entry["sha256"], **deepcopy(spec)}, files


@pytest.fixture(autouse=True)
def xml_fixture_licenses(monkeypatch):
    import install_ovui_expat
    import install_ovui_python

    manifest = deepcopy(install_ovui_python.load_manifest())
    for platform, names in {
            "windows": ("python.exe", "python3.dll", "python312.dll"),
            "linux": ("bin/python3.12", "lib/libpython3.12.so", "lib/libpython3.12.so.1.0")}.items():
        manifest["platforms"][platform]["core_sha256"] = {
            name: hashlib.sha256(b"python").hexdigest() for name in names}
    monkeypatch.setattr(install_ovui_python, "load_manifest", lambda: deepcopy(manifest))

    # The package fixtures use synthetic binaries and license bytes. Retain the
    # real source/input/module validators while selecting these fixture licenses.
    monkeypatch.setattr(install_ovui_expat, "LICENSE_HASHES", {
        name: hashlib.sha256(name.encode()).hexdigest() for name in install_ovui_expat.LICENSE_HASHES})


def xml_build_fixture(tmp_path):
    import install_ovui_expat as expat

    prefix = tmp_path / "xml-build-fixture"
    files = {name: name.encode() for name in (*expat.MODULE_PATHS, *expat.LICENSE_HASHES)}
    for name, data in files.items():
        path = prefix / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    record = expat.make_build_provenance(prefix, python_input_provenance("windows"), {"name": "MSVC"})
    return record, [("python/" + name, data) for name, data in files.items()]


def package(tmp_path, *, platform="windows", entries=None, manifest=None, modes=None):
    entries = entries or [("README.txt", b"portable fixture")]
    if manifest is None:
        manifest = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in entries)
    destination = tmp_path / "packages" / platform
    destination.mkdir(parents=True)
    archive = destination / f"OvdNext-{platform}-x86_64-0.2.3.zip"
    with warnings.catch_warnings(), zipfile.ZipFile(archive, "w") as target:
        warnings.simplefilter("ignore", UserWarning)
        for name, data in [*entries, ("SHA256SUMS", manifest.encode())]:
            info = zipfile.ZipInfo(f"OvdNext-{platform}-x86_64/{name}")
            info.filename = f"OvdNext-{platform}-x86_64/{name}"  # Preserve adversarial backslashes on Windows.
            info.create_system = 3
            info.external_attr = (modes or {}).get(name, 0o100644) << 16
            target.writestr(info, data)
    report = dict(version="0.2.3", platform=platform, source_commit="a" * 40,
                  source_dirty=False, allow_dirty=False, dependency_dirty={"physics": "", "ovui": ""},
                  distributions=[], native_sha256="b" * 64, **security_evidence(),
                  python_input_provenance=python_input_provenance(platform),
                  archive=archive.name, archive_bytes=archive.stat().st_size,
                  archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    report["python_dependency_notices"] = python_dependency_notice_fixture(platform)[0]
    if platform == "windows":
        report["expat_build_provenance"] = {"fixture": "verified XML rebuild"}
    for name in ("package-report.json", archive.name + ".package-report.json"):
        (destination / name).write_text(json.dumps(report), encoding="utf-8")
    (destination / (archive.name + ".sha256")).write_text(f"{report['archive_sha256']}  {archive.name}\n")
    return archive, report


def test_inventory_rejects_corrupt_payload_even_with_correct_outer_checksum(tmp_path):
    archive, _ = package(tmp_path, manifest="0" * 64 + "  README.txt\n")
    with pytest.raises(ValueError, match="checksum"):
        verify.archive_inventory(archive, "windows")


@pytest.mark.parametrize("manifest", ["", "0" * 64 + "  missing.txt\n",
                                      "0" * 64 + "  README.txt\n" + "0" * 64 + "  README.txt\n"])
def test_inventory_requires_exact_nonduplicate_manifest_coverage(tmp_path, manifest):
    archive, _ = package(tmp_path, manifest=manifest)
    with pytest.raises(ValueError):
        verify.archive_inventory(archive, "windows")


@pytest.mark.parametrize("name", ["../escape.txt", "/escape.txt", "a/../../escape.txt", "C:/escape.txt",
                                  "a\\escape.txt", "a/./file", "a//file", "CON.txt", "name.", "name ",
                                  "name:stream", "a?b"])
def test_inventory_rejects_escaping_and_unsafe_windows_paths(tmp_path, name):
    archive, _ = package(tmp_path, entries=[(name, b"unsafe")])
    with pytest.raises(ValueError, match="path"):
        verify.archive_inventory(archive, "windows")
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize("entries", [[("README.txt", b"one"), ("README.txt", b"two")],
                                     [("README.txt", b"one"), ("readme.txt", b"two")],
                                     [("a", b"file"), ("a/b", b"child")]])
def test_inventory_rejects_duplicate_and_conflicting_paths(tmp_path, entries):
    archive, _ = package(tmp_path, entries=entries)
    with pytest.raises(ValueError):
        verify.archive_inventory(archive, "windows")


def test_inventory_rejects_symlink_before_extraction(tmp_path):
    archive, _ = package(tmp_path, modes={"README.txt": 0o120777})
    with pytest.raises(ValueError, match="regular"):
        verify.archive_inventory(archive, "windows")


@pytest.mark.parametrize("nonexecutable", ["OvdNext.sh", "python/bin/python3.12"])
def test_linux_package_requires_executable_launcher_and_python(tmp_path, nonexecutable):
    entries = [("OvdNext.sh", b"#!/bin/sh\n"), ("python/bin/python3.12", b"python")]
    modes = {name: 0o100755 for name, _ in entries}
    modes[nonexecutable] = 0o100644
    archive, _ = package(tmp_path, platform="linux", entries=entries, modes=modes)
    with pytest.raises(ValueError, match="executable"):
        verify.archive_inventory(archive, "linux")


def test_linux_inventory_accepts_native_package_notice_paths(tmp_path):
    entries = [("OvdNext.sh", b"#!/bin/sh\n"), ("python/bin/python3.12", b"python"),
               ("lib/notices/system/libcrypt1:amd64/copyright", b"retained license")]
    archive, _ = package(tmp_path, platform="linux", entries=entries,
                         modes={"OvdNext.sh": 0o100755, "python/bin/python3.12": 0o100755})
    inventory = verify.archive_inventory(archive, "linux")
    assert set(inventory) == {"OvdNext.sh", "python/bin/python3.12", "SHA256SUMS",
                              "lib/notices/system/libcrypt1:amd64/copyright"}


def test_extracted_inventory_detects_changes_missing_files_and_added_files(tmp_path):
    archive, _ = package(tmp_path)
    inventory = verify.archive_inventory(archive, "windows")
    with zipfile.ZipFile(archive) as source:
        source.extractall(tmp_path / "relocated")
    bundle = tmp_path / "relocated/OvdNext-windows-x86_64"
    verify.verify_payload(bundle, inventory)
    (bundle / "README.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify.verify_payload(bundle, inventory)
    (bundle / "README.txt").unlink()
    with pytest.raises(ValueError, match="inventory"):
        verify.verify_payload(bundle, inventory)


def probe(report, *, instances=0):
    provenance = {key: value for key, value in report.items() if not key.startswith("archive")}
    return dict(version="0.2.3", isolated=True, native_provenance=provenance,
                capture=dict(scenes=1, instances=instances, presentation="Presentation"))


@pytest.mark.parametrize("field,value", [("source_commit", "c" * 40), ("version", "0.2.2"),
                                         ("source_dirty", True), ("native_sha256", "d" * 64)])
def test_runtime_probe_rejects_provenance_mismatch(tmp_path, field, value):
    _, report = package(tmp_path)
    result = probe(report)
    result["native_provenance"][field] = value
    with pytest.raises(ValueError, match="provenance"):
        verify.validate_probe(result, report, "cloth_physx34.pxd2")


@pytest.mark.parametrize("field,value", [("source_dirty", True), ("allow_dirty", True),
                                         ("dependency_dirty", {"physics": "modified", "ovui": ""})])
def test_runtime_probe_rejects_matching_dirty_provenance(tmp_path, field, value):
    _, report = package(tmp_path)
    report[field] = value
    result = probe(report)
    assert result["native_provenance"][field] == report[field]
    with pytest.raises(ValueError, match="Runtime provenance mismatch"):
        verify.validate_probe(result, report, "cloth_physx34.pxd2")


@pytest.mark.parametrize("field,value", [("scenes", 0), ("presentation", "NoneType"), ("instances", -1)])
def test_runtime_probe_requires_meaningful_capture_presentation(tmp_path, field, value):
    _, report = package(tmp_path)
    result = probe(report)
    result["capture"][field] = value
    with pytest.raises(ValueError, match="capture"):
        verify.validate_probe(result, report, "cloth_physx34.pxd2")


def test_pxd2_zero_instances_at_frame_zero_is_valid_but_empty_ovd_is_not(tmp_path):
    _, report = package(tmp_path)
    verify.validate_probe(probe(report), report, "cloth_physx34.pxd2")
    with pytest.raises(ValueError, match="capture"):
        verify.validate_probe(probe(report), report, "multi_geom.ovd")
    verify.validate_probe(probe(report, instances=14), report, "multi_geom.ovd")


def test_process_failure_preserves_stdout_and_stderr(tmp_path):
    log = tmp_path / "failed.log"
    with pytest.raises(subprocess.CalledProcessError):
        verify.run_logged([sys.executable, "-I", "-c", "import sys; print('retained'); sys.stderr.write('failure'); sys.exit(7)"],
                          log, tmp_path, os.environ.copy())
    assert "retained" in log.read_text() and "failure" in log.read_text()


@pytest.mark.parametrize("failure,error", [
    ("outer_checksum", "Incorrect windows package size"),
    ("missing_resource", "Missing or ambiguous native module"),
    ("wrong_commit", "Unverified windows package provenance"),
    ("missing_notice", "Bundle omits required resource: notices/NOTICES.txt"),
    ("corrupt_font", "Font hash mismatch: Geist-400.ttf"),
    ("qt_runtime", "Forbidden Qt runtime dependency: Qt6Core.dll"),
    ("compiler_runtime", "Linux host compiler runtime must not be bundled:"),
])
def test_real_cli_rejects_invalid_package_and_retains_failure_report(tmp_path, failure, error):
    if failure in {"outer_checksum", "missing_resource", "wrong_commit"}:
        archive, _ = package(tmp_path)
    else:
        extra = {"qt_runtime": [("lib/Qt6Core.dll", b"forbidden Qt")],
                 "compiler_runtime": [("lib/libstdc++.so.6", b"forbidden compiler runtime")]}
        archive, _ = complete_package(tmp_path, omit="app/ovd_ui/notices/NOTICES.txt" if failure == "missing_notice" else None,
                                      corrupt_font=failure == "corrupt_font", extra_entries=extra.get(failure, ()))
    if failure == "outer_checksum":
        with archive.open("ab") as target:
            target.write(b"corrupt")
    work = tmp_path / "evidence"
    result = subprocess.run([sys.executable, "-I", str(ROOT / "ci/verify_ovui_package.py"),
                             "--packages", str(archive.parents[1]), "--platform", "windows",
                             "--version", "0.2.3", "--source-commit", ("c" if failure == "wrong_commit" else "a") * 40,
                             "--work-root", str(work)], capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    report = json.loads((work / "verification.json").read_text())
    assert report["status"] == "failed"
    assert report["error"].startswith("ValueError: " + error), report
    if failure in {"outer_checksum", "wrong_commit"}:
        assert not (work / "relocated package é測試").exists()


def test_existing_work_root_is_preserved(tmp_path):
    archive, _ = package(tmp_path)
    work = tmp_path / "evidence"
    work.mkdir()
    (work / "sentinel").write_text("do not replace")
    result = subprocess.run([sys.executable, str(ROOT / "ci/verify_ovui_package.py"),
                             "--packages", str(archive.parents[1]), "--platform", "windows",
                             "--version", "0.2.3", "--source-commit", "a" * 40, "--work-root", str(work)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert list(work.iterdir()) == [work / "sentinel"]


def complete_package(tmp_path, launcher=b"launcher", *, platform="windows", omit=None, corrupt_font=False, extra_entries=(), pillow_provenance=None, pillow_license_declared=True, corrupt_python_core=False, xml_failure=None, corrupt_python_notice=False):
    assets = ROOT / "python/ovd_ui/assets"
    entries = [("app/ovd_ui/assets/" + path.relative_to(assets).as_posix(), path.read_bytes())
               for path in assets.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    notices = ("NOTICES.txt", "CHANGELOG.md", "OvdNext-LICENSE.txt", "OvdNext-NOTICE.txt",
               "OVUI-LICENSE.txt", "OVUI-THIRD_PARTY_NOTICES.md",
               "Physics-LICENSE.md", "PVD-source-notice.txt", "PxShared-source-notice.txt",
               "pybind11-LICENSE.txt", "Python-LICENSE.txt", "Expat-LICENSE.txt", "HACL-LICENSE.txt")
    entries += [("app/ovd_ui/notices/" + name, b"retained notice") for name in notices]
    entries = [(name, (ROOT / "ci/ovui_package" / Path(name).name).read_bytes()
                if name in {"app/ovd_ui/notices/Expat-LICENSE.txt", "app/ovd_ui/notices/HACL-LICENSE.txt"} else data)
               for name, data in entries]
    if xml_failure == "corrupt_expat_notice":
        entries = [(name, b"incomplete Expat notice" if name == "app/ovd_ui/notices/Expat-LICENSE.txt" else data)
                   for name, data in entries]
    if xml_failure == "corrupt_hacl_notice":
        entries = [(name, b"incomplete HACL notice" if name == "app/ovd_ui/notices/HACL-LICENSE.txt" else data)
                   for name, data in entries]
    info = "app/pillow-13.0.0.dist-info/"
    provenance = security_evidence()["pillow"]["build_provenance"] if pillow_provenance is None else pillow_provenance
    metadata = b"Name: Pillow\nVersion: 13.0.0\n" + (b"License-File: ZLIB-NG.txt\n" if pillow_license_declared else b"")
    entries += [(info + "METADATA", metadata),
                (info + "ovdnext-build.json", json.dumps(provenance).encode()),
                (info + "licenses/ZLIB-NG.txt", ZLIB_LICENSE)]
    windows = platform == "windows"
    launcher_name = "OvdNext.cmd" if windows else "OvdNext.sh"
    python_name = "python/python.exe" if windows else "python/bin/python3.12"
    from install_ovui_python import core_paths

    runtime_files = [("python/" + name, b"python") for name in core_paths(windows)]
    if corrupt_python_core:
        runtime_files[-1] = (runtime_files[-1][0], b"changed ABI core")
    entries += [*runtime_files, ("python/runtime-input-provenance.json", json.dumps(python_input_provenance(platform)).encode())]
    notice_files = python_dependency_notice_fixture(platform)[1]
    entries += [(name, b"changed dependency notice" if corrupt_python_notice and name.endswith("LICENSE.zlib.txt") else data)
                for name, data in notice_files]
    if windows:
        xml_record, xml_entries = xml_build_fixture(tmp_path)
        if xml_failure == "xml_source_mutation":
            xml_record["sources"]["expat"]["sha256"] = "0" * 64
        changed = {"xml_module_mutation": "python/DLLs/_elementtree.pyd",
                   "xml_license_mutation": "python/LICENSE.expat.txt"}.get(xml_failure)
        entries += [(name, b"changed XML payload" if name == changed else data) for name, data in xml_entries]
        entries.append(("python/ovdnext-expat-build.json", json.dumps(xml_record).encode()))
    entries += [("app/_pvd_native." + ("pyd" if windows else "so"), b"native fixture"), ("README.txt", b"portable"),
                ("app/PIL/_imaging." + ("pyd" if windows else "so"), b"pinned Pillow native fixture"),
                (launcher_name, launcher), ("portable_launch.py", b"entry"), *extra_entries]
    entries = [(name, b"corrupt font" if corrupt_font and name == "app/ovd_ui/assets/fonts/Geist-400.ttf" else data)
               for name, data in entries if name != omit]
    archive, report = package(tmp_path, platform=platform, entries=entries,
                              modes={launcher_name: 0o100755, python_name: 0o100755})
    report["native_sha256"] = hashlib.sha256(b"native fixture").hexdigest()
    if windows:
        report["expat_build_provenance"] = xml_record
    for name in ("package-report.json", archive.name + ".package-report.json"):
        (archive.parent / name).write_text(json.dumps(report), encoding="utf-8")
    return archive, report


def completed_child(result, stdout, stderr):
    stdout.write(result.stdout.encode("utf-8"))
    stderr.write(result.stderr.encode("utf-8"))

    def wait(*, timeout):
        assert timeout == 120
        return result.returncode

    return SimpleNamespace(wait=wait, returncode=result.returncode)


@pytest.mark.parametrize("platform", ["windows", "linux"])
@pytest.mark.parametrize("failure", [None, "headless", "gui", "missing_screenshot", "empty_screenshot", "mutation",
                                      "provenance", "nonisolated", "native_hash", "old_python", "old_expat", "missing_expat", "jpeg2000", "missing_png",
                                      "security_provenance", "expat_provenance", "missing_python_input", "python_input_mismatch", "python_core_mutation",
                                      "expat_build_mismatch", "xml_module_mutation", "xml_license_mutation", "xml_source_mutation",
                                      "missing_expat_notice", "corrupt_expat_notice",
                                      "missing_hacl_notice", "corrupt_hacl_notice",
                                      "missing_python_notice", "python_notice_mutation", "python_notice_provenance",
                                      "python_notice_source_mismatch", "python_notice_report_missing",
                                      "missing_pillow_manifest", "wrong_pillow_source_hash",
                                      "missing_pillow_vendor_license", "undeclared_pillow_vendor_license"])
def test_verification_runs_bundled_capture_and_gui_checks_and_retains_evidence(tmp_path, monkeypatch, failure, platform):
    if platform == "linux" and failure in {"expat_build_mismatch", "xml_module_mutation", "xml_license_mutation", "xml_source_mutation"}:
        pytest.skip("Linux Expat is included in the verified base interpreter")
    manifest = security_evidence()["pillow"]["build_provenance"]
    if failure == "wrong_pillow_source_hash":
        manifest["sources"][1]["sha256"] = "0" * 64
    omit = {"missing_python_input": "python/runtime-input-provenance.json",
            "missing_python_notice": "python/licenses/LICENSE.zlib.txt",
            "missing_expat_notice": "app/ovd_ui/notices/Expat-LICENSE.txt",
            "missing_hacl_notice": "app/ovd_ui/notices/HACL-LICENSE.txt",
            "missing_pillow_manifest": "app/pillow-13.0.0.dist-info/ovdnext-build.json",
            "missing_pillow_vendor_license": "app/pillow-13.0.0.dist-info/licenses/ZLIB-NG.txt"}.get(failure)
    archive, report = complete_package(tmp_path, platform=platform, omit=omit, pillow_provenance=manifest,
                                      pillow_license_declared=failure != "undeclared_pillow_vendor_license",
                                      corrupt_python_core=failure == "python_core_mutation", xml_failure=failure,
                                      corrupt_python_notice=failure == "python_notice_mutation")
    if failure == "native_hash":
        report["native_sha256"] = "0" * 64
    if failure == "security_provenance":
        report["python"] = "3.12.14"
    if failure == "expat_provenance":
        report["expat"] = "expat_2.8.5"
    if failure == "python_input_mismatch":
        report["python_input_provenance"]["archive_sha256"] = "0" * 64
    if failure == "expat_build_mismatch":
        report["expat_build_provenance"] = {"fixture": "wrong XML rebuild"}
    if failure == "python_notice_provenance":
        report["python_dependency_notices"] = {"fixture": "wrong notice provenance"}
    if failure == "python_notice_source_mismatch":
        report["python_dependency_notices"]["source_archive"]["sha256"] = "0" * 64
    if failure == "python_notice_report_missing":
        report.pop("python_dependency_notices")
    if failure in {"native_hash", "security_provenance", "expat_provenance", "python_input_mismatch", "expat_build_mismatch",
                   "python_notice_provenance", "python_notice_source_mismatch", "python_notice_report_missing"}:
        for name in ("package-report.json", archive.name + ".package-report.json"):
            (archive.parent / name).write_text(json.dumps(report), encoding="utf-8")
    work = tmp_path / "evidence"
    bundle = work / f"relocated package é測試/OvdNext-{platform}-x86_64"
    for name in ("VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "_OLD_VIRTUAL_PATH"):
        monkeypatch.setenv(name, "untrusted developer environment")
    monkeypatch.setenv("PYTHONPATH", "untrusted developer modules")
    monkeypatch.setenv("CUDA_PATH", "untrusted developer toolkit")
    monkeypatch.setenv("GALLIUM_DRIVER", "llvmpipe")
    monkeypatch.setenv("LIBGL_ALWAYS_SOFTWARE", "1")
    monkeypatch.setenv("GLCONTEXT_WIN_LIBGL", "untrusted developer OpenGL")
    redirected = ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME",
                  "XDG_DATA_HOME", "XDG_CACHE_HOME", "TMP", "TEMP", "TMPDIR")
    for name in redirected:
        monkeypatch.setenv(name, "untrusted developer directory")
    opengl = tmp_path / "opengl fixture.dll"
    opengl.write_bytes(b"OpenGL library fixture")

    def static_linkage(app, windows):
        assert app == bundle / "app" and windows == (platform == "windows")
        assert (app / "PIL" / ("_imaging.pyd" if windows else "_imaging.so")).read_bytes() == b"pinned Pillow native fixture"
        return "verified pinned static zlib-ng linkage"

    monkeypatch.setattr(verify, "validate_pillow_static_linkage", static_linkage)

    def runtime(command, *, cwd, env, stdout, stderr, start_new_session):
        assert Path(cwd) == work
        if command[0] == "unzip":
            assert platform == "linux"
            assert command == ["unzip", "-q", str(archive), "-d", str(bundle.parent)]
            with zipfile.ZipFile(archive) as source:
                source.extractall(bundle.parent)
            return completed_child(subprocess.CompletedProcess(command, 0, "extracted Linux package", ""), stdout, stderr)
        assert not {"VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "_OLD_VIRTUAL_PATH",
                    "PYTHONPATH", "CUDA_PATH", "GLCONTEXT_WIN_LIBGL"} & env.keys()
        for name in redirected:
            assert Path(env[name]) == work / name.lower()
            assert Path(env[name]).is_dir()
        if "-c" in command:
            python = bundle / ("python/python.exe" if platform == "windows" else "python/bin/python3.12")
            assert command[:4] == [str(python), "-I", "-B", "-c"]
            assert command[-1] == str(bundle / "app")
            evidence = security_evidence()
            if failure == "old_python":
                evidence["python"] = "3.12.13"
            if failure == "old_expat":
                evidence["expat"] = "expat_2.8.5"
            if failure == "missing_expat":
                evidence.pop("expat")
            if failure == "jpeg2000":
                evidence["pillow"]["features"]["jpg_2000"] = True
            if failure == "missing_png":
                evidence["pillow"]["png_decode"] = False
            return completed_child(subprocess.CompletedProcess(command, 0, json.dumps(evidence), "runtime security evidence"), stdout, stderr)
        if "--check" in command:
            assert not {"GALLIUM_DRIVER", "LIBGL_ALWAYS_SOFTWARE"} & env.keys()
            capture = Path(command[-1])
            assert capture.parent == work / "captures é測試"
            assert capture.read_bytes() == (ROOT / "test/data" / capture.name).read_bytes()
            assert command[-3] == str(bundle / ("OvdNext.cmd" if platform == "windows" else "OvdNext.sh"))
            result = probe(report, instances=14 if capture.suffix == ".ovd" else 0)
            if failure == "provenance":
                result["native_provenance"]["source_commit"] = "c" * 40
            if failure == "nonisolated":
                result["isolated"] = False
            return completed_child(subprocess.CompletedProcess(command, 7 if failure == "headless" else 0,
                                   json.dumps(result), "headless diagnostic"), stdout, stderr)
        assert env["GALLIUM_DRIVER"] == "llvmpipe"
        assert env["LIBGL_ALWAYS_SOFTWARE"] == "1"
        python = bundle / ("python/python.exe" if platform == "windows" else "python/bin/python3.12")
        assert command[:4] == [str(python), "-I", "-B", str(ROOT / "ci/smoke_ovui_gui.py")]
        assert command[command.index("--bundle") + 1] == str(bundle)
        assert command[command.index("--opengl-library") + 1] == str(opengl.resolve())
        if failure != "missing_screenshot":
            Path(command[command.index("--screenshot") + 1]).write_bytes(b"" if failure == "empty_screenshot" else b"screenshot evidence")
        if failure == "mutation":
            (bundle / "README.txt").write_bytes(b"changed by runtime")
        return completed_child(subprocess.CompletedProcess(command, 9 if failure == "gui" else 0,
                               "GUI evidence", "GUI diagnostic"), stdout, stderr)

    monkeypatch.setattr(verify.subprocess, "Popen", runtime)
    code = verify.main(["--packages", str(archive.parents[1]), "--platform", platform, "--version", "0.2.3",
                        "--source-commit", "a" * 40, "--work-root", str(work), "--opengl-library", str(opengl)])
    evidence = json.loads((work / "verification.json").read_text())
    assert code == (1 if failure else 0), evidence
    assert evidence["status"] == ("failed" if failure else "passed")
    if failure:
        errors = {"headless": "CalledProcessError:", "gui": "CalledProcessError:",
                  "missing_screenshot": "ValueError: GUI smoke produced no screenshot",
                  "empty_screenshot": "ValueError: GUI smoke produced no screenshot",
                  "mutation": "ValueError: Extracted payload checksum mismatch:",
                  "provenance": "ValueError: Runtime provenance mismatch:",
                  "nonisolated": "ValueError: Runtime provenance mismatch:",
                  "native_hash": "ValueError: Native module checksum differs from package provenance",
                  "old_python": "ValueError: Packaged Python must be 3.12.16",
                  "old_expat": "ValueError: Packaged Expat must be expat_2.9.0",
                  "missing_expat": "ValueError: Packaged Expat must be expat_2.9.0",
                  "jpeg2000": "ValueError: Pillow JPEG2000/OpenJPEG support must be disabled",
                  "missing_png": "ValueError: Pillow PNG decoding is required",
                  "security_provenance": "ValueError: Runtime security evidence differs from package provenance",
                  "expat_provenance": "ValueError: Runtime security evidence differs from package provenance",
                  "missing_python_input": "FileNotFoundError:",
                  "python_input_mismatch": "ValueError: Python input provenance differs from package provenance",
                  "python_core_mutation": "ValueError: Pinned CPython input provenance does not match runtime payload",
                  "expat_build_mismatch": "ValueError: Expat build provenance differs from package provenance",
                  "xml_module_mutation": "ValueError: XML rebuild module provenance mismatch",
                  "xml_license_mutation": "ValueError: XML rebuild license provenance mismatch",
                  "xml_source_mutation": "ValueError: XML rebuild source provenance mismatch",
                  "missing_expat_notice": "ValueError: Bundle omits required resource: notices/Expat-LICENSE.txt",
                  "corrupt_expat_notice": "ValueError: Expat license notice hash mismatch",
                  "missing_hacl_notice": "ValueError: Bundle omits required resource: notices/HACL-LICENSE.txt",
                  "corrupt_hacl_notice": "ValueError: HACL license notice hash mismatch",
                  "missing_python_notice": "ValueError: Python dependency notice",
                  "python_notice_mutation": "ValueError: Python dependency notice",
                  "python_notice_provenance": "ValueError: Python dependency notice provenance differs",
                  "python_notice_source_mismatch": "ValueError: Python dependency notice provenance differs",
                  "python_notice_report_missing": "ValueError: Python dependency notice provenance differs",
                  "missing_pillow_manifest": "ValueError: Pillow verified source provenance is required",
                  "wrong_pillow_source_hash": "ValueError: Pillow verified source provenance is required",
                  "missing_pillow_vendor_license": "ValueError: Pillow pinned zlib-ng license is required",
                  "undeclared_pillow_vendor_license": "ValueError: Pillow pinned zlib-ng license is required"}
        assert evidence["error"].startswith(errors[failure]), evidence
    if platform == "linux":
        assert "extracted Linux package" in (work / "extract.log").read_text()
    if not failure:
        assert evidence["runtime_security"] == security_evidence()
        assert "verified pinned static zlib-ng linkage" in (work / "pillow-native-dependencies.txt").read_text()
        assert evidence["captures"] == {
            "multi_geom.ovd": {"scenes": 1, "instances": 14, "presentation": "Presentation"},
            "cloth_physx34.pxd2": {"scenes": 1, "instances": 0, "presentation": "Presentation"},
            "ue4_shooter.pxd2": {"scenes": 1, "instances": 0, "presentation": "Presentation"}}
        assert (work / "gui.png").read_bytes() == b"screenshot evidence"
    if failure not in {"native_hash", "old_python", "old_expat", "missing_expat", "jpeg2000", "missing_png", "security_provenance", "expat_provenance",
                       "missing_python_input", "python_input_mismatch", "python_core_mutation", "expat_build_mismatch",
                       "xml_module_mutation", "xml_license_mutation", "xml_source_mutation",
                       "missing_expat_notice", "corrupt_expat_notice",
                       "missing_hacl_notice", "corrupt_hacl_notice",
                       "missing_python_notice", "python_notice_mutation", "python_notice_provenance",
                       "python_notice_source_mismatch", "python_notice_report_missing",
                       "missing_pillow_manifest", "wrong_pillow_source_hash", "missing_pillow_vendor_license",
                       "undeclared_pillow_vendor_license"}:
        assert "headless diagnostic" in (work / "multi_geom.ovd.log").read_text()
    if failure in {"gui", "missing_screenshot", "empty_screenshot", "mutation"} or not failure:
        assert "GUI diagnostic" in (work / "gui.log").read_text()


def test_process_timeout_preserves_partial_evidence(tmp_path):
    log = tmp_path / "timeout.log"
    with pytest.raises(subprocess.TimeoutExpired):
        verify.run_logged([sys.executable, "-I", "-c", "import sys, time; print('partial stdout', flush=True); "
                           "print('partial stderr', file=sys.stderr, flush=True); time.sleep(8)"],
                          log, tmp_path, os.environ.copy(), timeout=1)
    assert "partial stdout" in log.read_text() and "partial stderr" in log.read_text()


def test_launcher_process_accepts_unicode_directory_and_capture_arguments(tmp_path, monkeypatch):
    script = tmp_path / "runtime_fixture.py"
    platform = "windows" if os.name == "nt" else "linux"
    if platform == "windows":
        launcher = f'@echo off\r\n"{sys.executable}" -I "{script}" %*\r\n'.encode()
    else:
        launcher = f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -I {shlex.quote(str(script))} "$@"\n'.encode()
    archive, report = complete_package(tmp_path, launcher, platform=platform)
    script.write_text("import json, sys\nfrom pathlib import Path\n"
                      "assert sys.argv[1] == '--check'\nassert Path(sys.argv[2]).is_file()\n"
                      f"result = {probe(report, instances=14)!r}\nprint(json.dumps(result))\n", encoding="utf-8")
    work = tmp_path / "evidence"
    actual_popen = subprocess.Popen
    monkeypatch.setattr(verify, "validate_pillow_static_linkage", lambda *args: "static zlib-ng fixture")

    def runtime(command, *, stdout, stderr, **kwargs):
        if "-c" in command:
            return completed_child(subprocess.CompletedProcess(command, 0, json.dumps(security_evidence()), ""), stdout, stderr)
        if "--bundle" not in command:
            return actual_popen(command, stdout=stdout, stderr=stderr, **kwargs)
        Path(command[command.index("--screenshot") + 1]).write_bytes(b"GUI boundary fixture")
        return completed_child(subprocess.CompletedProcess(command, 0, "GUI boundary fixture", ""), stdout, stderr)

    monkeypatch.setattr(verify.subprocess, "Popen", runtime)
    code = verify.main(["--packages", str(archive.parents[1]), "--platform", platform, "--version", "0.2.3",
                        "--source-commit", "a" * 40, "--work-root", str(work)])
    assert code == 0, (work / "verification.json").read_text()
    assert len(json.loads((work / "verification.json").read_text())["captures"]) == 3


def test_timeout_terminates_shell_child_without_waiting_for_inherited_output_handles(tmp_path):
    child = tmp_path / "sleeping_child.py"
    child.write_text("import os, sys, time\nfrom pathlib import Path\n"
                     "Path('child.pid').write_text(str(os.getpid()))\n"
                     "print('retained child stdout', flush=True)\n"
                     "print('retained child stderr', file=sys.stderr, flush=True)\n"
                     "time.sleep(8)\nPath('child-finished').write_text('not terminated')\n")
    if os.name == "nt":
        launcher = tmp_path / "launcher.cmd"
        launcher.write_text(f'@echo off\n"{sys.executable}" -I "{child}"\n')
        command = [launcher]
    else:
        command = ["/bin/sh", "-c", '"$1" -I "$2"; wait', "shell-fixture", sys.executable, child]
    started = time.monotonic()
    log = tmp_path / "timeout.log"
    with pytest.raises(subprocess.TimeoutExpired):
        verify.run_logged(command, log, tmp_path, os.environ.copy(), timeout=1)
    elapsed = time.monotonic() - started
    assert elapsed < 5, f"Timeout waited {elapsed:.2f}s for a surviving shell child"
    assert "retained child stdout" in log.read_text() and "retained child stderr" in log.read_text()
    pid = int((tmp_path / "child.pid").read_text())
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel.OpenProcess(0x100000, False, pid)
        if handle:
            try:
                assert kernel.WaitForSingleObject(handle, 1000) == 0, "Timed-out child is still running"
            finally:
                kernel.CloseHandle(handle)
        else:
            assert ctypes.get_last_error() == 87, "Could not verify that the child exited"
    else:
        # Waiting for the shell does not wait for SIGKILL delivery to its child.
        # Match the Windows exit-observation bound; an orphan may remain a zombie.
        deadline = time.monotonic() + 1
        while True:
            try:
                os.kill(pid, 0)
                state = Path(f"/proc/{pid}/stat").read_text().split()[2]
            except (ProcessLookupError, FileNotFoundError):
                break
            if state == "Z":
                break
            assert time.monotonic() < deadline, f"Timed-out child is still running (state {state})"
            time.sleep(0.01)
    assert not (tmp_path / "child-finished").exists()


def extracted_package_fixture(tmp_path, platform):
    archive, report = complete_package(tmp_path, platform=platform)
    destination = tmp_path / "installed-fixture"
    with zipfile.ZipFile(archive) as stream:
        stream.extractall(destination)
    return destination / f"OvdNext-{platform}-x86_64", report


@pytest.mark.parametrize("platform", ["windows", "linux"])
def test_package_runtime_records_revalidate_and_retain_required_inputs(tmp_path, platform):
    import package_ovui_portable as package

    bundle, report = extracted_package_fixture(tmp_path, platform)
    records = package.runtime_input_records(bundle / "python", platform == "windows")
    assert records["python_input_provenance"] == report["python_input_provenance"]
    assert records["python_dependency_notices"] == report["python_dependency_notices"]
    if platform == "windows":
        assert records["expat_build_provenance"] == report["expat_build_provenance"]
        (bundle / "python/DLLs/_elementtree.pyd").write_bytes(b"foreign XML accelerator")
        with pytest.raises(ValueError, match="module provenance"):
            package.runtime_input_records(bundle / "python", True)
    else:
        assert set(records) == {"python_input_provenance", "python_dependency_notices"}
        (bundle / "python/bin/python3.12").write_bytes(b"foreign Python core")
        with pytest.raises(ValueError, match="runtime payload"):
            package.runtime_input_records(bundle / "python", False)


@pytest.mark.parametrize("platform", ["windows", "linux"])
@pytest.mark.parametrize("corrupt", [False, True, "missing_notice", "changed_notice"])
def test_packaging_main_binds_measured_expat_and_records_before_creating_output(tmp_path, monkeypatch, platform, corrupt):
    import package_ovui_portable as package

    installed, input_report = extracted_package_fixture(tmp_path, platform)
    prefix = installed / "python"
    if corrupt is True:
        changed = prefix / ("DLLs/_elementtree.pyd" if platform == "windows" else "bin/python3.12")
        changed.write_bytes(b"foreign runtime payload")
    if corrupt in {"missing_notice", "changed_notice"}:
        import install_ovui_python

        resources = tmp_path / "source-notices"
        notices = resources / platform
        notices.mkdir(parents=True)
        notice = notices / "LICENSE.zlib.txt"
        notice.write_bytes(b"verified zlib notice")
        manifest = install_ovui_python.load_manifest()
        manifest["platforms"][platform]["dependency_notices"]["files_sha256"] = {
            notice.name: hashlib.sha256(notice.read_bytes()).hexdigest()}
        monkeypatch.setattr(install_ovui_python, "load_manifest", lambda: deepcopy(manifest))
        monkeypatch.setattr(package, "PYTHON_NOTICE_RESOURCES", resources, raising=False)
        if corrupt == "missing_notice":
            notice.unlink()
        else:
            notice.write_bytes(b"altered zlib notice")
    monkeypatch.setattr(package.sys, "platform", "win32" if platform == "windows" else "linux")
    monkeypatch.setattr(package.sys, "base_prefix", str(prefix))
    monkeypatch.setattr(package.sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(package.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(package.platform, "python_version", lambda: "3.12.16")
    monkeypatch.setattr(package, "validate_version", lambda _: "0.2.3")
    extension = ".cp312-win_amd64.pyd" if platform == "windows" else ".cpython-312-x86_64-linux-gnu.so"
    monkeypatch.setattr(package.sysconfig, "get_config_var", lambda name: extension)
    native = tmp_path / "native" / ("_pvd_native" + extension)
    native.parent.mkdir()
    native.write_bytes(b"native fixture")
    versions = dict(line.split("==") for line in (ROOT / "python/requirements-runtime.txt").read_text().splitlines()
                    if line and not line.startswith("#"))
    monkeypatch.setattr(package.metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(package.metadata, "distribution", lambda name: SimpleNamespace(
        name=name, locate_file=lambda _: installed / "app"))
    monkeypatch.setattr(package, "validate_ovui_build", lambda _: None)
    physics, ovui = tmp_path / "physics", tmp_path / "ovui"

    def run(command, **kwargs):
        if command[0] == "git":
            if "status" in command:
                return ""
            if "--format=%ct" in command:
                return "1791501259"
            if "-C" in command:
                return package.PHYSICS_PIN if command[command.index("-C") + 1] == physics else package.OVUI_PIN
            return "a" * 40
        assert "-c" in command
        return json.dumps(security_evidence())

    monkeypatch.setattr(package, "run", run)
    monkeypatch.setattr(package, "native_dependencies", lambda *args: "native fixture dependencies")
    monkeypatch.setattr(package, "copy_python", lambda p, b, w: b / "python" / ("python.exe" if w else "bin/python3.12"))
    monkeypatch.setattr(package, "copy_tree", lambda source, target: target.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(package, "copy_distribution", lambda dist, version, app: {"name": dist.name, "version": version})
    monkeypatch.setattr(package, "stage_libraries", lambda *args: {})
    monkeypatch.setattr(package, "stage_notices", lambda *args: None)
    monkeypatch.setattr(package, "prepare_linux_elf", lambda *args: None)
    recorded = []
    write_json = package.write_json

    def capture_record(path, value):
        if path.name == "runtime-build.json":
            recorded.append(value)
        write_json(path, value)

    def inspect_generated_readme(bundle):
        readme = (bundle / "README.txt").read_text()
        assert "internal nvidia preview" not in readme.casefold()
        assert "internal audience" not in readme.casefold()
        assert "pending distribution approval" not in readme.casefold()
        assert "See app/ovd_ui/notices/, lib/notices/, python/licenses/ and\n" in readme
        assert "native-libraries.json for terms and library inventory." in readme
        raise ValueError("Stop after real package provenance and README construction")

    monkeypatch.setattr(package, "write_json", capture_record)
    monkeypatch.setattr(package, "inspect_bundle", inspect_generated_readme)
    output = tmp_path / "new-output"
    with pytest.raises(SystemExit) as error:
        package.main(["--output", str(output), "--native", str(native), "--physics-root", str(physics), "--ovui-root", str(ovui)])
    assert error.value.code == 1
    if corrupt:
        assert not output.exists()
        assert recorded == []
    else:
        assert len(recorded) == 1
        assert recorded[0]["expat"] == "expat_2.9.0"
        assert recorded[0]["python_input_provenance"] == input_report["python_input_provenance"]
        assert recorded[0]["python_dependency_notices"] == input_report["python_dependency_notices"]
        if platform == "windows":
            assert recorded[0]["expat_build_provenance"] == input_report["expat_build_provenance"]
        else:
            assert "expat_build_provenance" not in recorded[0]
