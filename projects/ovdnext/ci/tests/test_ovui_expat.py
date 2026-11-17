# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""XML rebuild inputs and shipped module provenance must fail closed."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci"))
SPEC = importlib.util.spec_from_file_location("install_ovui_expat", ROOT / "ci/install_ovui_expat.py")
expat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(expat)


def archive(tmp_path, names):
    target = tmp_path / "source.tar.gz"
    with tarfile.open(target, "w:gz") as stream:
        for name in names:
            member = tarfile.TarInfo(name)
            member.size = 4
            stream.addfile(member, io.BytesIO(b"test"))
    return target


def test_source_checksum_mismatch_rejected(tmp_path):
    target = archive(tmp_path, ["source/file.c"])
    with pytest.raises(ValueError, match="SHA-256"):
        expat.verify_source_archive(target, {"bytes": target.stat().st_size, "sha256": "0" * 64})


@pytest.mark.parametrize("name", ["../escape", "source/../../escape", "/escape", "wrong/file", "source/file:stream", "source\\file"])
def test_source_archive_cannot_write_outside_root(tmp_path, name):
    source = archive(tmp_path, [name])
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="Unsafe"):
        expat.extract_sources(source, destination, "source")
    assert not destination.exists()


def test_source_links_rejected_before_writing(tmp_path):
    source = tmp_path / "linked.tar.gz"
    with tarfile.open(source, "w:gz") as stream:
        member = tarfile.TarInfo("source/linked")
        member.type = tarfile.SYMTYPE
        member.linkname = "outside"
        stream.addfile(member)
    with pytest.raises(ValueError, match="Unsafe"):
        expat.extract_sources(source, tmp_path / "extracted", "source")
    assert not (tmp_path / "extracted").exists()


def test_source_duplicates_rejected(tmp_path):
    source = archive(tmp_path, ["source/file", "source/file"])
    with pytest.raises(ValueError, match="Duplicate"):
        expat.extract_sources(source, tmp_path / "extracted", "source")


def test_existing_source_destination_preserved(tmp_path):
    source = archive(tmp_path, ["source/file"])
    destination = tmp_path / "extracted"
    destination.mkdir()
    sentinel = destination / "keep"
    sentinel.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        expat.extract_sources(source, destination, "source")
    assert sentinel.read_bytes() == b"original"


def prepared_prefix(tmp_path, monkeypatch):
    prefix = tmp_path / "python"
    prefix.mkdir()
    input_record = {"archive_sha256": "a" * 64}
    monkeypatch.setattr(expat.runtime, "read_runtime_provenance", lambda path, windows: input_record)
    for name in expat.MODULE_PATHS:
        target = prefix / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(name.encode())
    for name in expat.LICENSE_HASHES:
        (prefix / name).write_bytes(name.encode())
    monkeypatch.setattr(expat, "LICENSE_HASHES", {name: hashlib.sha256(name.encode()).hexdigest() for name in expat.LICENSE_HASHES})
    record = expat.make_build_provenance(prefix, input_record, {"name": "MSVC", "python_compiler": "MSC v.1944"})
    (prefix / expat.PROVENANCE_NAME).write_text(json.dumps(record), encoding="utf-8")
    return prefix, record


def test_provenance_binds_both_modules(tmp_path, monkeypatch):
    prefix, record = prepared_prefix(tmp_path, monkeypatch)
    assert expat.read_build_provenance(prefix) == record
    (prefix / "DLLs/_elementtree.pyd").write_bytes(b"old accelerator")
    with pytest.raises(ValueError, match="module"):
        expat.read_build_provenance(prefix)


def test_provenance_binds_exact_input_archive(tmp_path, monkeypatch):
    prefix, record = prepared_prefix(tmp_path, monkeypatch)
    record["input_archive_sha256"] = "b" * 64
    (prefix / expat.PROVENANCE_NAME).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="input"):
        expat.read_build_provenance(prefix)


def test_provenance_rejects_changed_source_pin(tmp_path, monkeypatch):
    prefix, record = prepared_prefix(tmp_path, monkeypatch)
    record["sources"]["expat"]["sha256"] = "0" * 64
    (prefix / expat.PROVENANCE_NAME).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        expat.read_build_provenance(prefix)


def test_provenance_binds_required_licenses(tmp_path, monkeypatch):
    prefix, record = prepared_prefix(tmp_path, monkeypatch)
    (prefix / "LICENSE.expat.txt").write_bytes(b"removed license")
    with pytest.raises(ValueError, match="license"):
        expat.read_build_provenance(prefix)


def test_failing_xml_regressions_keep_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(expat, "probe_xml", lambda *args: {"expat": "expat_2.9.0"})
    calls = []

    def run(*args):
        calls.append(args)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, "XML tests", output="test output\n", stderr="regression failure\n")
        return subprocess.CompletedProcess("mixed CAPI", 0, "", "")

    monkeypatch.setattr(expat, "run_python", run)
    with pytest.raises(RuntimeError, match="XML tests failed"):
        expat.validate_build("python", tmp_path / "artifacts", tmp_path / "sources", tmp_path)
    assert (tmp_path / "xml-regression.log").read_text() == "test output\nregression failure\n"


@pytest.mark.skipif(sys.platform != "win32" or sys.version_info[:3] != (3, 12, 16),
                    reason="Requires the pinned Windows runtime")
def test_current_runtime_accepts_matching_xml_capi(tmp_path, monkeypatch):
    """The upstream accelerator now already embeds the same Expat version."""
    run_python = expat.run_python

    def run(executable, code, arguments=()):
        if "unittest" in code:
            return subprocess.CompletedProcess("suite covered separately", 0, "", "Ran 490 tests in 0.1s\nOK\n")
        return run_python(executable, code, arguments)

    monkeypatch.setattr(expat, "run_python", run)
    measured, validation = expat.validate_build(
        sys.executable, Path(sys.base_prefix) / "DLLs", tmp_path, tmp_path)
    assert measured["expat"] == "expat_2.9.0"
    assert validation["baseline_capi_compatible"] is True


@pytest.mark.parametrize("cached", [True, False])
def test_source_archive_rejects_wrong_bytes_before_use(tmp_path, monkeypatch, cached):
    entry = expat.SOURCES["expat"]
    wrong = b"x" * entry["bytes"]
    if cached:
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / entry["filename"]).write_bytes(wrong)
        monkeypatch.setattr(expat.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Cache validation must not download"))
    else:
        cache = None
        monkeypatch.setattr(expat.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(wrong))
    with pytest.raises(ValueError, match="SHA-256"):
        expat.source_archive(entry, cache, tmp_path)


@pytest.mark.parametrize("field,value", [("schema", 2), ("platform", "linux"),
                                         ("python_version", "3.12.14"), ("expat_version", "expat_2.8.5"),
                                         ("abi", "cp313-win_amd64")])
def test_rebuild_record_rejects_changed_identity(tmp_path, monkeypatch, field, value):
    prefix, record = prepared_prefix(tmp_path, monkeypatch)
    record[field] = value
    (prefix / expat.PROVENANCE_NAME).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="identity provenance"):
        expat.read_build_provenance(prefix)


def test_rebuild_record_rejects_changed_license_hash_even_when_file_is_intact(tmp_path, monkeypatch):
    prefix, record = prepared_prefix(tmp_path, monkeypatch)
    record["licenses"]["LICENSE.expat.txt"] = "0" * 64
    (prefix / expat.PROVENANCE_NAME).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="license provenance"):
        expat.read_build_provenance(prefix)


@pytest.mark.parametrize("count,accepted", [(12, False), (399, False), (400, True), (490, True)])
def test_validation_requires_full_xml_suite_and_preserves_counts(tmp_path, monkeypatch, count, accepted):
    monkeypatch.setattr(expat, "probe_xml", lambda *args: {"expat": "expat_2.9.0"})
    calls = []
    output = f"Ran {count} tests in 0.1s\n\nOK (skipped=12)\n"

    def run(*args):
        calls.append(args)
        return subprocess.CompletedProcess("XML subprocess", 0, "", output if len(calls) == 2 else "")

    monkeypatch.setattr(expat, "run_python", run)
    if accepted:
        measured, validation = expat.validate_build("python", tmp_path / "artifacts", tmp_path / "sources", tmp_path)
        assert measured == {"expat": "expat_2.9.0"}
        assert validation == {"native_xml_imports": True, "baseline_capi_compatible": True,
                              "xml_regression_tests": count, "xml_regression_skips": 12}
    else:
        with pytest.raises(ValueError, match="expected suite"):
            expat.validate_build("python", tmp_path / "artifacts", tmp_path / "sources", tmp_path)
    assert (tmp_path / "xml-regression.log").read_text() == output


def module_install_fixture(tmp_path, monkeypatch, previous_licenses=False):
    prefix = tmp_path / "runtime"
    original = {"DLLs/pyexpat.pyd": b"original parser", "DLLs/_elementtree.pyd": b"original accelerator"}
    if previous_licenses:
        original.update({"LICENSE.expat.txt": b"previous Expat license",
                         "LICENSE.cpython-xml.txt": b"previous Python license"})
    for name, data in original.items():
        path = prefix / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    work = tmp_path / "build"
    (work / "artifacts").mkdir(parents=True)
    for name in ("pyexpat", "_elementtree"):
        (work / "artifacts" / f"{name}.cp312-win_amd64.pyd").write_bytes(("rebuilt " + name).encode())
    for name, root, filename in (("LICENSE.cpython-xml.txt", "Python-3.12.16", "LICENSE"),
                                 ("LICENSE.expat.txt", "expat-2.9.0", "COPYING")):
        (work / root).mkdir()
        (work / root / filename).write_bytes(name.encode())
    monkeypatch.setattr(expat, "LICENSE_HASHES", {name: hashlib.sha256(name.encode()).hexdigest()
                                               for name in ("LICENSE.cpython-xml.txt", "LICENSE.expat.txt")})
    input_record = {"archive_sha256": "a" * 64}
    monkeypatch.setattr(expat.runtime, "read_runtime_provenance", lambda path, windows: input_record)
    return prefix, work, original, input_record


@pytest.mark.parametrize("phase", ["copy", "probe", "marker"])
@pytest.mark.parametrize("previous_licenses", [False, True])
def test_failed_module_install_restores_prefix_without_new_files(tmp_path, monkeypatch, phase, previous_licenses):
    prefix, work, original, input_record = module_install_fixture(tmp_path, monkeypatch, previous_licenses)

    def probe(executable, expected):
        assert executable == prefix / "python.exe" and expected == "expat_2.9.0"
        assert (prefix / "DLLs/_elementtree.pyd").read_bytes() == b"rebuilt _elementtree"
        if phase == "probe":
            raise ValueError("Rejected installed XML probe")
        return {"expat": "expat_2.9.0"}

    monkeypatch.setattr(expat, "probe_xml", probe)
    if phase == "copy":
        copy2 = expat.shutil.copy2

        def copy(source, target):
            if str(target).endswith("_elementtree.pyd.new"):
                Path(target).write_bytes(b"partial staged module")
                raise OSError("Rejected staged copy")
            return copy2(source, target)

        monkeypatch.setattr(expat.shutil, "copy2", copy)
    if phase == "marker":
        def reject_record(path):
            assert (path / expat.PROVENANCE_NAME).is_file()
            raise ValueError("Rejected installed XML marker")

        monkeypatch.setattr(expat, "read_build_provenance", reject_record)
    with pytest.raises((ValueError, OSError), match="Rejected"):
        expat.install_modules(prefix, work, input_record, {"name": "MSVC"}, {})
    actual = {path.relative_to(prefix).as_posix(): path.read_bytes() for path in prefix.rglob("*") if path.is_file()}
    assert actual == original


def test_successful_module_install_probes_and_binds_both_modules(tmp_path, monkeypatch):
    prefix, work, _, input_record = module_install_fixture(tmp_path, monkeypatch)
    probes = []

    def probe(executable, expected):
        probes.append((executable, expected))
        assert (prefix / "DLLs/_elementtree.pyd").read_bytes() == b"rebuilt _elementtree"
        return {"expat": "expat_2.9.0"}

    monkeypatch.setattr(expat, "probe_xml", probe)
    record = expat.install_modules(prefix, work, input_record, {"name": "MSVC"}, {"xml_regression_tests": 490})
    assert probes == [(prefix / "python.exe", "expat_2.9.0")]
    assert expat.read_build_provenance(prefix) == record
    assert record["runtime_files"] == {
        "DLLs/pyexpat.pyd": hashlib.sha256(b"rebuilt pyexpat").hexdigest(),
        "DLLs/_elementtree.pyd": hashlib.sha256(b"rebuilt _elementtree").hexdigest()}
    assert not list(prefix.rglob("*.new"))


def windows_cli_fixture(tmp_path, monkeypatch):
    prefix, record = prepared_prefix(tmp_path, monkeypatch)
    monkeypatch.setattr(expat.sys, "platform", "win32")
    monkeypatch.setattr(expat.sys, "base_prefix", str(prefix))
    monkeypatch.setattr(expat.sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(expat.platform, "python_version", lambda: "3.12.16")
    monkeypatch.setattr(expat.platform, "machine", lambda: "AMD64")
    return prefix, record


def test_main_revalidates_reused_module_bytes(tmp_path, monkeypatch):
    prefix, _ = windows_cli_fixture(tmp_path, monkeypatch)
    (prefix / "DLLs/_elementtree.pyd").write_bytes(b"foreign accelerator")
    monkeypatch.setattr(expat, "probe_xml", lambda *a: pytest.fail("Corrupted marker must fail before execution"))
    with pytest.raises(ValueError, match="module provenance"):
        expat.main(["--work-root", str(tmp_path / "build")])
    assert not (tmp_path / "build/build-report.json").exists()


def test_main_rejects_work_directory_inside_managed_prefix(tmp_path, monkeypatch):
    prefix, _ = windows_cli_fixture(tmp_path, monkeypatch)
    work = prefix / "nested-build"
    with pytest.raises(SystemExit) as error:
        expat.main(["--work-root", str(work)])
    assert error.value.code == 2
    assert not work.exists()


@pytest.mark.parametrize("name", ["setuptools", "wheel"])
def test_build_rejects_unpinned_tools_before_writing_or_compiling(tmp_path, monkeypatch, name):
    monkeypatch.setattr(expat.importlib.metadata, "version", lambda requested:
                        "0.0.0" if requested == name else {"setuptools": "80.9.0", "wheel": "0.45.1"}[requested])
    with pytest.raises(ValueError, match=f"requires {name}=="):
        expat.build_modules(tmp_path, tmp_path / "missing-python")
    assert not (tmp_path / "setup.py").exists()
