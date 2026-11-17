# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verified portable interpreter inputs must fail closed before extraction/use."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("install_ovui_python", ROOT / "ci/install_ovui_python.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def archive(tmp_path, names):
    target = tmp_path / "runtime.tar.gz"
    with tarfile.open(target, "w:gz") as stream:
        for name in names:
            member = tarfile.TarInfo(name)
            member.size = 4
            stream.addfile(member, io.BytesIO(b"test"))
    return target


def test_checksum_mismatch_rejected_before_extraction(tmp_path):
    source = archive(tmp_path, ["python/Lib/example.py"])
    with pytest.raises(ValueError, match="SHA-256"):
        runtime.verify_archive(source, {"bytes": source.stat().st_size, "sha256": "0" * 64})


@pytest.mark.parametrize("name", ["../outside", "/outside", "python/../../outside", "other/file", "python/file:stream", "python\\outside"])
def test_unsafe_runtime_path_rejected_without_writing(tmp_path, name):
    source = archive(tmp_path, [name])
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="Unsafe"):
        runtime.extract_runtime(source, destination)
    assert not destination.exists()


def test_duplicate_runtime_member_rejected_before_extraction(tmp_path):
    source = archive(tmp_path, ["python/Lib/example.py", "python/Lib/example.py"])
    with pytest.raises(ValueError, match="Duplicate"):
        runtime.extract_runtime(source, tmp_path / "extracted")


def test_runtime_symlink_cannot_escape_destination(tmp_path):
    source = tmp_path / "runtime.tar.gz"
    with tarfile.open(source, "w:gz") as stream:
        regular = tarfile.TarInfo("python/regular")
        regular.size = 4
        stream.addfile(regular, io.BytesIO(b"test"))
        member = tarfile.TarInfo("python/escape")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        stream.addfile(member)
    with pytest.raises((ValueError, tarfile.FilterError)):
        runtime.extract_runtime(source, tmp_path / "extracted")
    assert not (tmp_path / "outside").exists()
    assert not (tmp_path / "extracted").exists()


def test_runtime_extraction_preserves_existing_destination(tmp_path):
    source = archive(tmp_path, ["python/Lib/example.py"])
    destination = tmp_path / "extracted"
    destination.mkdir()
    sentinel = destination / "keep"
    sentinel.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        runtime.extract_runtime(source, destination)
    assert sentinel.read_bytes() == b"original"


def test_verified_archive_extracts_only_python_root(tmp_path):
    source = archive(tmp_path, ["python/Lib/example.py"])
    runtime.verify_archive(source, {"bytes": source.stat().st_size, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    prefix = runtime.extract_runtime(source, tmp_path / "extracted")
    assert prefix == tmp_path / "extracted/python"
    assert (prefix / "Lib/example.py").read_bytes() == b"test"


CORE_PATHS = {
    True: ("python.exe", "python3.dll", "python312.dll"),
    False: ("bin/python3.12", "lib/libpython3.12.so", "lib/libpython3.12.so.1.0"),
}


def prepared_prefix(tmp_path, monkeypatch, windows):
    prefix = tmp_path / "python"
    manifest = runtime.load_manifest()
    manifest["platforms"]["windows" if windows else "linux"]["core_sha256"] = {
        name: hashlib.sha256(b"original").hexdigest() for name in CORE_PATHS[windows]}
    monkeypatch.setattr(runtime, "load_manifest", lambda: manifest)
    for name in CORE_PATHS[windows]:
        target = prefix / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"original")
    return prefix, manifest


@pytest.mark.parametrize("windows", [True, False])
def test_core_paths_cover_every_pinned_abi_file(windows):
    assert runtime.core_paths(windows) == CORE_PATHS[windows]


@pytest.mark.parametrize(("windows", "name"), [(w, name) for w, names in CORE_PATHS.items() for name in names])
def test_runtime_provenance_detects_changed_core(tmp_path, monkeypatch, windows, name):
    prefix, _ = prepared_prefix(tmp_path, monkeypatch, windows)
    runtime.write_provenance(prefix, windows)
    assert runtime.read_runtime_provenance(prefix, windows)["required_expat_version"] == "expat_2.9.0"
    (prefix / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="runtime payload"):
        runtime.read_runtime_provenance(prefix, windows)


@pytest.mark.parametrize("windows", [True, False])
def test_provenance_writer_rejects_foreign_core_before_creating_marker(tmp_path, monkeypatch, windows):
    prefix, _ = prepared_prefix(tmp_path, monkeypatch, windows)
    (prefix / CORE_PATHS[windows][0]).write_bytes(b"foreign interpreter")
    with pytest.raises(ValueError, match="runtime payload"):
        runtime.write_provenance(prefix, windows)
    assert not (prefix / runtime.PROVENANCE_NAME).exists()


@pytest.mark.parametrize("windows", [True, False])
def test_provenance_reader_rejects_core_and_record_changed_together(tmp_path, monkeypatch, windows):
    prefix, _ = prepared_prefix(tmp_path, monkeypatch, windows)
    record = runtime.write_provenance(prefix, windows)
    name = CORE_PATHS[windows][0]
    (prefix / name).write_bytes(b"foreign interpreter")
    record["runtime_files"][name] = hashlib.sha256(b"foreign interpreter").hexdigest()
    (prefix / runtime.PROVENANCE_NAME).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="runtime payload"):
        runtime.read_runtime_provenance(prefix, windows)


@pytest.mark.parametrize("windows", [True, False])
@pytest.mark.parametrize("damage", ["missing", "empty", "extra", "partial", "malformed", "non-string", "non-map"])
def test_provenance_requires_complete_manifest_core_pins(tmp_path, monkeypatch, windows, damage):
    prefix, manifest = prepared_prefix(tmp_path, monkeypatch, windows)
    entry = manifest["platforms"]["windows" if windows else "linux"]
    if damage == "missing":
        entry.pop("core_sha256")
    elif damage == "empty":
        entry["core_sha256"] = {}
    elif damage == "extra":
        entry["core_sha256"]["unexpected"] = "a" * 64
    elif damage == "partial":
        entry["core_sha256"].pop(CORE_PATHS[windows][0])
    elif damage == "malformed":
        entry["core_sha256"][CORE_PATHS[windows][0]] = "not-a-sha256"
    elif damage == "non-string":
        entry["core_sha256"][CORE_PATHS[windows][0]] = 123
    else:
        entry["core_sha256"] = []
    with pytest.raises(ValueError, match="core SHA-256"):
        runtime.write_provenance(prefix, windows)
    assert not (prefix / runtime.PROVENANCE_NAME).exists()


@pytest.mark.parametrize("windows", [True, False])
@pytest.mark.parametrize("field", ["platform", "python_version", "base_expat_version", "required_expat_version",
                                   "upstream_commit", "archive", "archive_sha256", "schema"])
def test_provenance_reader_rejects_changed_identity(tmp_path, monkeypatch, windows, field):
    prefix, _ = prepared_prefix(tmp_path, monkeypatch, windows)
    record = runtime.write_provenance(prefix, windows)
    record[field] = "tampered"
    (prefix / runtime.PROVENANCE_NAME).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="runtime payload"):
        runtime.read_runtime_provenance(prefix, windows)


def measured_runtime(windows):
    return {"python": "3.12.16", "implementation": "cpython", "bits": 64,
            "expat": "expat_2.9.0",
            "openssl": "OpenSSL test", "zlib": "test"}


@pytest.mark.parametrize("windows", [True, False])
@pytest.mark.parametrize(("field", "wrong"), [("python", "3.12.15"), ("implementation", "pypy"),
                                               ("bits", 32), ("expat", "expat_2.8.4")])
def test_runtime_probe_rejects_incorrect_identity(tmp_path, monkeypatch, windows, field, wrong):
    measured = measured_runtime(windows)
    measured[field] = wrong
    monkeypatch.setattr(runtime.subprocess, "run", lambda args, **kwargs:
        subprocess.CompletedProcess(args, 0, json.dumps(measured), ""))
    with pytest.raises(ValueError, match="identity"):
        runtime.probe_runtime(tmp_path, windows)


@pytest.mark.parametrize("windows", [True, False])
def test_runtime_probe_runs_isolated_without_bootstrap_credentials(tmp_path, monkeypatch, windows):
    monkeypatch.setenv("GH_TOKEN", "dummy-only")
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-only")
    monkeypatch.setenv("PYTHONPATH", "foreign")
    monkeypatch.setenv("LD_LIBRARY_PATH", "foreign")
    calls = []
    def execute(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, json.dumps(measured_runtime(windows)), "")
    monkeypatch.setattr(runtime.subprocess, "run", execute)
    assert runtime.probe_runtime(tmp_path, windows) == measured_runtime(windows)
    args, options = calls[0]
    assert args[:3] == [str(tmp_path / ("python.exe" if windows else "bin/python3.12")), "-I", "-c"]
    assert not {"GH_TOKEN", "GITHUB_TOKEN", "PYTHONPATH", "LD_LIBRARY_PATH"} & options["env"].keys()
    assert "_elementtree.XMLParser()" in args[3]


@pytest.mark.parametrize("windows", [True, False])
@pytest.mark.parametrize("damage", ["sha256", "bytes"])
def test_installer_rejects_supplied_bad_archive_before_extraction(tmp_path, monkeypatch, windows, damage):
    monkeypatch.setattr(runtime.sys, "platform", "win32" if windows else "linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    source = archive(tmp_path, ["python/Lib/example.py"])
    manifest = runtime.load_manifest()
    entry = manifest["platforms"]["windows" if windows else "linux"]
    entry.update(bytes=source.stat().st_size, sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    entry[damage] = "0" * 64 if damage == "sha256" else entry["bytes"] + 1
    monkeypatch.setattr(runtime, "load_manifest", lambda: manifest)
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="SHA-256"):
        runtime.main(["--work-root", str(work), "--archive", str(source)])
    assert not (work / "runtime").exists()


@pytest.mark.parametrize("windows", [True, False])
def test_installer_verifies_and_records_fresh_supplied_runtime(tmp_path, monkeypatch, capsys, windows):
    monkeypatch.setattr(runtime.sys, "platform", "win32" if windows else "linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    source = archive(tmp_path, ["python/" + name for name in CORE_PATHS[windows]])
    manifest = runtime.load_manifest()
    entry = manifest["platforms"]["windows" if windows else "linux"]
    entry.update(bytes=source.stat().st_size, sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 core_sha256={name: hashlib.sha256(b"test").hexdigest() for name in CORE_PATHS[windows]})
    monkeypatch.setattr(runtime, "load_manifest", lambda: manifest)
    monkeypatch.setattr(runtime.subprocess, "run", lambda args, **kwargs:
        subprocess.CompletedProcess(args, 0, json.dumps(measured_runtime(windows)), ""))
    work = tmp_path / "work"
    assert runtime.main(["--work-root", str(work), "--archive", str(source)]) == 0
    prefix = work / "runtime/python"
    assert capsys.readouterr().out.strip() == str(prefix / ("python.exe" if windows else "bin/python3.12"))
    assert runtime.read_runtime_provenance(prefix, windows)["runtime_files"] == entry["core_sha256"]
    assert json.loads((work / "input-verification.json").read_text()) == measured_runtime(windows)


@pytest.mark.parametrize("cached", [True, False])
def test_download_rejects_bad_cache_or_download_before_publishing(tmp_path, monkeypatch, cached):
    cache = tmp_path / "cache"
    cache.mkdir()
    entry = {"filename": "runtime.tar.gz", "bytes": 5, "sha256": hashlib.sha256(b"right").hexdigest()}
    target = cache / entry["filename"]
    if cached:
        target.write_bytes(b"wrong")
    def download(args, **kwargs):
        assert not cached, "A bad cache entry must not be silently replaced"
        (Path(args[args.index("--dir") + 1]) / entry["filename"]).write_bytes(b"wrong")
    monkeypatch.setattr(runtime.subprocess, "run", download)
    with pytest.raises(ValueError, match="SHA-256"):
        runtime.download_archive(entry, cache, "gh")
    assert sorted(p.name for p in cache.iterdir()) == ([entry["filename"]] if cached else [])
    if cached:
        assert target.read_bytes() == b"wrong"


@pytest.mark.parametrize("cached", [True, False])
def test_download_accepts_verified_cache_or_publishes_verified_download(tmp_path, monkeypatch, cached):
    cache = tmp_path / "cache"
    cache.mkdir()
    entry = {"filename": "runtime.tar.gz", "bytes": 5, "sha256": hashlib.sha256(b"right").hexdigest()}
    target = cache / entry["filename"]
    if cached:
        target.write_bytes(b"right")
    def download(args, **kwargs):
        assert not cached, "A verified cache entry must be reused"
        (Path(args[args.index("--dir") + 1]) / entry["filename"]).write_bytes(b"right")
    monkeypatch.setattr(runtime.subprocess, "run", download)
    assert runtime.download_archive(entry, cache, "gh") == target
    assert target.read_bytes() == b"right"
    assert sorted(p.name for p in cache.iterdir()) == [entry["filename"]]


@pytest.mark.parametrize(("windows", "extra"), [(True, []), (False, ["--archive", "unused.tar.gz"])])
def test_reuse_is_linux_only_and_never_ignores_explicit_archive(tmp_path, monkeypatch, windows, extra):
    monkeypatch.setattr(runtime.sys, "platform", "win32" if windows else "linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    work = tmp_path / "work"
    with pytest.raises(SystemExit) as error:
        runtime.main(["--work-root", str(work), "--reuse-verified", *extra])
    assert error.value.code == 2
    assert not work.exists()


def test_installer_reuses_verified_linux_runtime_and_probes_before_returning(tmp_path, monkeypatch, capsys):
    prefix, _ = prepared_prefix(tmp_path / "work/runtime", monkeypatch, False)
    runtime.write_provenance(prefix, False)
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    calls = []
    def probe(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, json.dumps(measured_runtime(False)), "")
    monkeypatch.setattr(runtime.subprocess, "run", probe)
    monkeypatch.setattr(runtime, "download_archive", lambda *args: pytest.fail("verified runtime must not download"))
    assert runtime.main(["--work-root", str(tmp_path / "work"), "--reuse-verified"]) == 0
    assert capsys.readouterr().out.strip() == str(prefix / "bin/python3.12")
    assert calls[0][0] == str(prefix / "bin/python3.12")
    assert (prefix / "bin/python3.12").read_bytes() == b"original"


@pytest.mark.parametrize("damage", ["missing-marker", "tampered-core", "wrong-identity"])
def test_linux_reuse_fails_closed_with_remediation(tmp_path, monkeypatch, capsys, damage):
    prefix, _ = prepared_prefix(tmp_path / "work/runtime", monkeypatch, False)
    runtime.write_provenance(prefix, False)
    if damage == "missing-marker":
        (prefix / runtime.PROVENANCE_NAME).unlink()
    elif damage == "tampered-core":
        (prefix / "bin/python3.12").write_bytes(b"foreign")
    measured = measured_runtime(False)
    if damage == "wrong-identity":
        measured["expat"] = "expat_2.8.5"
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime.subprocess, "run", lambda args, **kwargs:
        subprocess.CompletedProcess(args, 0, json.dumps(measured), ""))
    monkeypatch.setattr(runtime, "download_archive", lambda *args: pytest.fail("unverified runtime must not download"))
    assert runtime.main(["--work-root", str(tmp_path / "work"), "--reuse-verified"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "incomplete or unverified" in output.err and "move it aside" in output.err
    assert str(tmp_path / "work/runtime") in output.err
    assert prefix.exists()
