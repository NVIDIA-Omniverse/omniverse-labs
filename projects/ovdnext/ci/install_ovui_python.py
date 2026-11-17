# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install hash-pinned upstream CPython releases into a job-owned prefix.

Both inputs include Expat 2.9.0. Windows still rebuilds both XML modules with
verified sources and records their exact provenance via install_ovui_expat.py.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

MANIFEST_PATH = Path(__file__).with_name("ovui-python-3.12.16.json")
PROVENANCE_NAME = "runtime-input-provenance.json"


def load_manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_archive(path, entry):
    if Path(path).stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
        raise ValueError("CPython archive size/SHA-256 mismatch")


def download_archive(entry, cache, gh):
    manifest = load_manifest()
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    filename = entry["filename"]
    if Path(filename).name != filename or any(c in filename for c in "\\/:"):
        raise ValueError("Unsafe runtime asset name")
    target = cache / filename
    if target.exists():
        verify_archive(target, entry)
        return target
    # Only this freshly allocated child of the resolved cache is cleaned up.
    with tempfile.TemporaryDirectory(prefix="runtime-download-", dir=cache) as temporary:
        staged = Path(temporary).resolve()
        if not staged.is_relative_to(cache):
            raise ValueError("Runtime download escapes cache")
        subprocess.run([str(gh), "release", "download", manifest["download_tag"],
                        "--repo", manifest["download_repository"], "--pattern", filename,
                        "--dir", str(staged)], check=True)
        downloaded = staged / filename
        verify_archive(downloaded, entry)
        downloaded.replace(target)
    return target


def extract_runtime(archive, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Runtime destination already exists: {destination}")
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        names = set()
        for member in members:
            name = PurePosixPath(member.name)
            if (not name.parts or name.parts[0] != "python" or name.is_absolute()
                    or ".." in name.parts or re.search(r"[\\:\x00-\x1f]", member.name)
                    or not (member.isfile() or member.isdir() or member.issym() or member.islnk())):
                raise ValueError(f"Unsafe runtime archive member: {member.name}")
            canonical = name.as_posix()
            if canonical in names:
                raise ValueError(f"Duplicate runtime archive member: {member.name}")
            names.add(canonical)
            # Validate links as well as files before creating the destination.
            tarfile.data_filter(member, str(destination))
        destination.mkdir(parents=True, exist_ok=False)
        stream.extractall(destination, members=members, filter="data")
    return destination / "python"


def core_paths(windows):
    # Windows' two XML extension modules are rebuilt separately and recorded
    # in ovdnext-expat-build.json. The CPython ABI core must remain unchanged.
    return (("python.exe", "python3.dll", "python312.dll") if windows else
            ("bin/python3.12", "lib/libpython3.12.so", "lib/libpython3.12.so.1.0"))


def input_provenance(windows):
    manifest = load_manifest()
    name = "windows" if windows else "linux"
    entry = manifest["platforms"][name]
    return {"schema": 1, "platform": name, "python_version": manifest["python_version"],
            "base_expat_version": entry["base_expat_version"],
            "required_expat_version": manifest["required_expat_version"],
            "upstream_commit": manifest["upstream_commit"], "archive": entry["filename"],
            "archive_sha256": entry["sha256"]}


def verified_core_hashes(prefix, windows):
    entry = load_manifest()["platforms"]["windows" if windows else "linux"]
    pins = entry.get("core_sha256")
    if (not isinstance(pins, dict) or set(pins) != set(core_paths(windows))
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in pins.values())):
        raise ValueError("Pinned CPython manifest requires complete core SHA-256 pins")
    measured = {name: sha256(Path(prefix) / name) for name in core_paths(windows)}
    if measured != pins:
        raise ValueError("Pinned CPython input provenance does not match runtime payload")
    return measured


def write_provenance(prefix, windows):
    prefix = Path(prefix).resolve()
    provenance = {**input_provenance(windows),
                  "runtime_files": verified_core_hashes(prefix, windows)}
    (prefix / PROVENANCE_NAME).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def read_runtime_provenance(prefix, windows):
    prefix = Path(prefix).resolve()
    provenance = json.loads((prefix / PROVENANCE_NAME).read_text(encoding="utf-8"))
    expected = input_provenance(windows)
    expected["runtime_files"] = verified_core_hashes(prefix, windows)
    if provenance != expected:
        raise ValueError("Pinned CPython input provenance does not match runtime payload")
    return provenance


def probe_runtime(prefix, windows):
    executable = Path(prefix) / ("python.exe" if windows else "bin/python3.12")
    code = ("import json,platform,struct,sys,pyexpat,_elementtree,ssl,zlib; "
            "assert _elementtree.XMLParser().version == pyexpat.EXPAT_VERSION.replace('expat_', 'Expat '); "
            "import xml.etree.ElementTree as ET; ET.fromstring('<ovd/>'); "
            "print(json.dumps({'python':platform.python_version(),'implementation':sys.implementation.name,"
            "'bits':struct.calcsize('P')*8,'expat':pyexpat.EXPAT_VERSION,"
            "'openssl':ssl.OPENSSL_VERSION,'zlib':zlib.ZLIB_RUNTIME_VERSION}))")
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(("PYTHON", "LD_")) and key not in {"GH_TOKEN", "GITHUB_TOKEN"}}
    output = subprocess.run([str(executable), "-I", "-c", code], env=env, cwd=prefix,
                            capture_output=True, text=True, check=True)
    measured = json.loads(output.stdout)
    entry = load_manifest()["platforms"]["windows" if windows else "linux"]
    if (measured["python"] != "3.12.16" or measured["implementation"] != "cpython"
            or measured["bits"] != 64 or measured["expat"] != entry["base_expat_version"]):
        raise ValueError(f"Unexpected CPython runtime identity: {measured}")
    return measured


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--archive", type=Path, help="Verify and use an already downloaded input")
    parser.add_argument("--reuse-verified", action="store_true",
                        help="Reuse an existing Linux runtime after provenance and identity checks")
    parser.add_argument("--gh", default="gh")
    args = parser.parse_args(argv)
    if sys.platform not in {"win32", "linux"} or platform.machine().lower() not in {"amd64", "x86_64"}:
        parser.error("Requires Windows/Linux x86_64")
    windows = sys.platform == "win32"
    if args.reuse_verified and (windows or args.archive):
        parser.error("--reuse-verified is Linux-only and cannot be combined with --archive")
    entry = load_manifest()["platforms"]["windows" if windows else "linux"]
    work = args.work_root.resolve()
    work.mkdir(parents=True, exist_ok=True)
    destination = work / "runtime"
    if args.reuse_verified and (destination.exists() or destination.is_symlink()):
        prefix = destination / "python"
        try:
            read_runtime_provenance(prefix, False)
            measured = probe_runtime(prefix, False)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            print(f"{destination} is incomplete or unverified: {error}; "
                  "move it aside and rerun ci/setup_ovui.sh.", file=sys.stderr)
            return 1
        (work / "input-verification.json").write_text(json.dumps(measured, indent=2) + "\n", encoding="utf-8")
        print(prefix / "bin/python3.12")
        return 0
    source = args.archive or download_archive(entry, work / "cache", args.gh)
    verify_archive(source, entry)
    prefix = extract_runtime(source, destination)
    measured = probe_runtime(prefix, windows)
    write_provenance(prefix, windows)
    (work / "input-verification.json").write_text(json.dumps(measured, indent=2) + "\n", encoding="utf-8")
    print(prefix / ("python.exe" if windows else "bin/python3.12"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
