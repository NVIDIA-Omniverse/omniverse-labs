#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retired Linux wheelhouse entry point; helpers retain historical validation.

Use ci/package_ovui_portable.py with the prepared application runtime instead.
The wheelhouse helpers and bundle scripts remain as historical regression
fixtures; release_version still serves the shared version checks.
"""
from __future__ import annotations

from email.parser import Parser
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def run(command, *, cwd=ROOT, env=None, capture=False):
    result = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                            check=True, text=True, stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else ""


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def release_version(changelog):
    first = next((line for line in changelog.splitlines() if line.startswith("## ")), "")
    match = re.fullmatch(r"## v((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)) \(\d{4}-\d{2}-\d{2}\)", first)
    if not match:
        raise ValueError("CHANGELOG.md must begin its release history with a canonical semantic version")
    return match[1]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def reject_qt_runtime(identity):
    if re.match(r"^(?:lib)?(?:qt[0-9]*[a-z]|qwt(?:[-_.0-9]|$)|pyside[0-9]*(?:[-_.]|$)|pyqt[0-9]*(?:[-_.]|$))", identity, re.I):
        raise ValueError(f"Forbidden Qt runtime dependency: {identity}")


def wheel_metadata(path):
    with zipfile.ZipFile(path) as archive:
        metadata = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError(f"Expected one wheel metadata entry: {path}")
        info = Parser().parsestr(archive.read(metadata[0]).decode())
        reject_qt_runtime(info["Name"])
        notices = [n for n in archive.namelist() if re.search(r"license|notice|copying", n, re.I)]
        return {"name": info["Name"], "version": info["Version"],
                "requires": info.get_all("Requires-Dist", []), "notices": notices}


def inspect_product_wheel(path):
    """Reject a wheel that could accidentally omit native code or design assets."""
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for name in names:
            for part in name.split("/"):
                reject_qt_runtime(part)
        native = [n for n in names if n.startswith("_pvd_native.") and n.endswith(".so")]
        if len(native) != 1 or "none-any" in path.name:
            raise ValueError("Product wheel must contain exactly one platform-tagged native extension")
        required = {"ovd_ui/runtime-build.json", "ovd_ui/notices/NOTICES.txt",
                    "ovd_ui/assets/colors.json",
                    "ovd_ui/notices/CHANGELOG.md", "ovd_ui/assets/fonts/manifest.json",
                    "ovd_ui/assets/fonts/Geist-OFL.txt", "ovd_ui/assets/fonts/DMMono-OFL.txt"}
        missing = required - names
        if missing:
            raise ValueError(f"Product wheel omits required resources: {sorted(missing)}")
        fonts = json.loads(archive.read("ovd_ui/assets/fonts/manifest.json"))["fonts"]
        for name, spec in fonts.items():
            actual = hashlib.sha256(archive.read("ovd_ui/assets/fonts/" + name)).hexdigest()
            if actual != spec["sha256"]:
                raise ValueError(f"Packaged font differs from approved source: {name}")
        for name in ("first", "previous", "reverse", "play", "pause", "next", "last",
                     "theme", "live", "live_paused", "dots", "close",
                     "visible", "hidden", "opaque", "transparent"):
            for suffix in ("svg", "png"):
                if f"ovd_ui/assets/icons/{name}.{suffix}" not in names:
                    raise ValueError(f"Missing packaged control icon: {name}.{suffix}")
        if any(n.startswith("tests/") or n.endswith("/extract_fonts.py") for n in names):
            raise ValueError("The runtime wheel must not include tests or asset generation tools")
        return native[0]


def write_lock(wheelhouse, target):
    records = []
    seen = set()
    for wheel in sorted(Path(wheelhouse).glob("*.whl")):
        info = wheel_metadata(wheel)
        name = re.sub(r"[-_.]+", "-", info["name"]).lower()
        if name in seen:
            raise ValueError(f"Multiple wheels for one dependency: {name}")
        seen.add(name)
        records.append({**info, "file": wheel.name, "sha256": sha256(wheel)})
    Path(target).write_text("# Exact local wheel bytes; install with --require-hashes.\n" + "".join(
        f"{r['name']}=={r['version']} --hash=sha256:{r['sha256']}\n" for r in records))
    return records


def deterministic_archive(source, destination, epoch):
    with Path(destination).open("xb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in [source, *sorted(source.rglob("*"))]:
                    if path.is_symlink():
                        raise ValueError(f"Bundle contains a symlink: {path}")
                    info = archive.gettarinfo(str(path), arcname=str(path.relative_to(source.parent)))
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = epoch
                    with path.open("rb") if path.is_file() else open(os.devnull, "rb") as data:
                        archive.addfile(info, data if path.is_file() else None)


def main():
    raise SystemExit(
        "Superseded by ci/package_ovui_portable.py; the retired wheelhouse "
        "used the stock Pillow wheel with OpenJPEG. Prepare the application "
        "venv with ci/install_ovui_pillow.py, then run: "
        "python -B ci/package_ovui_portable.py --native /absolute/path/to/_pvd_native<ABI-suffix> "
        "--physics-root /pinned/physics --ovui-root /pinned/ovui --output /new/package-directory"
    )


if __name__ == "__main__":
    main()
