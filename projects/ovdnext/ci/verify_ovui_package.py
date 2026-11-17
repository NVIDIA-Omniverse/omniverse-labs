# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify a downloaded portable package using only its extracted runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_ovui_portable import (inspect_bundle, probe_environment, runtime_security_command, sha256,
                                   validate_pillow_payload, validate_pillow_static_linkage,
                                   validate_runtime_inventory, validate_security_runtime)
from release_ovui import verified_assets

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_FIELDS = {"archive", "archive_sha256", "archive_bytes", "expanded_bytes", "unique_payload_bytes"}
FIXTURES = ("multi_geom.ovd", "cloth_physx34.pxd2", "ue4_shooter.pxd2")


def archive_inventory(archive, platform):
    """Validate ZIP names, types, executable modes and exact SHA256SUMS coverage."""
    prefix = f"OvdNext-{platform}-x86_64/"
    with zipfile.ZipFile(archive) as source:
        entries, folded = {}, set()
        for item in source.infolist():
            name = item.filename.removeprefix(prefix)
            parts = name.split("/")
            if (item.orig_filename != item.filename or not item.filename.startswith(prefix) or any(part in {"", ".", ".."} for part in parts)
                    or any(re.search(r'[\\\x00-\x1f]', part) for part in parts)
                    or platform == "windows" and any(re.search(r'[<>:"|?*]', part) or part.endswith((".", " "))
                           or re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part, re.I)
                           for part in parts)):
                raise ValueError(f"Unsafe ZIP path: {item.filename}")
            if name.casefold() in folded:
                raise ValueError(f"Duplicate ZIP path: {name}")
            if stat.S_IFMT(item.external_attr >> 16) not in (0, stat.S_IFREG) or item.is_dir():
                raise ValueError(f"ZIP entry is not a regular file: {name}")
            entries[name] = item
            folded.add(name.casefold())
        if any(parent.as_posix().casefold() in folded for name in entries for parent in PurePosixPath(name).parents):
            raise ValueError("Conflicting ZIP file and directory paths")
        if platform == "linux":
            for name in ("OvdNext.sh", "python/bin/python3.12"):
                if name not in entries or not entries[name].external_attr >> 16 & 0o111:
                    raise ValueError(f"Linux runtime is not executable: {name}")
        if "SHA256SUMS" not in entries:
            raise ValueError("Missing SHA256SUMS inventory")
        inventory = {}
        for line in source.read(entries["SHA256SUMS"]).decode("utf-8").splitlines():
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
            if not match or match[2] in inventory:
                raise ValueError("Malformed or duplicate SHA256SUMS inventory")
            inventory[match[2]] = match[1]
        if set(inventory) != set(entries) - {"SHA256SUMS"}:
            raise ValueError("SHA256SUMS does not exactly cover the ZIP inventory")
        for name, item in entries.items():
            with source.open(item) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if name != "SHA256SUMS" and digest != inventory[name]:
                raise ValueError(f"ZIP payload checksum mismatch: {name}")
            inventory[name] = digest
    return inventory


def verify_payload(bundle, inventory):
    files = [path for path in bundle.rglob("*") if path.is_file() or path.is_symlink()]
    if any(path.is_symlink() for path in files) or {p.relative_to(bundle).as_posix() for p in files} != set(inventory):
        raise ValueError("Extracted payload inventory changed")
    for path in files:
        if sha256(path) != inventory[path.relative_to(bundle).as_posix()]:
            raise ValueError(f"Extracted payload checksum mismatch: {path}")


def validate_probe(probe, report, fixture):
    expected = {key: value for key, value in report.items() if key not in ARCHIVE_FIELDS}
    if (probe.get("version") != report["version"] or probe.get("isolated") is not True
            or probe.get("native_provenance") != expected or report.get("source_dirty") is not False
            or report.get("allow_dirty") is not False or any(report.get("dependency_dirty", {}).values())):
        raise ValueError(f"Runtime provenance mismatch: {fixture}")
    capture = probe.get("capture", {})
    if (type(capture.get("scenes")) is not int or capture["scenes"] < 1
            or capture.get("presentation") != "Presentation" or type(capture.get("instances")) is not int
            or capture["instances"] < (1 if fixture.endswith(".ovd") else 0)):
        raise ValueError(f"Invalid capture presentation: {fixture}")


def run_logged(command, log, cwd, env, *, timeout=120):
    command = [str(value) for value in command]
    # Files avoid waiting for output pipes inherited by a launcher's descendants.
    with tempfile.TemporaryFile(dir=log.parent) as stdout, tempfile.TemporaryFile(dir=log.parent) as stderr:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=os.name != "nt")
        failure = None
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            failure = error
            try:
                if os.name == "nt":
                    taskkill = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/taskkill.exe"
                    subprocess.run([str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                                   stdout=stderr, stderr=subprocess.STDOUT, timeout=5, check=True)
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            finally:
                process.kill()
                process.wait(timeout=5)
        finally:
            stdout.seek(0)
            stderr.seek(0)
            output, errors = (stream.read().decode("utf-8", errors="replace") for stream in (stdout, stderr))
            log.write_text((str(failure) + "\n" if failure else "") + output + "\n" + errors, encoding="utf-8")
        if failure:
            raise failure
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, output, errors)
        return output


def verify(args):
    archive, _, report_path = verified_assets(args.packages, args.version, args.source_commit, platforms=(args.platform,))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    inventory = archive_inventory(archive, args.platform)
    relocated = args.work_root / "relocated package é測試"
    relocated.mkdir()
    windows = args.platform == "windows"
    if windows:
        with zipfile.ZipFile(archive) as source:
            source.extractall(relocated)
    else:
        run_logged(["unzip", "-q", archive, "-d", relocated], args.work_root / "extract.log", args.work_root, os.environ.copy())
    bundle = relocated / f"OvdNext-{args.platform}-x86_64"
    verify_payload(bundle, inventory)
    inspect_bundle(bundle)
    validate_runtime_inventory(bundle, report["distributions"])
    native = next(path for path in (bundle / "app").glob("_pvd_native.*") if path.suffix in {".so", ".pyd"})
    if sha256(native) != report["native_sha256"]:
        raise ValueError("Native module checksum differs from package provenance")
    from install_ovui_python import read_runtime_provenance

    if read_runtime_provenance(bundle / "python", windows) != report.get("python_input_provenance"):
        raise ValueError("Python input provenance differs from package provenance")
    from package_ovui_portable import python_dependency_notice_records

    if python_dependency_notice_records(bundle / "python/licenses", args.platform) != report.get("python_dependency_notices"):
        raise ValueError("Python dependency notice provenance differs from package provenance")
    if windows:
        from install_ovui_expat import read_build_provenance

        expat_build_provenance = read_build_provenance(bundle / "python")
        if expat_build_provenance != report.get("expat_build_provenance"):
            raise ValueError("Expat build provenance differs from package provenance")
    env = probe_environment(bundle, windows)
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "_OLD_VIRTUAL_PATH",
                "GALLIUM_DRIVER", "LIBGL_ALWAYS_SOFTWARE"):
        env.pop(key, None)
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "TMP", "TEMP", "TMPDIR"):
        directory = args.work_root / key.lower()
        directory.mkdir()
        env[key] = str(directory)
    captures = args.work_root / "captures é測試"
    captures.mkdir()
    launcher = bundle / ("OvdNext.cmd" if windows else "OvdNext.sh")
    python = bundle / ("python/python.exe" if windows else "python/bin/python3.12")
    runtime_security = json.loads(run_logged(runtime_security_command(python, bundle / "app"),
                                            args.work_root / "runtime-security.log", args.work_root, env))
    validate_security_runtime(runtime_security)
    validate_pillow_payload(bundle, runtime_security["pillow"])
    (args.work_root / "pillow-native-dependencies.txt").write_text(validate_pillow_static_linkage(bundle / "app", windows) + "\n")
    if runtime_security != {"python": report.get("python"), "expat": report.get("expat"), "pillow": report.get("pillow")}:
        raise ValueError("Runtime security evidence differs from package provenance")
    (args.work_root / "runtime-security.json").write_text(json.dumps(runtime_security, indent=2) + "\n", encoding="utf-8")
    results = {}
    for fixture in FIXTURES:
        capture = captures / fixture
        shutil.copyfile(ROOT / "test/data" / fixture, capture)
        command = [launcher, "--check", capture]
        output = run_logged(command, args.work_root / (fixture + ".log"), args.work_root, env)
        result = json.loads(output)
        (args.work_root / (fixture + ".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
        validate_probe(result, report, fixture)
        results[fixture] = result["capture"]
    screenshot = args.work_root / "gui.png"
    command = [python, "-I", "-B", ROOT / "ci/smoke_ovui_gui.py", "--bundle", bundle, "--screenshot", screenshot]
    if args.opengl_library:
        command += ["--opengl-library", args.opengl_library.resolve(strict=True)]
    gui_env = env.copy()
    for key in ("GALLIUM_DRIVER", "LIBGL_ALWAYS_SOFTWARE"):
        if key in os.environ:
            gui_env[key] = os.environ[key]
    run_logged(command, args.work_root / "gui.log", args.work_root, gui_env)
    if not screenshot.is_file() or not screenshot.stat().st_size:
        raise ValueError("GUI smoke produced no screenshot")
    verify_payload(bundle, inventory)
    return {"archive_sha256": report["archive_sha256"], "runtime_security": runtime_security,
            "captures": results, "screenshot": str(screenshot)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", required=True, type=Path)
    parser.add_argument("--platform", required=True, choices=("windows", "linux"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--opengl-library", type=Path)
    args = parser.parse_args(argv)
    args.work_root = args.work_root.resolve()
    try:
        args.work_root.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"Verification needs a new work directory: {error}", file=sys.stderr)
        return 1
    result = dict(platform=args.platform, version=args.version, source_commit=args.source_commit, status="failed")
    try:
        result.update(verify(args), status="passed")
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        print(result["error"], file=sys.stderr)
    (args.work_root / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
