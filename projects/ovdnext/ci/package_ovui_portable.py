#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage the pinned installed runtime into a relocatable OvdNext application."""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.12.16"
PHYSICS_PIN = "a12f79ce585ffb79e5ffa4848cb45efc3de2f632"
OVUI_PIN = "a81db103a4d99ff2d75368a444e167889536ff42"
EXPAT_LICENSE_SHA256 = "31b15de82aa19a845156169a17a5488bf597e561b2c318d159ed583139b25e87"
HACL_LICENSE_SHA256 = "3a2394d9022fc168845231b08e98a6eade120257465968655cc4d79b7e9828a9"
PYTHON_NOTICE_RESOURCES = ROOT / "ci/ovui_package/python-runtime-notices"
HOST_COMPILER_RUNTIME = re.compile(r"lib(?:stdc\+\+|gcc_s)\.so(?:\..+)?")
NOTICES = ("NOTICES.txt", "CHANGELOG.md", "OvdNext-LICENSE.txt", "OvdNext-NOTICE.txt", "OVUI-LICENSE.txt", "OVUI-THIRD_PARTY_NOTICES.md",
           "Physics-LICENSE.md", "PVD-source-notice.txt", "PxShared-source-notice.txt",
           "pybind11-LICENSE.txt", "Python-LICENSE.txt", "Expat-LICENSE.txt", "HACL-LICENSE.txt")


def run(command, cwd=ROOT, env=None):
    # A linked worktree made by Windows git retains a drive path in its .git
    # pointer. Scope the WSL translation to this command, never repository config.
    if command[0] == "git" and sys.platform == "linux":
        repository = Path(command[command.index("-C") + 1]) if "-C" in command else Path(cwd)
        pointer = repository / ".git"
        gitdir = pointer
        if pointer.is_file():
            match = re.fullmatch(r"gitdir: ([A-Za-z]):[/\\](.+)", pointer.read_text().strip())
            if match:
                env = {**(os.environ if env is None else env),
                       "GIT_DIR": "/mnt/" + match[1].lower() + "/" + match[2].replace("\\", "/"),
                       "GIT_WORK_TREE": str(repository)}
                gitdir = Path(env["GIT_DIR"])
        alternates = gitdir / "objects/info/alternates"
        if alternates.is_file():
            translated = []
            for line in alternates.read_text().splitlines():
                match = re.fullmatch(r"([A-Za-z]):[/\\](.+)", line)
                if match:
                    translated.append("/mnt/" + match[1].lower() + "/" + match[2].replace("\\", "/"))
            if translated:
                env = {**(os.environ if env is None else env), "GIT_ALTERNATE_OBJECT_DIRECTORIES": ":".join(translated)}
    return subprocess.run([str(x) for x in command], cwd=cwd, env=env, check=True,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def copy_file(source, target):
    if not source.is_file():
        raise ValueError(f"Missing runtime input: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise ValueError(f"Conflicting runtime files: {target}")
    shutil.copyfile(source, target)
    target.chmod(source.stat().st_mode & 0o777)


def new_output(path, sources):
    if path.exists() or path.is_symlink():
        raise ValueError("Output must be a new directory; existing output is preserved")
    path = path.resolve()
    if any(path.is_relative_to(source.resolve()) for source in sources):
        raise ValueError("Output must be outside runtime source directories")
    path.mkdir(parents=True)
    return path


def validate_version(root):
    first = next((line for line in (root / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
                  if line.startswith("## ")), "")
    heading = re.fullmatch(r"## v((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)) \(\d{4}-\d{2}-\d{2}\)", first)
    declared = re.search(r'^__version__ = "([^"]+)"$',
                         (root / "python/ovd_ui/__init__.py").read_text(), re.M)
    if not heading or not declared or heading[1] != declared[1]:
        raise ValueError("Application version must match canonical CHANGELOG.md version")
    return heading[1]


def copy_distribution(dist, expected, target):
    reject_qt_runtime(dist.metadata["Name"])
    if dist.version != expected:
        raise ValueError(f"Wrong package version for {dist.metadata['Name']}: {dist.version}, expected {expected}")
    base = Path(dist.locate_file("")).resolve()
    if not dist.files:
        raise ValueError(f"No distribution file inventory: {dist.metadata['Name']}")
    # Installer-created command wrappers are not importable runtime payloads.
    commands = {entry.name for entry in dist.entry_points if entry.group in {"console_scripts", "gui_scripts"}}
    wrappers = {name + suffix for name in commands for suffix in ("", ".exe", "-script.py")}
    for entry in dist.files:
        source = Path(dist.locate_file(entry)).resolve()
        if not source.is_relative_to(base):
            if source.parent in {base.parent.parent / "Scripts", base.parent.parent.parent / "bin"} and source.name in wrappers:
                continue
            raise ValueError(f"Distribution file escapes install tree: {entry}")
        if not runtime_member(Path(str(entry))):
            continue
        copy_file(source, target / source.relative_to(base))
    return {"name": dist.metadata["Name"], "version": dist.version}


def runtime_member(path):
    # OvUI's companion agent/inspector CLI is not part of the viewer runtime.
    # It decodes Base64 responses from configurable HTTP endpoints permissively.
    if path.parts[0] == "ovui_agent_skills":
        return False
    if path.parts[0].endswith(".dist-info"):
        return True
    if any(part in {"tests", "test", "__pycache__"} for part in path.parts):
        return False
    if path.parts[0] == "numpy" and "include" in path.parts:
        return False
    if re.match(r"^(?:_test|_ctypes_test|_xxtestfuzz|_tkinter\.|_dbm\.|(?:tcl|tk)\d.*\.dll)", path.name):
        return False
    if re.search(r"_tests\.(?:.*\.)?(?:so|pyd|pyi)$", path.name):
        return False
    return path.suffix not in {".pyc", ".pdb", ".lib", ".a"}


def probe_environment(bundle, windows, source=None):
    env = {key: value for key, value in (os.environ if source is None else source).items()
           if not key.upper().startswith(("PYTHON", "LD_")) and key not in
           {"CUDA_PATH", "CUDAToolkit_ROOT", "CUDA_HOME", "GLCONTEXT_WIN_LIBGL", "GALLIUM_DRIVER"}}
    if windows:
        system = env.get("SystemRoot", env.get("SYSTEMROOT", r"C:\Windows"))
        env["PATH"] = system + r"\System32;" + system
    else:
        env["PATH"] = "/usr/bin:/bin"
        env["LD_LIBRARY_PATH"] = str(bundle / "lib") + ":" + str(bundle / "python/lib")
    env["OVUIINSPECT_ENABLED"] = "0"
    return env


def validate_security_runtime(evidence):
    """Validate observed interpreter/codec behavior, never only reported pins."""
    from install_ovui_pillow import validate_pillow_evidence

    if evidence.get("python") != PYTHON_VERSION:
        raise ValueError(f"Packaged Python must be {PYTHON_VERSION}: {evidence.get('python')}")
    if evidence.get("expat") != "expat_2.9.0":
        raise ValueError(f"Packaged Expat must be expat_2.9.0: {evidence.get('expat')}")
    validate_pillow_evidence(evidence.get("pillow", {}))


def runtime_security_command(python, app):
    from install_ovui_pillow import PILLOW_PROBE

    return [python, "-I", "-B", "-c", PILLOW_PROBE, app]


def validate_pillow_payload(bundle, observed):
    """Require source/license evidence in the distributed bytes as well as JSON output."""
    from install_ovui_pillow import PILLOW_SOURCE, validate_pillow_evidence

    info = bundle / "app" / f"pillow-{PILLOW_SOURCE['version']}.dist-info"
    try:
        provenance = json.loads((info / "ovdnext-build.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Pillow verified source provenance is required") from error
    license_file = info / "licenses/ZLIB-NG.txt"
    try:
        license_digest = sha256(license_file)
        declared = "License-File: ZLIB-NG.txt" in (info / "METADATA").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError("Pillow pinned zlib-ng license is required") from error
    actual = {**observed, "build_provenance": provenance, "vendor_license_sha256": license_digest,
              "vendor_license_declared": declared}
    validate_pillow_evidence(actual)


def validate_ovui_build(dist):
    info = Path(dist.locate_file("omni/ui/_build_info.py"))
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(info.read_text()).body
              if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    if values.get("__commit__") != OVUI_PIN or values.get("__version__") != dist.version:
        raise ValueError("Installed OVUI build does not match its declared version/source pin")


def copy_cuda(dist, bundle, windows):
    if dist.version != "12.9.79":
        raise ValueError(f"Wrong CUDA runtime version: {dist.version}")
    base = Path(dist.locate_file("")).resolve()
    member = "nvidia/cuda_runtime/" + ("bin/cudart64_12.dll" if windows else "lib/libcudart.so.12")
    if not dist.files:
        raise ValueError("Missing CUDA runtime file inventory")
    for entry in dist.files:
        source = Path(dist.locate_file(entry)).resolve()
        if not source.is_relative_to(base):
            raise ValueError(f"CUDA distribution file escapes install tree: {entry}")
        if str(entry).replace("\\", "/") == member:
            copy_file(source, bundle / "lib" / source.name)
        elif entry.parts[0].endswith(".dist-info"):
            copy_file(source, bundle / "lib/notices" / Path(str(entry)))
    if not (bundle / "lib" / Path(member).name).is_file():
        raise ValueError(f"Missing CUDA runtime: {member}")
    return {"name": dist.metadata["Name"], "version": dist.version}


def reject_qt_runtime(identity):
    """Match package/library identities, never optional source adapter text."""
    name = str(identity).replace("\\", "/").rsplit("/", 1)[-1]
    if re.match(r"^(?:lib)?(?:qt[0-9]*[a-z]|qwt(?:[-_.0-9]|$)|pyside[0-9]*(?:[-_.]|$)|pyqt[0-9]*(?:[-_.]|$))", name, re.I):
        raise ValueError(f"Forbidden Qt runtime dependency: {identity}")


def native_dependencies(native, windows, *, require_static_zlib=False):
    if windows:
        dumpbin = shutil.which("dumpbin")
        if not dumpbin:
            roots = [Path(os.environ.get(name, "")) / "Microsoft Visual Studio" for name in
                     ("ProgramFiles", "ProgramFiles(x86)")]
            candidates = [p for root in roots if root.is_dir()
                          for p in root.glob("*/**/VC/Tools/MSVC/*/bin/Hostx64/x64/dumpbin.exe")]
            if not candidates:
                raise ValueError("dumpbin is required for native dependency verification")
            dumpbin = str(sorted(candidates)[-1])
        listing = run([dumpbin, "/dependents", native])
        identities = re.findall(r"(?im)^\s*([^\s]+\.dll)\s*$", listing)
        if not identities:
            raise ValueError("dumpbin returned no DLL dependencies")
    else:
        listing = run(["readelf", "-d", native])
        identities = re.findall(r"Shared library: \[([^\]]+)\]", listing)
        if not identities:
            raise ValueError("readelf returned no NEEDED dependencies")
    for identity in identities:
        reject_qt_runtime(identity)
        if require_static_zlib and re.match(r"^(?:lib)?(?:z(?:lib)?(?:-ng)?|zlibwapi)\d*(?:[-_.]|$)", identity, re.I):
            raise ValueError(f"Pillow must use static pinned zlib-ng, found dynamic dependency: {identity}")
    if not windows:
        resolved = run(["ldd", native])
        if "not found" in resolved:
            raise ValueError(f"Unresolved native dependency: {resolved}")
        listing += "\n" + resolved
    return listing


def validate_pillow_static_linkage(app, windows):
    cores = [path for path in (app / "PIL").glob("_imaging.*") if path.suffix in {".pyd", ".so"}]
    if len(cores) != 1:
        raise ValueError("Missing or ambiguous Pillow native core")
    return native_dependencies(cores[0], windows, require_static_zlib=True)


def validate_runtime_inventory(bundle, distributions):
    if (bundle / "app/ovui_agent_skills").exists():
        raise ValueError("OvUI agent tooling must not be bundled with the viewer")
    for dist in distributions:
        reject_qt_runtime(dist["name"])
    for path in bundle.rglob("*"):
        if path.is_file() and (path.suffix.lower() in {".dll", ".pyd"} or ".so" in path.name):
            reject_qt_runtime(path.name)
            if re.match(r"^(?:lib)?openjp(?:2|eg)(?:[-_.]|$)", path.name, re.I):
                raise ValueError(f"Forbidden OpenJPEG runtime dependency: {path}")
            if HOST_COMPILER_RUNTIME.fullmatch(path.name):
                raise ValueError(f"Linux host compiler runtime must not be bundled: {path}")
    for info in bundle.rglob("*.dist-info/METADATA"):
        name = next((line[6:] for line in info.read_text().splitlines() if line.startswith("Name: ")), "")
        if not name:
            raise ValueError(f"Missing distribution identity: {info}")
        reject_qt_runtime(name)


def inspect_bundle(bundle):
    if not (bundle / "README.txt").is_file():
        raise ValueError("Bundle omits required README.txt")
    app = bundle / "app"
    native = [p for p in app.glob("_pvd_native.*") if p.suffix in {".pyd", ".so"}]
    if len(native) != 1:
        raise ValueError("Missing or ambiguous native module")
    required = ["assets/colors.json", "assets/fonts/manifest.json", "assets/fonts/Geist-OFL.txt",
                "assets/fonts/DMMono-OFL.txt", *["notices/" + name for name in NOTICES]]
    for relative in required:
        if not (app / "ovd_ui" / relative).is_file():
            raise ValueError(f"Bundle omits required resource: {relative}")
    if sha256(app / "ovd_ui/notices/Expat-LICENSE.txt") != EXPAT_LICENSE_SHA256:
        raise ValueError("Expat license notice hash mismatch")
    if sha256(app / "ovd_ui/notices/HACL-LICENSE.txt") != HACL_LICENSE_SHA256:
        raise ValueError("HACL license notice hash mismatch")
    fonts = json.loads((app / "ovd_ui/assets/fonts/manifest.json").read_text())["fonts"]
    for name, spec in fonts.items():
        if sha256(app / "ovd_ui/assets/fonts" / name) != spec["sha256"]:
            raise ValueError(f"Font hash mismatch: {name}")
    for name in ("first", "previous", "reverse", "play", "pause", "next", "last", "theme", "live",
                 "live_paused", "dots", "close", "visible", "hidden", "opaque", "transparent"):
        for suffix in ("png", "svg"):
            if not (app / "ovd_ui/assets/icons" / f"{name}.{suffix}").is_file():
                raise ValueError(f"Missing icon: {name}.{suffix}")


def file_mode(path):
    return 0o755 if path.suffix == ".sh" or path.stat().st_mode & 0o111 else 0o644


def make_archive(bundle, destination, epoch):
    files = sorted(p for p in bundle.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in bundle.rglob("*")):
        raise ValueError("Bundle contains a symlink")
    if destination.suffix == ".zip":
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                info = zipfile.ZipInfo(path.relative_to(bundle.parent).as_posix(),
                                       time.gmtime(max(epoch, 315532800))[:6])
                info.create_system = 3
                info.external_attr = (0o100000 | file_mode(path)) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes(), compresslevel=9)
    else:
        with destination.open("xb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                inodes = {}
                for path in files:
                    info = tarfile.TarInfo(path.relative_to(bundle.parent).as_posix())
                    info.size, info.mtime, info.mode = path.stat().st_size, epoch, file_mode(path)
                    stat = path.stat()
                    inode = (stat.st_dev, stat.st_ino)
                    if stat.st_nlink > 1 and inode in inodes:
                        info.type, info.linkname, info.size = tarfile.LNKTYPE, inodes[inode], 0
                        archive.addfile(info)
                        continue
                    inodes[inode] = info.name
                    with path.open("rb") as content:
                        archive.addfile(info, content)


def finish_report(bundle, archive, provenance, budget):
    if archive.stat().st_size > budget * 1024 * 1024:
        raise ValueError(f"Archive exceeds {budget:g} MiB budget: {archive.stat().st_size} bytes")
    digest = sha256(archive)
    archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n")
    expanded = sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())
    unique = (expanded if archive.suffix == ".zip" else
              sum({(p.stat().st_dev, p.stat().st_ino): p.stat().st_size
                   for p in bundle.rglob("*") if p.is_file()}.values()))
    report = {**provenance, "archive": archive.name, "archive_sha256": digest,
              "archive_bytes": archive.stat().st_size,
              "expanded_bytes": expanded, "unique_payload_bytes": unique}
    write_json(archive.parent / "package-report.json", report)
    write_json(archive.with_name(archive.name + ".package-report.json"), report)
    return report


def python_dependency_notice_records(directory, platform):
    """Bind restored notices to the platform input, sources and delivered bytes."""
    from install_ovui_python import load_manifest

    entry = load_manifest()["platforms"][platform]
    spec = entry.get("dependency_notices")
    pins = spec.get("files_sha256") if isinstance(spec, dict) else None
    if (not isinstance(pins, dict) or not pins
            or any(not isinstance(name, str) or not re.fullmatch(r"LICENSE\.[A-Za-z0-9_.-]+\.txt", name)
                   or not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for name, value in pins.items())):
        raise ValueError("Python dependency notices require complete filename/SHA-256 pins")
    directory = Path(directory)
    measured = {}
    for name in pins:
        notice = directory / name
        if notice.is_symlink() or not notice.is_file():
            raise ValueError(f"Python dependency notice is missing or not a regular file: {name}")
        measured[name] = sha256(notice)
    if measured != pins:
        raise ValueError("Python dependency notice hash mismatch")
    return {"input_archive_sha256": entry["sha256"], **spec, "files_sha256": measured}


def copy_python(prefix, bundle, windows):
    platform = "windows" if windows else "linux"
    resources = PYTHON_NOTICE_RESOURCES / platform
    notices = python_dependency_notice_records(resources, platform)
    target = bundle / "python"
    target.mkdir()
    if windows:
        for name in ("python.exe", "python3.dll", "python312.dll", "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt"):
            copy_file(prefix / name, target / name)
        stdlib = prefix / "Lib"
        destination = target / "Lib"
        # This DLL serves excluded Tcl only; pip distribution DLLs are unchanged.
        copy_tree(prefix / "DLLs", target / "DLLs", exclude={"zlib1.dll"})
    else:
        copy_file(prefix / "bin/python3.12", target / "bin/python3.12")
        stdlib = prefix / "lib/python3.12"
        destination = target / "lib/python3.12"
        for library in sorted((prefix / "lib").glob("libpython3.12.so*")):
            copy_file(library, target / "lib" / library.name)
    # Restore the audited delivered-consumer notices omitted by install-only
    # conversion without changing the pinned interpreter input.
    for name in notices["files_sha256"]:
        copy_file(resources / name, target / "licenses" / name)
    python_dependency_notice_records(target / "licenses", platform)
    copy_tree(stdlib, destination)
    for name in ("runtime-input-provenance.json", *(["ovdnext-expat-build.json", "LICENSE.expat.txt", "LICENSE.cpython-xml.txt"] if windows else [])):
        if (prefix / name).is_file():
            copy_file(prefix / name, target / name)
    return target / ("python.exe" if windows else "bin/python3.12")


def copy_tree(source, target, *, exclude=()):
    if not source.is_dir():
        raise ValueError(f"Missing runtime directory: {source}")
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in {"site-packages", "__pycache__", "test", "tests", "ensurepip", "idlelib", "tkinter"} for part in relative.parts):
            continue
        if path.is_file() and path.name not in exclude and runtime_member(relative) and path.name != "extract_fonts.py":
            if not path.resolve().is_relative_to(source.resolve()):
                raise ValueError(f"Runtime file escapes source: {path}")
            copy_file(path, target / relative)


def linux_library_notice(source, bundle):
    candidates = [source, source.resolve()]
    candidates += [Path(str(path).replace("/usr/lib/", "/lib/", 1)) for path in candidates if str(path).startswith("/usr/lib/")]
    package = None
    for candidate in dict.fromkeys(candidates):
        try:
            owners = run(["dpkg-query", "-S", candidate])
        except subprocess.CalledProcessError:
            continue
        for line in owners.splitlines():
            if ": " in line and line.rsplit(": ", 1)[1] == str(candidate):
                package = line.rsplit(": ", 1)[0]
                break
        if package:
            break
    if not package:
        raise ValueError(f"No installed package/notice coverage for Linux library: {source}")
    name, version = run(["dpkg-query", "-W", "-f=${binary:Package}\t${Version}", package]).split("\t")
    copyright_file = Path("/usr/share/doc") / name.split(":", 1)[0] / "copyright"
    if not copyright_file.is_file():
        raise ValueError(f"Missing copyright notice for Linux package: {name}")
    target = bundle / "lib/notices/system" / name / "copyright"
    copy_file(copyright_file, target)
    licenses = []
    for filename in sorted(set(re.findall(r"/usr/share/common-licenses/([A-Za-z0-9_.+-]+)", copyright_file.read_text()))):
        filename = filename.rstrip(".")
        license_file = Path("/usr/share/common-licenses") / filename
        destination = bundle / "lib/notices/common-licenses" / filename
        copy_file(license_file, destination)
        licenses.append({"file": destination.relative_to(bundle).as_posix(), "sha256": sha256(destination)})
    return {"provider": "dpkg", "package": name, "version": version,
            "notice": target.relative_to(bundle).as_posix(), "notice_sha256": sha256(target), "licenses": licenses}


def prepare_linux_elf(bundle, symbols):
    """Split our native symbols; preserve vendor ELF bytes and internal aliases."""
    payloads = {}
    for path in sorted(bundle.rglob("*")):
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                continue
        owned_native = path.parent == bundle / "app" and path.name.startswith("_pvd_native.") and path.suffix == ".so"
        digest = (sha256(path), owned_native)
        if digest in payloads:
            # Both paths are discovered inside our new bundle. Never link an input.
            path.unlink()
            os.link(payloads[digest], path)
            continue
        payloads[digest] = path
        if not owned_native:
            continue
        sections = run(["readelf", "-SW", path])
        if ".debug_" in sections or ".symtab" in sections:
            debug = symbols / (path.relative_to(bundle).as_posix() + ".debug")
            debug.parent.mkdir(parents=True, exist_ok=True)
            run(["objcopy", "--only-keep-debug", path, debug])
            run(["strip", "--strip-unneeded", path])
            run(["objcopy", "--add-gnu-debuglink=" + str(debug), path])


def stage_libraries(native, bundle, windows, msvc_runtime):
    lib = bundle / "lib"
    lib.mkdir()
    notices = {}
    copy_cuda(metadata.distribution("nvidia-cuda-runtime-cu12"), bundle, windows)
    if windows:
        if msvc_runtime is None or not (msvc_runtime / "msvcp140.dll").is_file():
            raise ValueError("Supply --msvc-runtime pointing to the x64 Microsoft.VC*.CRT redistributable directory")
        for path in sorted(msvc_runtime.glob("*.dll")):
            copy_file(path, lib / path.name)
    else:
        seeds = [native, bundle / "python/bin/python3.12",
                 *Path(sys.base_prefix).glob("lib/python3.12/lib-dynload/*.so"),
                 *[p for p in (bundle / "app").rglob("*.so*") if p.is_file()]]
        seeds = [p for p in seeds if runtime_member(Path(p.name))]
        # ldd gives the resolved closure. Keep the platform ABI and drivers on the host.
        excluded = re.compile(r"^(?:ld-linux|linux-vdso|lib(?:c|m|pthread|dl|rt|resolv|util|anl|cuda)\.so|lib(?:GL|EGL|GLX|OpenGL|vulkan|nvidia|drm))")
        env = {**os.environ, "LD_LIBRARY_PATH": str(lib) + ":" + str(bundle / "python/lib")}
        for seed in seeds:
            listing = run(["ldd", seed], env=env)
            if "not found" in listing:
                raise ValueError(f"Unresolved native dependency: {seed}\n{listing}")
            for line in listing.splitlines():
                match = re.match(r"\s*(\S+) => (/\S+) ", line)
                if match:
                    reject_qt_runtime(match[1])
                if match and not excluded.match(match[1]) and not HOST_COMPILER_RUNTIME.fullmatch(match[1]):
                    source = Path(match[2]).resolve()
                    if source.is_relative_to(bundle.resolve()):
                        continue
                    target = lib / match[1]
                    notices[target.relative_to(bundle).as_posix()] = linux_library_notice(source, bundle)
                    copy_file(source, target)
    return notices


def stage_notices(bundle, ovui, physics, prefix):
    notices = bundle / "app/ovd_ui/notices"
    notices.mkdir(exist_ok=True)
    for source, name in ((ROOT / "CHANGELOG.md", "CHANGELOG.md"),
                         (ROOT / "LICENSE", "OvdNext-LICENSE.txt"),
                         (ROOT / "NOTICE", "OvdNext-NOTICE.txt"),
                         (ROOT / "ci/ovui_package/Expat-LICENSE.txt", "Expat-LICENSE.txt"),
                         (ROOT / "ci/ovui_package/HACL-LICENSE.txt", "HACL-LICENSE.txt"),
                         (ovui / "LICENSE", "OVUI-LICENSE.txt"),
                         (ovui / "THIRD_PARTY_NOTICES.md", "OVUI-THIRD_PARTY_NOTICES.md")):
        copy_file(source, notices / name)
    python_license = next((p for p in (prefix / "LICENSE.txt", prefix / "lib/python3.12/LICENSE.txt") if p.is_file()), None)
    if python_license is None:
        raise ValueError("Missing Python license")
    copy_file(python_license, notices / "Python-LICENSE.txt")
    pybind = metadata.distribution("pybind11")
    license_file = next(f for f in pybind.files if str(f).endswith("licenses/LICENSE"))
    copy_file(Path(pybind.locate_file(license_file)), notices / "pybind11-LICENSE.txt")
    for source, name in (("src/PxShared/include/foundation/Px.h", "PxShared-source-notice.txt"),
                         ("src/PVDCore/PvdCommStreamMetaData.h", "PVD-source-notice.txt")):
        (notices / name).write_text((ROOT / source).read_text().split("#ifndef", 1)[0])
    (notices / "Physics-LICENSE.md").write_text(run(["git", "-C", physics, "show", "HEAD:physx/documentation/license/LICENSE.md"]) + "\n")
    (notices / "NOTICES.txt").write_text(
        "OvdNext portable runtime notices\n\n"
        "OvdNext software: Apache-2.0; see OvdNext-LICENSE.txt and OvdNext-NOTICE.txt.\n"
        "Dependency terms remain unchanged. NVIDIA PVD, PxShared and Physics notices,\n"
        "OVUI license and third-party notices, pybind11 and Python\n"
        "licenses accompany this application. Python distributions retain metadata\n"
        "and license files in app/. Fonts retain their SIL Open Font Licenses.\n\n"
        "Expat (XML parser in the Python runtime): see Expat-LICENSE.txt.\n\n"
        "HACL* and KaRaMeL (Python hash implementations): see HACL-LICENSE.txt.\n\n"
        "Python dependency notices: see python/licenses/ in the bundle.\n\n"
        "See native-libraries.json for the shipped library inventory.\n"
        "Windows includes Microsoft Visual C++ redistributable CRT under Microsoft\n"
        "terms: https://visualstudio.microsoft.com/license-terms/ . Both platforms\n"
        "include NVIDIA CUDA runtime 12.9.79 with license and metadata in lib/notices/.\n"
        "Graphics drivers remain host-provided.\n")
    copy_file(ROOT / "CHANGELOG.md", bundle / "CHANGELOG.md")


def runtime_input_records(prefix, windows):
    """Revalidate runtime bytes before creating any package output."""
    from install_ovui_python import read_runtime_provenance

    records = {"python_input_provenance": read_runtime_provenance(prefix, windows)}
    if windows:
        from install_ovui_expat import read_build_provenance

        records["expat_build_provenance"] = read_build_provenance(prefix)
    platform = "windows" if windows else "linux"
    records["python_dependency_notices"] = python_dependency_notice_records(
        PYTHON_NOTICE_RESOURCES / platform, platform)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("output", "native", "physics-root", "ovui-root"):
        parser.add_argument("--" + option, required=True, type=Path)
    parser.add_argument("--msvc-runtime", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--max-archive-mib", type=float, default=160)
    args = parser.parse_args(argv)
    windows = sys.platform == "win32"
    if sys.platform not in {"win32", "linux"} or platform.machine().lower() not in {"amd64", "x86_64"} or sys.version_info[:2] != (3, 12):
        parser.error("Requires Windows/Linux x86_64 CPython 3.12")
    if platform.python_version() != PYTHON_VERSION:
        parser.error(f"Packaged Python must be {PYTHON_VERSION}: {platform.python_version()}")
    try:
        version = validate_version(ROOT)
        native = args.native.resolve()
        if not native.is_file() or native.name != "_pvd_native" + sysconfig.get_config_var("EXT_SUFFIX"):
            raise ValueError("Missing native module or CPython ABI mismatch")
        dirty = run(["git", "status", "--porcelain"])
        dependency_dirty = {}
        for name, source, pin in (("physics", args.physics_root, PHYSICS_PIN), ("ovui", args.ovui_root, OVUI_PIN)):
            if run(["git", "-C", source, "rev-parse", "HEAD"]) != pin:
                raise ValueError(f"Wrong {name} source pin")
            dependency_dirty[name] = run(["git", "-C", source, "status", "--porcelain"])
        if (dirty or any(dependency_dirty.values())) and not args.allow_dirty:
            raise ValueError("Runtime sources are dirty; --allow-dirty is required for a development artifact")
        pins = []
        for line in (ROOT / "python/requirements-runtime.txt").read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                match = re.fullmatch(r"([a-zA-Z0-9_.-]+)==([0-9.]+)", line)
                if not match:
                    raise ValueError(f"Runtime dependency must use an exact pin: {line}")
                pins.append(match.groups())
        if len(pins) != 8 or len({name for name, _ in pins}) != 8:
            raise ValueError("Expected exactly eight pinned runtime distributions")
        for name, expected in pins:
            if metadata.version(name) != expected:
                raise ValueError(f"Wrong package version: {name}")
        validate_ovui_build(metadata.distribution("ovui"))
        installed_app = Path(metadata.distribution("Pillow").locate_file(""))
        runtime_security = json.loads(run(runtime_security_command(sys.executable, installed_app)))
        validate_security_runtime(runtime_security)
        prefix = Path(sys.base_prefix).resolve()
        input_records = runtime_input_records(prefix, windows)
        dependency_listing = native_dependencies(native, windows)
        output = new_output(args.output, [ROOT / "python", prefix, Path(sys.prefix), args.physics_root, args.ovui_root, native.parent])
        (output / "native-dependencies.txt").write_text(dependency_listing + "\n")
        target_platform = "windows" if windows else "linux"
        bundle = output / f"OvdNext-{target_platform}-x86_64"
        bundle.mkdir()
        python = copy_python(prefix, bundle, windows)
        copy_tree(ROOT / "python/ovd_ui", bundle / "app/ovd_ui")
        copy_file(native, bundle / "app" / native.name)
        for symbol in native.parent.glob("*.pdb"):
            copy_file(symbol, output / "symbols" / symbol.name)
        distributions = [copy_distribution(metadata.distribution(name), expected, bundle / "app") for name, expected in pins]
        library_notices = stage_libraries(native, bundle, windows, args.msvc_runtime)
        stage_notices(bundle, args.ovui_root, args.physics_root, prefix)
        if not windows:
            prepare_linux_elf(bundle, output / "symbols")
        copy_file(ROOT / "ci/ovui_package/portable_launch.py", bundle / "portable_launch.py")
        if windows:
            (bundle / "OvdNext.cmd").write_bytes(b'@echo off\r\n"%~dp0python\\python.exe" -I -B "%~dp0portable_launch.py" %*\r\nexit /b %errorlevel%\r\n')
        else:
            (bundle / "OvdNext.sh").write_text('#!/bin/sh\nset -eu\nroot=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\nexport PATH=/usr/bin:/bin\nunset LD_PRELOAD PYTHONPATH PYTHONHOME\nexport LD_LIBRARY_PATH="$root/lib:$root/python/lib"\nexec "$root/python/bin/python3.12" -I -B "$root/portable_launch.py" "$@"\n')
            (bundle / "OvdNext.sh").chmod(0o755)
        epoch = int(os.environ.get("SOURCE_DATE_EPOCH") or run(["git", "show", "-s", "--format=%ct", "HEAD"]))
        provenance = {"version": version, "platform": target_platform, "source_commit": run(["git", "rev-parse", "HEAD"]),
                      "source_dirty": bool(dirty), "allow_dirty": args.allow_dirty, "dependency_dirty": dependency_dirty,
                      "physics_commit": PHYSICS_PIN, "ovui_commit": OVUI_PIN,
                      "native_input_sha256": sha256(native), "native_sha256": sha256(bundle / "app" / native.name),
                      "python": runtime_security["python"], "expat": runtime_security["expat"], "pillow": runtime_security["pillow"],
                      **input_records,
                      "source_date_epoch": epoch, "distributions": distributions,
                      "cuda_runtime": {"name": "nvidia-cuda-runtime-cu12", "version": "12.9.79"},
                      "msvc_runtime": args.msvc_runtime.name if args.msvc_runtime else None}
        write_json(bundle / "app/ovd_ui/runtime-build.json", provenance)
        libraries = [{"file": p.relative_to(bundle).as_posix(), "sha256": sha256(p),
                      **library_notices.get(p.relative_to(bundle).as_posix(), {})} for p in sorted(bundle.rglob("*"))
                     if p.is_file() and (p.suffix in {".dll", ".pyd"} or ".so" in p.name)]
        validate_runtime_inventory(bundle, distributions)
        write_json(bundle / "native-libraries.json", libraries)
        expanded_mib = 1 + sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file()) // (1024 * 1024)
        (bundle / "README.txt").write_text(
            f"OvdNext {version} portable application\n\n"
            "Component licenses and notices accompany this application.\n\n"
            "Extract the whole archive. On Windows, run OvdNext.cmd. On Linux, use\n"
            "unzip in a way that retains Unix executable modes, then run ./OvdNext.sh.\n"
            "A capture path may be passed as the first argument. Use --check for\n"
            "isolated import verification, or --check CAPTURE to load a capture and\n"
            "obtain a native presentation without opening the GUI.\n"
            "No host Python, pip, or Qt installation is required. Python and the pinned\n"
            "application runtime are bundled; keep python/, app/ and lib/ beside the\n"
            "launcher.\n"
            f"Expanded payload is approximately {expanded_mib} MiB; allow extra space\n"
            "for capture caches. Exact archive/extracted bytes are in the accompanying\n"
            "platform/version-qualified .zip.package-report.json published beside the\n"
            "archive. ZIP extraction materializes duplicate library aliases as\n"
            "independent files; the report accounts for their extracted bytes.\n\n"
            "Linux baseline: Ubuntu 22.04 x64 with a desktop session, OpenGL 3.3 support\n"
            "with host libGL.so.1/libEGL.so.1 (libgl1/libegl1 or driver libglvnd),\n"
            "and an installed NVIDIA driver providing libcuda.so.1. Host drivers are not\n"
            "bundled. Newer Ubuntu releases require validation before use. A compatible\n"
            "host libstdc++.so.6 and libgcc_s.so.1 are also required; they remain\n"
            "host-provided to share the graphics/driver C++ runtime.\n"
            "Windows baseline: Windows 11 x64 with OpenGL 3.3 support and a current GPU\n"
            "driver, including a Vulkan loader. Host graphics drivers are not bundled.\n\n"
            f"Source state: {'development snapshot (--allow-dirty)' if args.allow_dirty else 'clean source build'}.\n"
            "app/ovd_ui/runtime-build.json records version, commit, dirty state and\n"
            "native-module hash; SHA256SUMS inventories payload bytes. A local dirty\n"
            "snapshot does not prove that the native module matches modified sources.\n"
            "Release CI must rebuild native code from the same clean source revision.\n"
            "See app/ovd_ui/notices/, lib/notices/, python/licenses/ and\n"
            "native-libraries.json for terms and library inventory. CHANGELOG.md contains\n"
            "the full release history.\n")
        inspect_bundle(bundle)
        env = probe_environment(bundle, windows)
        packaged_security = json.loads(run(runtime_security_command(python, bundle / "app"), cwd=output, env=env))
        validate_security_runtime(packaged_security)
        validate_pillow_payload(bundle, packaged_security["pillow"])
        (output / "pillow-native-dependencies.txt").write_text(validate_pillow_static_linkage(bundle / "app", windows) + "\n")
        if packaged_security != runtime_security:
            raise ValueError("Staged runtime security evidence differs from the installed build runtime")
        write_json(output / "runtime-security.json", packaged_security)
        probe = json.loads(run([python, "-I", "-B", bundle / "portable_launch.py", "--check"], cwd=output, env=env))
        if probe.get("version") != version or not probe.get("isolated"):
            raise ValueError("Packaged probe returned wrong version or non-isolated interpreter")
        write_json(output / "probe.json", probe)
        (bundle / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(bundle).as_posix()}\n"
                                              for p in sorted(bundle.rglob("*")) if p.is_file()))
        archive = output / f"{bundle.name}-{version}.zip"
        make_archive(bundle, archive, epoch)
        print(json.dumps(finish_report(bundle, archive, provenance, args.max_archive_mib)))
    except (ValueError, OSError, metadata.PackageNotFoundError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Packaging failed: {error}\n{getattr(error, 'stderr', '') or ''}")


if __name__ == "__main__":
    main()
