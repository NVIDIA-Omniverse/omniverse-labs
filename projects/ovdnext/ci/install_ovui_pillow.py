#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a codec-minimal Pillow runtime from verified sources, without OpenJPEG.

OvdNext uses PNG decoding, RGBA bytes and ImageColor. The pinned OvUI runtime
does not import Pillow. Disable optional external codec libraries instead of
letting build-host libraries change the distributed application's capabilities.
Pillow's built-in decoders remain available.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request
import uuid
import zipfile

PILLOW_SOURCE = {
    "name": "Pillow", "version": "13.0.0", "filename": "pillow-13.0.0.tar.gz",
    "url": "https://files.pythonhosted.org/packages/ce/a4/9cc30690a394a8305e8be4c9036aa16d0dfddadd09079860b582f562496d/pillow-13.0.0.tar.gz",
    "sha256": "cdcd1ba258178f7356d74170c9a81ba4eab6a5d55748bc402a5a2a6fe6be3510",
}
ZLIB_SOURCE = {
    "name": "zlib-ng", "version": "2.3.3", "filename": "zlib-ng-2.3.3.tar.gz",
    "url": "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.3.3.tar.gz",
    "sha256": "f9c65aa9c852eb8255b636fd9f07ce1c406f061ec19a2e7d508b318ca0c907d1",
}
ZLIB_LICENSE_SHA256 = "6c9f0d975b41afaa34d22f55bb8986ce69e5cb7ad327cb2b28820cd425edf5ee"
DISABLED_CODECS = ("jpeg", "jpeg2000", "tiff", "freetype", "raqm", "lcms",
                   "webp", "imagequant", "xcb", "avif")
DISABLED_FEATURES = ("jpg", "jpg_2000", "libtiff", "freetype2", "raqm",
                     "littlecms2", "webp", "libimagequant", "xcb", "avif")

# Run in a new isolated interpreter so a previously imported stock wheel cannot
# hide a replaced dependency. An optional app directory selects the exact
# installed/staged tree; -I does not accept PYTHONPATH.
PILLOW_PROBE = r'''
import _elementtree,hashlib,io,json,platform,pyexpat,struct,sys,zlib
from importlib import metadata
from pathlib import Path
parser=_elementtree.XMLParser()
parser.feed("<ovd/>")
if parser.close().tag!="ovd":
    raise RuntimeError("Accelerated XML parser failed runtime verification")
if len(sys.argv)>1:
    selected=Path(sys.argv[1]).resolve()
    sys.path.insert(0,str(selected))
import PIL
from PIL import Image,ImageColor,features
if len(sys.argv)>1 and not Path(PIL.__file__).resolve().is_relative_to(selected):
    raise RuntimeError("Pillow was not imported from the selected application tree")
names=("zlib","jpg","jpg_2000","libtiff","freetype2","webp","littlecms2","raqm","libimagequant","xcb","avif")
unknown=[name for name in names if not any(name in table for table in
    (features.modules,features.codecs,features.features))]
if unknown:
    raise RuntimeError(f"Unknown Pillow feature names: {unknown}")
package_root=Path(PIL.__file__).resolve().parent.parent
matching=[]
for directory in package_root.glob("*.dist-info"):
    distribution=metadata.Distribution.at(directory)
    if distribution.metadata.get("Name","").lower()=="pillow":
        matching.append((distribution,directory))
if len(matching)!=1:
    raise RuntimeError("Exactly one Pillow distribution beside the imported package is required")
distribution,info=matching[0]
manifest=info/"ovdnext-build.json"
license_file=info/"licenses/ZLIB-NG.txt"
provenance=json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else None
license_hash=hashlib.sha256(license_file.read_bytes()).hexdigest() if license_file.is_file() else None
def chunk(kind,data):
    return struct.pack(">I",len(data))+kind+data+struct.pack(">I",zlib.crc32(kind+data))
pixels=bytes((255,0,0,255,0,255,0,128))
png=(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",2,1,8,6,0,0,0))
     +chunk(b"IDAT",zlib.compress(b"\x00"+pixels))+chunk(b"IEND",b""))
with Image.open(io.BytesIO(png)) as decoded:
    png_ok=decoded.format=="PNG" and decoded.size==(2,1) and decoded.convert("RGBA").tobytes()==pixels
evidence={"version":PIL.__version__,
          "features":{name:bool(features.check(name)) for name in names},
          "png_decode":png_ok,
          "image_color":ImageColor.getcolor("#12345678","RGBA")== (18,52,86,120),
          "raw_rgba":Image.frombytes("RGBA",(2,1),pixels).tobytes()==pixels,
          "zlib_version":features.version("zlib"),
          "zlib_ng_version":features.version("zlib_ng"),
          "build_provenance":provenance,"vendor_license_sha256":license_hash,
          "vendor_license_declared":"ZLIB-NG.txt" in distribution.metadata.get_all("License-File",[])}
print(json.dumps({"python":platform.python_version(),"expat":pyexpat.EXPAT_VERSION,"pillow":evidence},sort_keys=True))
'''


def validate_pillow_evidence(evidence):
    if not isinstance(evidence, dict) or evidence.get("version") != "13.0.0":
        raise ValueError("Pillow 13.0.0 image capability evidence is required")
    features = evidence.get("features")
    if not isinstance(features, dict) or features.get("jpg_2000") is not False:
        raise ValueError("Pillow JPEG2000/OpenJPEG support must be disabled")
    if evidence.get("png_decode") is not True or features.get("zlib") is not True:
        raise ValueError("Pillow PNG decoding is required")
    if any(evidence.get(name) is not True for name in ("image_color", "raw_rgba")):
        raise ValueError("Pillow ImageColor and raw RGBA support are required")
    if any(features.get(name) is not False for name in DISABLED_FEATURES):
        raise ValueError("Pillow must disable its optional external codec libraries")
    provenance = {"purpose": "OvdNext PNG icons, raw RGBA and ImageColor",
                  "sources": [PILLOW_SOURCE, ZLIB_SOURCE], "build_python": "3.12.16",
                  "disabled_codecs": list(DISABLED_CODECS), "zlib_linkage": "static",
                  "build_settings": {"zlib": "enable", "platform-guessing": "disable"}}
    if evidence.get("build_provenance") != provenance:
        raise ValueError("Pillow verified source provenance is required")
    if (evidence.get("vendor_license_sha256") != ZLIB_LICENSE_SHA256
            or evidence.get("vendor_license_declared") is not True):
        raise ValueError("Pillow pinned zlib-ng license is required")
    if (evidence.get("zlib_version") != "1.3.1.zlib-ng"
            or evidence.get("zlib_ng_version") != ZLIB_SOURCE["version"]):
        raise ValueError("Pillow must use the pinned zlib-ng 2.3.3 runtime")


def controlled_build_environment(source, prefix, work):
    """Suppress host codec discovery while retaining normal compiler lookup."""
    blocked = {"AVIF_ROOT", "FREETYPE_ROOT", "HARFBUZZ_ROOT", "FRIBIDI_ROOT",
               "IMAGEQUANT_ROOT", "JPEG2K_ROOT", "JPEG_ROOT", "LCMS_ROOT",
               "RAQM_ROOT", "TIFF_ROOT", "WEBP_ROOT", "ZLIB_ROOT", "CFLAGS",
               "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH",
               "CPATH", "LIBRARY_PATH", "LD_RUN_PATH", "LD_LIBRARY_PATH", "INCLUDE", "LIB",
               "CMAKE_PREFIX_PATH", "CMAKE_TOOLCHAIN_FILE", "DISTUTILS_USE_SDK", "MSSdk"}
    blocked = {item.upper() for item in blocked}
    if sys.platform == "linux":
        # The upstream install-only runtime retains build-container paths such
        # as /tools/llvm/bin/llvm-ar. Select tools from this build host without
        # altering its sysconfig or replacing the pinned compiler/linker flags.
        blocked.update({"CC", "CXX", "AR", "RANLIB", "ARFLAGS", "CPP",
                        "LDSHARED", "LDCXXSHARED"})
    env = {name: value for name, value in source.items() if name.upper() not in blocked}
    if sys.platform == "linux":
        tools = {}
        for name, executable in (("CC", "cc"), ("CXX", "c++"), ("AR", "ar"), ("RANLIB", "ranlib")):
            path = shutil.which(executable, path=source.get("PATH", os.defpath))
            if path is None:
                raise ValueError(f"Linux source build requires {executable} (install build-essential)")
            # CC/CXX's -pthread is part of the pinned CPython command, separate
            # from CFLAGS. Keep those option tails when changing the executable.
            options = shlex.split(sysconfig.get_config_var(name) or "")[1:] if name in {"CC", "CXX"} else []
            tools[name] = shlex.join([path, *options])
        original_cxx = sysconfig.get_config_var("CXX")
        linker_cxx = sysconfig.get_config_var("LDCXXSHARED")
        if not original_cxx or not linker_cxx or not linker_cxx.startswith(original_cxx):
            raise ValueError("Unexpected pinned Linux C++ shared-linker configuration")
        # setuptools replaces LDSHARED on a CC override, but not LDCXXSHARED.
        tools["LDCXXSHARED"] = tools["CXX"] + linker_cxx[len(original_cxx):]
        env.update(tools)
    work = Path(work).resolve()
    empty = work / "empty-pkgconfig"
    empty.mkdir(parents=True, exist_ok=False)
    # Pillow queries pkg-config even for disabled codecs, before ZLIB_ROOT.
    # Its executable existence check skips this deliberately absent path.
    env.update({"ZLIB_ROOT": str(prefix), "MAX_CONCURRENCY": "4", "PKG_CONFIG_PATH": "",
                "PKG_CONFIG_LIBDIR": str(empty), "PKG_CONFIG": str(work / "pkg-config-disabled")})
    return env


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_verified(url, expected, target):
    """Do not expose unverified bytes as an extractable build input."""
    target = Path(target)
    if target.exists():
        if sha256(target) != expected:
            raise ValueError(f"SHA-256 mismatch for existing source: {target.name}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + "." + uuid.uuid4().hex + ".download")
    try:
        with urllib.request.urlopen(url, timeout=60) as source, partial.open("xb") as output:
            shutil.copyfileobj(source, output)
        if sha256(partial) != expected:
            raise ValueError(f"SHA-256 mismatch for downloaded source: {target.name}")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def extract_source(archive, destination):
    """Pinned tarballs must still contain only ordinary relative source files."""
    destination = Path(destination)
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        roots = set()
        for member in members:
            name = PurePosixPath(member.name)
            if (not name.parts or name.is_absolute() or ".." in name.parts
                    or "\\" in member.name or ":" in member.name
                    or not (member.isfile() or member.isdir())):
                raise ValueError(f"Unsafe source archive member: {member.name}")
            roots.add(name.parts[0])
        if len(roots) != 1:
            raise ValueError("Unsafe source archive: expected one source directory")
        destination.mkdir(parents=True, exist_ok=False)
        stream.extractall(destination, members=members, filter="data")
    return destination / next(iter(roots))


def record_wheel_provenance(wheel, provenance, zlib_license):
    """Keep source identities/licenses in the wheel's real RECORD inventory."""
    wheel = Path(wheel)
    with zipfile.ZipFile(wheel) as archive:
        contents = {entry.filename: archive.read(entry) for entry in archive.infolist()}
        infos = {entry.filename: entry for entry in archive.infolist()}
    records = [name for name in contents if name.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise ValueError("Pillow wheel must have exactly one RECORD")
    record = records[0]
    info = record.rsplit("/", 1)[0]
    contents[info + "/ovdnext-build.json"] = (json.dumps(provenance, sort_keys=True, indent=2) + "\n").encode()
    contents[info + "/licenses/ZLIB-NG.txt"] = zlib_license.encode()
    metadata_file = info + "/METADATA"
    if metadata_file not in contents:
        raise ValueError("Pillow wheel is missing METADATA")
    metadata_bytes = contents[metadata_file]
    newline = b"\r\n" if b"\r\n" in metadata_bytes else b"\n"
    headers, separator, body = metadata_bytes.partition(newline + newline)
    if b"License-File: ZLIB-NG.txt" not in headers.splitlines():
        headers = headers.rstrip(b"\r\n") + newline + b"License-File: ZLIB-NG.txt"
    contents[metadata_file] = headers + (separator or newline) + body
    rows = io.StringIO(newline="")
    writer = csv.writer(rows, lineterminator="\n")
    for name, value in sorted(contents.items()):
        if name == record:
            writer.writerow((name, "", ""))
        else:
            digest = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode()
            writer.writerow((name, "sha256=" + digest, len(value)))
    contents[record] = rows.getvalue().encode()
    replacement = wheel.with_name(wheel.name + ".verified")
    try:
        with zipfile.ZipFile(replacement, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in sorted(contents.items()):
                entry = infos.get(name) or zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                archive.writestr(entry, value)
        replacement.replace(wheel)
    finally:
        replacement.unlink(missing_ok=True)


def run(command, env=None):
    subprocess.run(list(map(str, command)), env=env, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True, help="New build directory; existing files are preserved")
    parser.add_argument("--uv", default="uv", help="uv executable used for local wheel build/install")
    parser.add_argument("--source-cache", type=Path, help="Optional verified archive cache")
    args = parser.parse_args(argv)
    if sys.version_info[:3] != (3, 12, 16) or sys.platform not in {"win32", "linux"}:
        parser.error("Requires the pinned Windows/Linux CPython 3.12.16 venv")
    work = args.work_root.resolve()
    work.mkdir(parents=True, exist_ok=False)
    cache = args.source_cache.resolve() if args.source_cache else work / "sources"
    pillow_archive = download_verified(PILLOW_SOURCE["url"], PILLOW_SOURCE["sha256"], cache / PILLOW_SOURCE["filename"])
    zlib_archive = download_verified(ZLIB_SOURCE["url"], ZLIB_SOURCE["sha256"], cache / ZLIB_SOURCE["filename"])
    pillow = extract_source(pillow_archive, work / "pillow-source")
    zlib = extract_source(zlib_archive, work / "zlib-source")
    prefix, zlib_build = work / "zlib-prefix", work / "zlib-build"
    env = controlled_build_environment(os.environ, prefix, work)
    command = ["cmake", "-S", zlib, "-B", zlib_build, "-G",
               "Visual Studio 17 2022" if sys.platform == "win32" else "Ninja",
               *(["-A", "x64"] if sys.platform == "win32" else []),
               "-DCMAKE_BUILD_TYPE=Release", "-DZLIB_COMPAT=ON", "-DBUILD_SHARED_LIBS=OFF",
               "-DBUILD_TESTING=OFF", "-DWITH_NATIVE_INSTRUCTIONS=OFF",
               "-DCMAKE_POSITION_INDEPENDENT_CODE=ON", "-DCMAKE_INSTALL_LIBDIR=lib",
               "-DCMAKE_INSTALL_PREFIX=" + prefix.as_posix()]
    run(command, env)
    run(["cmake", "--build", zlib_build, "--config", "Release", "--parallel", "4"], env)
    run(["cmake", "--install", zlib_build, "--config", "Release"], env)
    if sys.platform == "win32":
        # Pillow searches for zlib.lib; zlib-ng's MSVC static target is named
        # zlibstatic.lib. Keep the static linkage, using Pillow's expected name.
        shutil.copyfile(prefix / "lib/zlibstatic.lib", prefix / "lib/zlib.lib")
    command = [args.uv, "build", pillow, "--python", sys.executable,
               "--wheel", "--no-build-isolation", "--out-dir", work / "wheel",
               "--config-setting", "zlib=enable", "--config-setting", "platform-guessing=disable"]
    for codec in DISABLED_CODECS:
        command += ["--config-setting", codec + "=disable"]
    run(command, env)
    wheels = list((work / "wheel").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected exactly one built Pillow wheel")
    wheel = wheels[0]
    provenance = {"purpose": "OvdNext PNG icons, raw RGBA and ImageColor",
                  "sources": [PILLOW_SOURCE, ZLIB_SOURCE],
                  "build_python": platform.python_version(),
                  "disabled_codecs": list(DISABLED_CODECS), "zlib_linkage": "static",
                  "build_settings": {"zlib": "enable", "platform-guessing": "disable"}}
    record_wheel_provenance(wheel, provenance, (zlib / "LICENSE.md").read_text(encoding="utf-8"))
    run([args.uv, "pip", "install", "--python", sys.executable, "--reinstall", "--no-deps", wheel])
    installed = subprocess.run([sys.executable, "-I", "-B", "-c", PILLOW_PROBE],
                               check=True, text=True, capture_output=True)
    evidence = json.loads(installed.stdout)
    validate_pillow_evidence(evidence["pillow"])
    from importlib import metadata
    from package_ovui_portable import validate_pillow_static_linkage
    native_imports = validate_pillow_static_linkage(
        Path(metadata.distribution("Pillow").locate_file("")), sys.platform == "win32")
    report = {**provenance, "wheel": wheel.name, "wheel_sha256": sha256(wheel),
              "verified_runtime": evidence, "pillow_native_imports": native_imports}
    if sys.platform == "linux":
        report["build_toolchain"] = {name: env[name] for name in ("CC", "CXX", "AR", "RANLIB", "LDCXXSHARED")}
    (work / "build-report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Pillow preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
