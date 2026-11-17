# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency preparation rejects altered sources and unintended native codecs."""
import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
from email.parser import BytesParser
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "install_ovui_pillow.py"


def helper():
    assert SCRIPT.is_file(), "The verified Pillow source-build helper is missing"
    spec = importlib.util.spec_from_file_location("ovui_pillow", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence():
    return {"version": "13.0.0", "features": {"zlib": True, "jpg": False,
            "jpg_2000": False, "libtiff": False, "freetype2": False,
            "webp": False, "littlecms2": False, "raqm": False,
            "libimagequant": False, "xcb": False, "avif": False},
            "png_decode": True, "image_color": True, "raw_rgba": True,
            "zlib_version": "1.3.1.zlib-ng", "zlib_ng_version": "2.3.3",
            "vendor_license_sha256": "6c9f0d975b41afaa34d22f55bb8986ce69e5cb7ad327cb2b28820cd425edf5ee",
            "vendor_license_declared": True,
            "build_provenance": {"purpose": "OvdNext PNG icons, raw RGBA and ImageColor",
                "sources": [dict(helper().PILLOW_SOURCE), dict(helper().ZLIB_SOURCE)],
                "build_python": "3.12.16", "disabled_codecs": ["jpeg", "jpeg2000", "tiff", "freetype", "raqm", "lcms", "webp", "imagequant", "xcb", "avif"],
                "zlib_linkage": "static", "build_settings": {"zlib": "enable", "platform-guessing": "disable"}}}


def test_altered_download_never_becomes_a_build_input(tmp_path):
    source = tmp_path / "upstream.tar.gz"
    source.write_bytes(b"altered archive")
    target = tmp_path / "verified.tar.gz"
    with pytest.raises(ValueError, match="SHA-256"):
        helper().download_verified(source.as_uri(), "0" * 64, target)
    assert not target.exists()


def test_verified_download_retains_exact_source_bytes(tmp_path):
    source = tmp_path / "upstream.tar.gz"
    source.write_bytes(b"pinned archive")
    target = tmp_path / "verified.tar.gz"
    helper().download_verified(source.as_uri(),
                               hashlib.sha256(source.read_bytes()).hexdigest(), target)
    assert target.read_bytes() == b"pinned archive"


def test_tampered_cached_source_is_rejected_without_fetching_or_replacement(tmp_path):
    target = tmp_path / "cached.tar.gz"
    target.write_bytes(b"tampered cached archive")
    with pytest.raises(ValueError, match="SHA-256 mismatch for existing source"):
        helper().download_verified((tmp_path / "never-fetched").as_uri(), "0" * 64, target)
    assert target.read_bytes() == b"tampered cached archive"


@pytest.mark.parametrize("name,link", [("../outside", None), ("/absolute", None),
                                      ("source/link", "../../outside")])
def test_source_archive_cannot_escape_build_directory(tmp_path, name, link):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo(name)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = link
            stream.addfile(member)
        else:
            member.size = 7
            stream.addfile(member, io.BytesIO(b"payload"))
    with pytest.raises(ValueError, match="Unsafe"):
        helper().extract_source(archive, tmp_path / "extract")
    assert not (tmp_path / "outside").exists()


def test_source_archive_keeps_valid_files_and_modes(tmp_path):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("source/configure")
        member.size = 7
        member.mode = 0o755
        stream.addfile(member, io.BytesIO(b"payload"))
    root = helper().extract_source(archive, tmp_path / "extract")
    assert (root / "configure").read_bytes() == b"payload"
    if os.name != "nt":
        assert os.access(root / "configure", os.X_OK)


@pytest.mark.parametrize("names", [("a/file", "b/file"), ("source\\file",), ("source/c:file",)])
def test_source_archive_rejects_multiple_roots_and_platform_path_escapes(tmp_path, names):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name in names:
            member = tarfile.TarInfo(name)
            member.size = 7
            stream.addfile(member, io.BytesIO(b"payload"))
    with pytest.raises(ValueError, match="Unsafe"):
        helper().extract_source(archive, tmp_path / "extract")
    assert not (tmp_path / "extract").exists()


def test_wrong_pillow_runtime_version_is_rejected():
    item = evidence()
    item["version"] = "12.3.0"
    with pytest.raises(ValueError, match="13.0.0"):
        helper().validate_pillow_evidence(item)


def test_previous_pillow_and_python_provenance_cannot_qualify_a_new_runtime():
    item = evidence()
    item["build_provenance"]["build_python"] = "3.12.15"
    item["build_provenance"]["sources"][0] = {
        "name": "Pillow", "version": "12.3.0", "filename": "pillow-12.3.0.tar.gz",
        "url": "https://files.pythonhosted.org/packages/1c/3d/bb7fca845737cf9d7dbde16ed1843984665ff2e0a518f5db43e77ec540b9/pillow-12.3.0.tar.gz",
        "sha256": "3b8182a766685eaa002637e28b4ec8d6b18819a0c71f579bf0dbaa5830297cce",
    }
    with pytest.raises(ValueError, match="source provenance"):
        helper().validate_pillow_evidence(item)


@pytest.mark.parametrize("field", ["png_decode", "image_color", "raw_rgba"])
def test_required_image_capability_failure_prevents_release(field):
    item = evidence()
    item[field] = False
    with pytest.raises(ValueError, match="Pillow"):
        helper().validate_pillow_evidence(item)


@pytest.mark.parametrize("feature", ["jpg_2000", "jpg", "libtiff", "freetype2",
                                     "webp", "littlecms2", "raqm", "libimagequant",
                                     "xcb", "avif"])
def test_unintended_native_codec_prevents_release(feature):
    item = evidence()
    item["features"][feature] = True
    with pytest.raises(ValueError, match="Pillow"):
        helper().validate_pillow_evidence(item)


def test_required_zlib_capability_prevents_release_when_absent():
    item = evidence()
    item["features"]["zlib"] = False
    with pytest.raises(ValueError, match="Pillow"):
        helper().validate_pillow_evidence(item)


def test_complete_image_capabilities_are_accepted():
    helper().validate_pillow_evidence(evidence())


@pytest.mark.parametrize("mutation", ["missing", "source_hash", "source_url", "source_version",
                                     "disabled_codecs", "shared_zlib", "python", "build_settings"])
def test_unverified_source_build_is_rejected(mutation):
    item = evidence()
    manifest = item["build_provenance"]
    if mutation == "missing":
        item.pop("build_provenance")
    elif mutation == "source_hash":
        manifest["sources"][1]["sha256"] = "0" * 64
    elif mutation == "source_url":
        manifest["sources"][0]["url"] = "https://example.org/unverified-source.tar.gz"
    elif mutation == "source_version":
        manifest["sources"][1]["version"] = "2.3.2"
    elif mutation == "disabled_codecs":
        manifest["disabled_codecs"].remove("jpeg2000")
    elif mutation == "shared_zlib":
        manifest["zlib_linkage"] = "shared"
    elif mutation == "python":
        manifest["build_python"] = "3.12.13"
    else:
        manifest["build_settings"]["platform-guessing"] = "enable"
    with pytest.raises(ValueError, match="provenance"):
        helper().validate_pillow_evidence(item)


@pytest.mark.parametrize("field,value", [("vendor_license_sha256", None),
                                         ("vendor_license_sha256", "0" * 64),
                                         ("vendor_license_declared", False)])
def test_missing_or_altered_vendor_license_is_rejected(field, value):
    item = evidence()
    item[field] = value
    with pytest.raises(ValueError, match="license"):
        helper().validate_pillow_evidence(item)


@pytest.mark.parametrize("field,value", [("zlib_version", "1.2.11"), ("zlib_ng_version", None),
                                         ("zlib_ng_version", "2.3.2")])
def test_host_compression_library_is_rejected(field, value):
    item = evidence()
    item[field] = value
    with pytest.raises(ValueError, match="zlib-ng"):
        helper().validate_pillow_evidence(item)


def test_host_codec_roots_cannot_override_pinned_zlib_in_child_build(tmp_path):
    module = helper()
    assert hasattr(module, "controlled_build_environment"), "Host library discovery is not isolated"
    # Keep this independent of the helper's blocklist so removing any guard fails.
    blocked = ("AVIF_ROOT", "FREETYPE_ROOT", "HARFBUZZ_ROOT", "FRIBIDI_ROOT",
               "IMAGEQUANT_ROOT", "JPEG2K_ROOT", "JPEG_ROOT", "LCMS_ROOT", "RAQM_ROOT",
               "TIFF_ROOT", "WEBP_ROOT", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS",
               "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "CPATH", "LIBRARY_PATH",
               "LD_RUN_PATH", "LD_LIBRARY_PATH", "INCLUDE", "LIB", "CMAKE_PREFIX_PATH",
               "CMAKE_TOOLCHAIN_FILE", "DISTUTILS_USE_SDK", "MSSDK")
    poisoned = {**os.environ, **dict.fromkeys(blocked, "/host/unverified"), "ZLIB_ROOT": "/host/zlib",
                "PKG_CONFIG": "/host/pkg-config", "PKG_CONFIG_PATH": "/host/pkgconfig",
                "mSsDk": "/host/mixed-case-sdk"}
    prefix = tmp_path / "verified-zlib"
    env = module.controlled_build_environment(poisoned, prefix, tmp_path / "build")
    child = subprocess.run([sys.executable, "-I", "-B", "-c",
                            "import os,json; print(json.dumps(dict(os.environ)))"],
                           env=env, check=True, text=True, capture_output=True)
    actual = json.loads(child.stdout)
    assert actual["ZLIB_ROOT"] == str(prefix)
    assert actual["PKG_CONFIG_PATH"] == ""
    assert not Path(actual["PKG_CONFIG"]).exists()
    assert Path(actual["PKG_CONFIG_LIBDIR"]).is_dir()
    assert actual["PATH"] == poisoned["PATH"]
    assert not set(blocked) & {name.upper() for name in actual}


def test_linux_source_build_replaces_nonportable_pbs_tool_paths(tmp_path, monkeypatch):
    module = helper()
    monkeypatch.setattr(module.sys, "platform", "linux")
    config = {"CC": "clang -pthread", "CXX": "clang++ -pthread",
              "LDCXXSHARED": "clang++ -pthread -shared -Wl,-z,relro -Wl,-z,now"}
    monkeypatch.setattr(module.sysconfig, "get_config_var", config.get)
    def find_tool(name, path):
        assert path == "/selected/build-tools"
        return "/usr/bin/" + name

    monkeypatch.setattr(module.shutil, "which", find_tool)
    source = {"PATH": "/selected/build-tools", "CC": "clang -pthread", "CXX": "clang++ -pthread",
              "AR": "/tools/llvm/bin/llvm-ar", "RANLIB": "/tools/llvm/bin/llvm-ranlib",
              "aR": "/unverified/archiver"}
    env = module.controlled_build_environment(source, tmp_path / "zlib", tmp_path / "build")
    assert {name: env[name] for name in ("CC", "CXX", "AR", "RANLIB")} == {
        "CC": "/usr/bin/cc -pthread", "CXX": "/usr/bin/c++ -pthread",
        "AR": "/usr/bin/ar", "RANLIB": "/usr/bin/ranlib"}
    assert "aR" not in env
    assert env["LDCXXSHARED"] == "/usr/bin/c++ -pthread -shared -Wl,-z,relro -Wl,-z,now"
    assert not {"CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS"} & set(env)
    assert env["PATH"] == source["PATH"]


@pytest.mark.parametrize("cxx,shared", [
    ("clang++ -pthread", "g++ -shared"),
    (None, "clang++ -pthread -shared"),
    ("clang++ -pthread", None),
])
def test_linux_source_build_rejects_unrecognized_shared_linker(tmp_path, monkeypatch, cxx, shared):
    module = helper()
    monkeypatch.setattr(module.sys, "platform", "linux")
    config = {"CC": "clang -pthread", "CXX": cxx, "LDCXXSHARED": shared}
    monkeypatch.setattr(module.sysconfig, "get_config_var", config.get)
    monkeypatch.setattr(module.shutil, "which", lambda name, path: "/usr/bin/" + name)
    with pytest.raises(ValueError, match="shared-linker"):
        module.controlled_build_environment({"PATH": "/usr/bin"},
                                            tmp_path / "zlib", tmp_path / "build")
    assert not (tmp_path / "build").exists()


def test_linux_source_build_reports_missing_tool_before_compiling(tmp_path, monkeypatch):
    module = helper()
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.shutil, "which", lambda name, path: None if name == "ar" else "/usr/bin/" + name)
    with pytest.raises(ValueError, match="ar.*build-essential"):
        module.controlled_build_environment({"PATH": "/usr/bin"}, tmp_path / "zlib", tmp_path / "build")


def test_windows_source_build_keeps_existing_toolchain_flow(tmp_path, monkeypatch):
    module = helper()
    monkeypatch.setattr(module.sys, "platform", "win32")
    source = {"PATH": os.environ["PATH"], "CC": "custom-cl", "CXX": "custom-cl",
              "AR": "custom-lib", "RANLIB": "custom-ranlib"}
    env = module.controlled_build_environment(source, tmp_path / "zlib", tmp_path / "build")
    assert {name: env[name] for name in ("CC", "CXX", "AR", "RANLIB")} == {
        name: source[name] for name in ("CC", "CXX", "AR", "RANLIB")}


def test_provenance_and_vendor_license_are_inventory_members(tmp_path):
    wheel = tmp_path / "pillow-12.3.0-cp312-cp312-win_amd64.whl"
    info = "pillow-12.3.0.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("PIL/__init__.py", b"# upstream package\n")
        archive.writestr(info + "/METADATA", b"Name: pillow\nVersion: 12.3.0\n")
        archive.writestr(info + "/RECORD", "")
    helper().record_wheel_provenance(wheel, {"sources": [{"name": "zlib-ng"}]},
                                     "zlib-ng license\n")
    with zipfile.ZipFile(wheel) as archive:
        report = info + "/ovdnext-build.json"
        license_file = info + "/licenses/ZLIB-NG.txt"
        assert json.loads(archive.read(report))["sources"] == [{"name": "zlib-ng"}]
        assert archive.read(license_file) == b"zlib-ng license\n"
        record = info + "/RECORD"
        rows = list(csv.reader(io.StringIO(archive.read(record).decode())))
        assert len(rows) == len(archive.namelist())
        assert {row[0] for row in rows} == set(archive.namelist())
        for name, digest, size in rows:
            if name == record:
                assert (digest, size) == ("", "")
            else:
                data = archive.read(name)
                expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                assert digest == "sha256=" + expected
                assert int(size) == len(data)
        assert b"License-File: ZLIB-NG.txt\n" in archive.read(info + "/METADATA")
        assert archive.read("PIL/__init__.py") == b"# upstream package\n"


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_vendor_license_metadata_header_precedes_preserved_description(tmp_path, newline):
    wheel = tmp_path / "pillow.whl"
    info = "pillow-12.3.0.dist-info"
    description = newline.join((b"# Pillow description", b"Preserve the upstream text.", b""))
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(info + "/METADATA", newline.join((b"Name: pillow", b"Version: 12.3.0", b"", b"")) + description)
        archive.writestr(info + "/RECORD", "")
    helper().record_wheel_provenance(wheel, {}, "license\n")
    with zipfile.ZipFile(wheel) as archive:
        value = archive.read(info + "/METADATA")
    assert BytesParser().parsebytes(value).get_all("License-File") == ["ZLIB-NG.txt"]
    assert value.split(newline + newline, 1)[1] == description
