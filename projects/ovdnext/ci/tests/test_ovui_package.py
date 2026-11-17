# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distribution boundary checks; the clean-venv probe exercises native runtime."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tomllib
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("package_ovui", ROOT / "ci/package_ovui_linux.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


def test_retired_wheelhouse_cannot_run_dependency_commands_or_create_output(tmp_path, monkeypatch):
    output = tmp_path / "legacy-package"
    monkeypatch.setattr(sys, "argv", [str(spec.origin), "--output", str(output), "--allow-dirty"])
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    def reject_command(*args, **kwargs):
        pytest.fail("The retired wheelhouse attempted a dependency command or download")
    monkeypatch.setattr(package, "run", reject_command)
    with pytest.raises(SystemExit, match="Superseded by ci/package_ovui_portable.py"):
        package.main()
    assert not output.exists()


def test_retired_wheelhouse_cli_reports_portable_replacement_without_outputs(tmp_path):
    output = tmp_path / "legacy-package"
    result = subprocess.run([sys.executable, "-B", str(spec.origin), "--output", str(output)],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert "Superseded by ci/package_ovui_portable.py" in result.stderr
    assert "python -B ci/package_ovui_portable.py" in result.stderr
    assert not output.exists()


def test_runtime_closure_and_canonical_version_agree():
    metadata = tomllib.loads((ROOT / "python/pyproject.toml").read_text())
    pins = [line for line in (ROOT / "python/requirements-runtime.txt").read_text().splitlines() if line and not line.startswith("#")]
    assert sorted(pins) == sorted(metadata["project"]["dependencies"])
    namespace = {}
    exec((ROOT / "python/ovd_ui/__init__.py").read_text(), namespace)
    assert namespace["__version__"] == package.release_version((ROOT / "CHANGELOG.md").read_text())
    assert not any(line.startswith(("pytest", "pybind11", "fastapi", "uvicorn", "requests")) for line in pins)
    assert hashlib.sha256((ROOT / "python/ovd_ui/assets/colors.json").read_bytes()).hexdigest() == "66b3238e9ad1ecf71cd6e95073f3f68113ffa137087ebc30c092ccf9049ed0da"


@pytest.mark.parametrize("invalid", ["0.1.14.dev0", "00.1.14", "0.1.14+build", "1.2"])
def test_invalid_newest_release_never_falls_back_to_older_version(invalid):
    with pytest.raises(ValueError):
        package.release_version(f"# Changelog\n\n## v{invalid} (2026-09-05)\n\n## v0.1.13 (2026-09-04)\n")


def wheel(tmp_path, *, omit=None, corrupt_font=False, filename="pvd_ovui-0.1.14-cp312-cp312-linux_x86_64.whl"):
    path = tmp_path / filename
    assets = ROOT / "python/ovd_ui/assets"
    entries = {p.relative_to(ROOT / "python").as_posix(): p.read_bytes() for p in assets.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".py"}
    entries.update({"_pvd_native.cpython-312-x86_64-linux-gnu.so": b"native fixture",
                    "ovd_ui/runtime-build.json": b"{}", "ovd_ui/notices/NOTICES.txt": b"notice",
                    "ovd_ui/notices/CHANGELOG.md": b"changelog",
                    "pvd_ovui-0.1.14.dist-info/METADATA": b"Name: pvd-ovui\nVersion: 0.1.14\n"})
    if omit:
        del entries[omit]
    if corrupt_font:
        entries["ovd_ui/assets/fonts/Geist-400.ttf"] = b"wrong font bytes"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


@pytest.mark.parametrize("missing", ["ovd_ui/assets/colors.json", "ovd_ui/assets/icons/play.png", "ovd_ui/assets/icons/live.svg", "ovd_ui/notices/CHANGELOG.md", "_pvd_native.cpython-312-x86_64-linux-gnu.so"])
def test_distribution_rejects_missing_native_assets_and_notices(tmp_path, missing):
    with pytest.raises(ValueError):
        package.inspect_product_wheel(wheel(tmp_path, omit=missing))


@pytest.mark.parametrize("name", ["visible", "hidden", "opaque", "transparent"])
@pytest.mark.parametrize("suffix", ["svg", "png"])
def test_distribution_rejects_missing_object_tree_icons(tmp_path, name, suffix):
    with pytest.raises(ValueError, match="Missing packaged control icon"):
        package.inspect_product_wheel(wheel(tmp_path, omit=f"ovd_ui/assets/icons/{name}.{suffix}"))


def test_distribution_rejects_altered_fonts_and_pure_python_tag(tmp_path):
    with pytest.raises(ValueError, match="font differs"):
        package.inspect_product_wheel(wheel(tmp_path, corrupt_font=True))
    with pytest.raises(ValueError, match="platform-tagged"):
        package.inspect_product_wheel(wheel(tmp_path, filename="pvd_ovui-0.1.14-py3-none-any.whl"))


def test_hash_lock_contains_exact_wheel_bytes_and_rejects_duplicate_distribution(tmp_path):
    path = wheel(tmp_path)
    package.inspect_product_wheel(path)
    records = package.write_lock(tmp_path, tmp_path / "requirements.lock")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert records[0]["sha256"] == expected
    assert f"pvd-ovui==0.1.14 --hash=sha256:{expected}" in (tmp_path / "requirements.lock").read_text()
    wheel(tmp_path, filename="pvd_ovui-0.1.14-cp312-cp312-other.whl")
    with pytest.raises(ValueError, match="Multiple wheels"):
        package.write_lock(tmp_path, tmp_path / "bad.lock")


def test_archive_metadata_is_deterministic_and_rejects_links(tmp_path):
    source = tmp_path / "bundle"
    source.mkdir()
    (source / "hello.txt").write_text("native package\n")
    first, second = tmp_path / "a.tar.gz", tmp_path / "b.tar.gz"
    package.deterministic_archive(source, first, 1700000000)
    package.deterministic_archive(source, second, 1700000000)
    assert first.read_bytes() == second.read_bytes()
    try:
        (source / "escape").symlink_to(tmp_path / "outside")
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege unavailable; exercised on Linux")
        raise
    with pytest.raises(ValueError, match="symlink"):
        package.deterministic_archive(source, tmp_path / "linked.tar.gz", 1700000000)


@pytest.mark.skipif(os.name == "nt", reason="POSIX installer; exercised on Linux")
def test_installer_relative_destination_is_anchored_before_entering_bundle(tmp_path):
    bundle, caller, commands = (tmp_path / name for name in ("bundle", "caller", "commands"))
    for directory in (bundle, caller, commands):
        directory.mkdir()
    script = bundle / "install.sh"
    script.write_bytes((ROOT / "ci/ovui_package/install.sh").read_bytes())
    log = tmp_path / "uv-args.txt"
    for name, body in (("sha256sum", "exit 0"), ("uv", 'printf "%s\\n" "$@" >> "$TEST_UV_LOG"')):
        path = commands / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)
    env = {**os.environ, "PATH": f"{commands}:{os.environ['PATH']}", "TEST_UV_LOG": str(log)}
    subprocess.run(["bash", str(script), "new parent/runtime"], cwd=caller, env=env, check=True, capture_output=True)
    assert str(caller / "new parent/runtime") in log.read_text().splitlines()
    assert str(caller / "new parent/runtime/bin/python") in log.read_text().splitlines()
    assert str(bundle / "new parent/runtime") not in log.read_text()
    (caller / "existing").mkdir()
    before = log.read_text()
    rejected = subprocess.run(["bash", str(script), "existing"], cwd=caller, env=env, capture_output=True)
    assert rejected.returncode != 0 and log.read_text() == before

@pytest.mark.parametrize("identity", ["libQt5Core.so.5", "libqwt.so.6", "PySide6", "PyQt5"])
def test_wheelhouse_rejects_qt_runtime_identity(tmp_path, identity):
    path = wheel(tmp_path)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(identity + "/METADATA" if identity.startswith("Py") else identity, b"fixture")
    with pytest.raises(ValueError, match="Qt"):
        package.inspect_product_wheel(path)

def test_installed_probe_accepts_current_notice_set_and_requires_remaining_notices(tmp_path):
    spec = importlib.util.spec_from_file_location("installed_probe", ROOT / "ci/ovui_package/probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    notices = tmp_path / "notices"
    notices.mkdir()
    required = ("OvdNext-LICENSE.txt", "OvdNext-NOTICE.txt",
                "OVUI-LICENSE.txt", "OVUI-THIRD_PARTY_NOTICES.md", "Physics-LICENSE.md",
                "PVD-source-notice.txt", "PxShared-source-notice.txt",
                "pybind11-LICENSE.txt", "CHANGELOG.md", "NOTICES.txt")
    for name in required:
        (notices / name).write_text("retained notice\n" * 20)
    probe.validate_notices(tmp_path)
    for name in required:
        path = notices / name
        content = path.read_bytes()
        path.unlink()
        with pytest.raises((AssertionError, FileNotFoundError)):
            probe.validate_notices(tmp_path)
        path.write_bytes(content)
