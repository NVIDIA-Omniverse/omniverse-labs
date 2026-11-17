# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Licensing at the portable-package and Python distribution boundaries."""
from email.parser import BytesParser
import hashlib
from importlib import metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_own_notices_retain_complete_apache_terms_and_match_root():
    # PhysX a12f79ce585ffb79e5ffa4848cb45efc3de2f632,
    # physx/documentation/license/LICENSE.md (SHA256 e5f95d1df6853a9398bcf757f8b6c18c67f62ee0777e6987101aac932a49c7fb).
    # Our complete license omits that file's NVIDIA copyright preamble (retained
    # in NOTICE) and uses HTTPS for the two Apache URLs; the terms are unchanged.
    license_bytes = (ROOT / "LICENSE").read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == (
        "c1b9df1275e769f3dbab000d1e457a2d4b0f28eb5da6c77e48dc37eeba202ed7")
    for root_name, packaged_name in (("LICENSE", "OvdNext-LICENSE.txt"), ("NOTICE", "OvdNext-NOTICE.txt")):
        root_bytes = (ROOT / root_name).read_bytes()
        assert (ROOT / "python/ovd_ui/notices" / packaged_name).read_bytes() == root_bytes
        assert b"CC-BY" not in root_bytes
    readme_license = (ROOT / "README.md").read_text(encoding="utf-8").split("## License", 1)[1].split("\n## ", 1)[0]
    assert "CC-BY" not in readme_license


def test_python_distribution_metadata_retains_application_dependency_and_font_licenses(tmp_path):
    # Exercise our actual metadata build, without native compilation, installs,
    # network access or copying machine-local build products into the fixture.
    for name, version in (("setuptools", "80.9.0"), ("wheel", "0.45.1")):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            reason = f"Metadata qualification requires {name}=={version}, got {installed}"
            if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
                pytest.fail(reason)
            pytest.skip(reason)
    source = tmp_path / "python"
    shutil.copytree(ROOT / "python", source, ignore=shutil.ignore_patterns(
        "__pycache__", ".pytest_cache", "*.egg-info", "build", "dist"))
    notices = source / "ovd_ui/notices"
    notices.mkdir(exist_ok=True)
    dependency = notices / "dependency-LICENSE.txt"
    dependency.write_bytes(b"Separately licensed dependency fixture\n")
    output = tmp_path / "metadata"
    output.mkdir()
    result = subprocess.run([sys.executable, "-I", "-B", "setup.py", "dist_info", "--output-dir", str(output)],
                            cwd=source, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    dist_info, = output.glob("*.dist-info")
    info = BytesParser().parsebytes((dist_info / "METADATA").read_bytes())
    assert info["Metadata-Version"] == "2.4"
    assert info["License-Expression"] == "Apache-2.0"
    expected = {"ovd_ui/notices/OvdNext-LICENSE.txt", "ovd_ui/notices/OvdNext-NOTICE.txt",
                "ovd_ui/notices/dependency-LICENSE.txt"}
    expected.update(path.relative_to(source).as_posix() for path in (source / "ovd_ui/assets/fonts").glob("*-OFL.txt"))
    assert len(expected) > 3, "Font notices must also survive metadata generation"
    assert expected <= set(info.get_all("License-File", []))
    for name in expected:
        assert (dist_info / "licenses" / name).read_bytes() == (source / name).read_bytes()
