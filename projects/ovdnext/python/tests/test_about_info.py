# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Version/provenance resource boundary; no native session or desktop needed."""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import zipfile

import pytest

from ovd_ui import __version__, app_info


def metadata():
    return {"version": __version__, "source_commit": "1" * 40,
            "source_dirty": False, "physics_commit": "2" * 40,
            "ovui_commit": "3" * 40, "native_sha256": "4" * 64,
            "source_date_epoch": 1234567890, "os_release": "not displayed"}


def test_canonical_version_and_exact_immutable_display_copy():
    data = metadata()
    info = app_info.parse_about_info(json.dumps(data).encode())
    data["source_commit"] = "5" * 40
    assert info.rows == (("Version", __version__), ("Source revision", "1" * 40),
        ("Source state", "Clean"), ("Physics revision", "2" * 40),
        ("OVUI revision", "3" * 40), ("Native SHA-256", "4" * 64))
    assert info.copy_text == "PVD on OVUI\n" + "\n".join(f"{key}: {value}" for key, value in info.rows)
    assert "1234567890" not in info.copy_text and "not displayed" not in info.copy_text
    with pytest.raises(FrozenInstanceError):
        info.rows = ()
    data["source_dirty"] = True
    assert ("Source state", "Modified") in app_info.parse_about_info(json.dumps(data).encode()).rows


@pytest.mark.parametrize("key,value", [
    ("version", "0.0.0"), ("version", None), ("source_dirty", "false"),
    ("source_dirty", 0), ("source_commit", "main"), ("physics_commit", "2" * 39),
    ("ovui_commit", "<untrusted>"), ("native_sha256", "4" * 63),
])
def test_invalid_metadata_never_displays_partial_provenance(key, value):
    data = metadata()
    data[key] = value
    info = app_info.parse_about_info(json.dumps(data).encode())
    assert info.rows == (("Version", __version__), ("Build information", "Unavailable"))
    assert "invalid" in info.provenance_status


@pytest.mark.parametrize("payload", [b"{", b"[]", b"null", b"\xff", b" " * 65537],
    ids=['incomplete', 'array', 'null', 'invalid-utf8', 'oversized'])
def test_malformed_or_oversized_resource_is_gracefully_unavailable(payload):
    info = app_info.parse_about_info(payload)
    assert info.rows[-1] == ("Build information", "Unavailable")
    assert info.rows[0][1] == __version__


def test_source_checkout_does_not_read_working_directory_build_metadata(tmp_path, monkeypatch):
    (tmp_path / "runtime-build.json").write_text(json.dumps(metadata()))
    monkeypatch.chdir(tmp_path)
    # Source root has no generated resource; a similarly named cwd file cannot
    # make a source launch impersonate a packaged build.
    assert not (Path(app_info.__file__).parent / "runtime-build.json").exists()
    info = app_info.load_about_info()
    assert info.rows[-1] == ("Build information", "Unavailable")
    assert "does not include" in info.provenance_status


def test_loader_reads_selected_package_resource_from_wheel_traversable(tmp_path, monkeypatch):
    wheel = tmp_path / "product.whl"
    with zipfile.ZipFile(wheel, "w") as output:
        output.writestr("ovd_ui/runtime-build.json", json.dumps(metadata()))
    with zipfile.ZipFile(wheel) as archive:
        monkeypatch.setattr(app_info.resources, "files", lambda package: zipfile.Path(archive, "ovd_ui/"))
        info = app_info.load_about_info()
    assert ("Source revision", "1" * 40) in info.rows
    assert len(info.rows) == 6


def test_missing_packaged_resource_retains_only_truthful_version(tmp_path, monkeypatch):
    monkeypatch.setattr(app_info.resources, "files", lambda _: tmp_path)
    info = app_info.load_about_info()
    assert info.copy_text == f"PVD on OVUI\nVersion: {__version__}\nBuild information: Unavailable"
