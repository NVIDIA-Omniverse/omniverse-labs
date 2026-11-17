# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen expectations fail closed if data is lost, truncated or edited."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import reference_data


@pytest.mark.parametrize("autocrlf", ["true", "false"])
@pytest.mark.parametrize("relative_path, expected_sha", [
    *[(f"python/tests/reference_data/{name}.json", pin["sha256"])
      for name, pin in reference_data.PINS.items()],
    ("python/ovd_ui/assets/colors.json",
     "66b3238e9ad1ecf71cd6e95073f3f68113ffa137087ebc30c092ccf9049ed0da"),
])
def test_checkout_preserves_pinned_json_bytes(tmp_path, autocrlf, relative_path, expected_sha):
    """Windows checkout conversion must not invalidate independent asset hashes."""
    repo = Path(__file__).resolve().parents[2]
    git = ["git", "-C", str(tmp_path), "-c", f"core.autocrlf={autocrlf}"]
    subprocess.run([*git, "init", "--quiet"], check=True, capture_output=True)
    (tmp_path / ".gitattributes").write_bytes((repo / ".gitattributes").read_bytes())
    # Canonical Git blobs are LF, regardless of this test runner's own checkout.
    payload = (repo / relative_path).read_bytes().replace(b"\r\n", b"\n")
    blob = subprocess.run([*git, "hash-object", "-w", "--stdin"], input=payload,
                          check=True, capture_output=True).stdout.decode().strip()
    checked_out = subprocess.run([*git, "cat-file", "--filters", f"--path={relative_path}", blob],
                                 check=True, capture_output=True).stdout
    assert hashlib.sha256(checked_out).hexdigest() == expected_sha


def test_all_frozen_corpora_load_with_complete_case_counts():
    assert callable(getattr(reference_data, "load", None)), "Missing checked corpus loader"
    for name in ("camera", "camera_basis", "near_plane", "graph_rectangles",
                 "orbit_topology", "profiling", "float_format_windows", "float_format_linux"):
        assert reference_data.load(name)


def test_one_byte_mutation_is_rejected_before_parsing(tmp_path):
    payload = bytearray((reference_data.ROOT / "camera.json").read_bytes())
    payload[-2] = ord("]")
    (tmp_path / "camera.json").write_bytes(payload)
    with pytest.raises(ValueError, match="SHA-256"):
        reference_data.load("camera", root=tmp_path)


def test_missing_corpus_is_a_failure_not_a_skip(tmp_path):
    with pytest.raises(FileNotFoundError):
        reference_data.load("camera", root=tmp_path)


@pytest.mark.parametrize("name", ["camera", "camera_basis", "graph_rectangles",
                                  "orbit_topology", "profiling", "float_format_windows", "near_plane"])
def test_regeneration_cannot_silently_drop_cases(tmp_path, monkeypatch, name):
    document = json.loads((reference_data.ROOT / (name + ".json")).read_bytes())
    data = document["data"]
    (data["positions"] if name == "near_plane" else data).pop()
    payload = json.dumps(document).encode()
    (tmp_path / (name + ".json")).write_bytes(payload)
    # Simulate an intentional hash refresh without approving reduced coverage.
    monkeypatch.setitem(reference_data.PINS[name], "sha256", hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="counts"):
        reference_data.load(name, root=tmp_path)
