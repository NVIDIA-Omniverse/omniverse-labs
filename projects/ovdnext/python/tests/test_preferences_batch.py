# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real atomic preference files preserve accepted state across failed batches."""
import json
import pytest

from ovd_ui import settings


def test_batch_preserves_unrelated_values_and_owns_accepted_values(tmp_path):
    path = tmp_path / "preferences.json"
    preferences = settings.Preferences(path)
    preferences.set("future_key", {"nested": [1, 2]})
    candidate = {"rendering": {"geometry_scale": 2.5}, "float_precision": 25,
                 "live_port": "localhost:55444", "keep_recordings": False}
    preferences.set_many(candidate)
    candidate["rendering"]["geometry_scale"] = 999
    restored = settings.Preferences(path)
    assert restored.values == preferences.values
    assert restored.get("rendering") == {"geometry_scale": 2.5}
    assert restored.get("future_key") == {"nested": [1, 2]}
    assert restored.get("float_precision") == 25
    assert restored.get("live_port") == 55444
    assert restored.get("keep_recordings") is False
    assert json.loads(path.read_text())["version"] == 1


@pytest.mark.parametrize("failure", ("validation", "serialization", "replace"))
def test_failed_batch_keeps_file_memory_and_no_temporary_file(tmp_path, monkeypatch, failure):
    path = tmp_path / "preferences.json"
    preferences = settings.Preferences(path)
    preferences.set_many({"float_precision": 6, "show_hex": False})
    original = path.read_bytes()
    candidate = {"float_precision": 25, "show_hex": True}
    expected = OSError
    if failure == "validation":
        candidate["keep_recordings"] = "invalid"
        expected = ValueError
    elif failure == "serialization":
        candidate["invalid"] = object()
        expected = TypeError
    else:
        def reject_replace(*_):
            raise OSError("read-only destination")
        monkeypatch.setattr(settings.os, "replace", reject_replace)
    with pytest.raises(expected):
        preferences.set_many(candidate)
    assert path.read_bytes() == original
    assert preferences.get("float_precision") == 6
    assert preferences.get("show_hex") is False
    assert list(tmp_path.iterdir()) == [path]
