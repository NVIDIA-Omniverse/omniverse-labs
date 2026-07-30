# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import ovfmi.population as population


def test_parse_source_does_not_forward_renderer_usd_plugin_paths(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "stage.usda"
    source.touch()
    captured = {}

    monkeypatch.setenv("PXR_PLUGINPATH_NAME", "renderer-plugins")
    monkeypatch.setenv("OV_PXR_PLUGINPATH_2511", "renamed-renderer-plugins")
    monkeypatch.setenv("UNRELATED_SETTING", "preserved")

    def fake_run(*args, **kwargs):
        captured.update(kwargs["env"])

        class Result:
            returncode = 0
            stdout = json.dumps({"instances": {}})
            stderr = ""

        return Result()

    monkeypatch.setattr(population.subprocess, "run", fake_run)

    assert population.parse_source(str(source)) == {"instances": {}}
    assert "PXR_PLUGINPATH_NAME" not in captured
    assert "OV_PXR_PLUGINPATH_2511" not in captured
    assert captured["UNRELATED_SETTING"] == "preserved"
