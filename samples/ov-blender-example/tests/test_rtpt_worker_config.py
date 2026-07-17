# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the RTPT worker-startup config authoring channel.

Real-GPU evidence (runtime measurements, "How to make the RTPT quality sliders take
effect"): the OVRTX worker ignores the RenderProduct ``omni:rtx:rtpt:*``
attributes but honors the same values as ``/rtx/rtpt/*`` carb settings read from
the worker package's ``ovrtx.config.json`` at process launch. These tests pin
the mapping, the config merge (preserving other opinions and tolerating carb's
trailing commas), package-root resolution, and the atomic best-effort write.
"""

from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import rtpt_worker_config  # noqa: E402
from ovrtx_blender_example.properties import RTPT_RENDER_SETTINGS  # noqa: E402


def test_carb_overrides_use_documented_defaults_when_quality_empty() -> None:
    # Empty quality -> UI defaults (1/3/15/true) -> wire defaults (3/3/15/true):
    # the documented runtime defaults are unchanged by the UI remap.
    overrides = rtpt_worker_config.rtpt_carb_overrides({})
    assert overrides == {
        "rtx": {
            "rtpt": {
                "maxBounces": 3,
                "maxSpecularAndTransmissionBounces": 3,
                "maxVolumeBounces": 15,
                "fireflyFilter": {"enabled": True},
            }
        }
    }


def test_carb_overrides_map_slider_values_to_rtpt_paths() -> None:
    overrides = rtpt_worker_config.rtpt_carb_overrides(
        {
            "rtpt_max_bounces": 8,
            "rtpt_max_specular_and_transmission_bounces": 5,
            "rtpt_max_volume_bounces": 2,
            "rtpt_firefly_filter_enabled": False,
        }
    )
    rtpt = overrides["rtx"]["rtpt"]
    # Max Bounces UI 8 -> wire 10 (+2 camera-ray offset); sub-caps pass through.
    assert rtpt["maxBounces"] == 10
    assert rtpt["maxSpecularAndTransmissionBounces"] == 5
    assert rtpt["maxVolumeBounces"] == 2
    assert rtpt["fireflyFilter"] == {"enabled": False}


def test_carb_overrides_types_are_int_and_bool() -> None:
    overrides = rtpt_worker_config.rtpt_carb_overrides(
        {"rtpt_max_bounces": 4.0, "rtpt_firefly_filter_enabled": 1}
    )
    # UI 4.0 -> wire int 6.
    assert overrides["rtx"]["rtpt"]["maxBounces"] == 6
    assert isinstance(overrides["rtx"]["rtpt"]["maxBounces"], int)
    assert overrides["rtx"]["rtpt"]["fireflyFilter"]["enabled"] is True


def test_carb_paths_derive_from_render_setting_attributes() -> None:
    # The single source of truth authors omni:rtx:rtpt:* attribute names; the
    # carb path drops only the vendor "omni" token.
    for spec in RTPT_RENDER_SETTINGS.values():
        segments = rtpt_worker_config._carb_path_segments(spec.attribute)
        assert segments[0] == "rtx" and segments[1] == "rtpt"


def test_compose_preserves_other_opinions_and_overwrites_rtpt() -> None:
    existing = json.dumps(
        {
            "log": {"level": "Info"},
            "app": {"graphics": {"api": "vulkan"}},
            "rtx": {"rtpt": {"maxBounces": 99, "keepMe": 1}},
        }
    )
    composed = json.loads(
        rtpt_worker_config.compose_worker_config(existing, {"rtpt_max_bounces": 8})
    )
    assert composed["log"] == {"level": "Info"}
    assert composed["app"] == {"graphics": {"api": "vulkan"}}
    # Our owned leaf is overwritten with the wire value (UI 8 -> wire 10);
    # sibling rtx.rtpt keys are preserved.
    assert composed["rtx"]["rtpt"]["maxBounces"] == 10
    assert composed["rtx"]["rtpt"]["keepMe"] == 1


def test_compose_tolerates_trailing_commas_like_carb() -> None:
    # carb's JSON serializer emits trailing commas that json rejects.
    existing = '{\n "log": {"level": "Info"},\n "crashreporter": {"dumpDir": "."},\n}\n'
    composed = json.loads(
        rtpt_worker_config.compose_worker_config(existing, {"rtpt_max_bounces": 2})
    )
    assert composed["crashreporter"] == {"dumpDir": "."}
    # UI 2 -> wire 4.
    assert composed["rtx"]["rtpt"]["maxBounces"] == 4


def test_compose_from_blank_config_authors_full_tree() -> None:
    # Empty quality -> wire default 3.
    composed = json.loads(rtpt_worker_config.compose_worker_config("", {}))
    assert composed["rtx"]["rtpt"]["maxBounces"] == 3


def _worker_command(package_root: Path) -> str:
    exe = package_root.parent / "ovrtx-bridge-server"
    return subprocess.list2cmdline(
        [str(exe), "--address", "127.0.0.1", "--port", "50051", "--package-root", str(package_root)]
    )


def test_worker_config_path_resolves_under_package_root(tmp_path: Path) -> None:
    package_root = tmp_path / "ovrtx-bridge-server"
    package_root.mkdir()
    path = rtpt_worker_config.worker_config_path(_worker_command(package_root))
    assert path == package_root / "ovrtx.config.json"


def test_worker_config_path_none_without_package_root() -> None:
    assert rtpt_worker_config.worker_config_path("ovrtx-bridge-server --port 50051") is None


def test_author_writes_when_changed_and_is_idempotent(tmp_path: Path) -> None:
    package_root = tmp_path / "ovrtx-bridge-server"
    package_root.mkdir()
    config = package_root / "ovrtx.config.json"
    config.write_text('{"log": {"level": "Info"},}\n', encoding="utf-8")
    command = _worker_command(package_root)

    first = rtpt_worker_config.author_worker_config(command, {"rtpt_max_bounces": 8})
    assert first["status"] == "written"
    written = json.loads(config.read_text(encoding="utf-8"))
    assert written["log"] == {"level": "Info"}
    # UI 8 -> wire 10.
    assert written["rtx"]["rtpt"]["maxBounces"] == 10

    again = rtpt_worker_config.author_worker_config(command, {"rtpt_max_bounces": 8})
    assert again["status"] == "unchanged"

    changed = rtpt_worker_config.author_worker_config(command, {"rtpt_max_bounces": 0})
    assert changed["status"] == "written"
    # UI 0 -> wire 2 (direct lighting only).
    assert json.loads(config.read_text(encoding="utf-8"))["rtx"]["rtpt"]["maxBounces"] == 2
    # No leftover temp file after the atomic replace.
    assert not (package_root / "ovrtx.config.json.tmp").exists()


def test_author_skips_when_no_package_root() -> None:
    result = rtpt_worker_config.author_worker_config("ovrtx-bridge-server --port 50051", {})
    assert result["status"] == "skipped"


def test_author_never_raises_on_unwritable_target(tmp_path: Path, monkeypatch) -> None:
    package_root = tmp_path / "ovrtx-bridge-server"
    package_root.mkdir()
    command = _worker_command(package_root)

    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(rtpt_worker_config.Path, "write_text", _boom)
    result = rtpt_worker_config.author_worker_config(command, {"rtpt_max_bounces": 8})
    assert result["status"] == "failed"
