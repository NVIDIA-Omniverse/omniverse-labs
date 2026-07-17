# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import validate_suite  # noqa: E402


JUNK_SHOP_GOLDEN = (
    validate_suite.ROOT
    / "docs"
    / "goldens"
    / "perf_junk_shop_1280x720"
    / "single-frame"
)


def _copy_junk_shop_golden(destination: Path) -> dict:
    metadata = json.loads((JUNK_SHOP_GOLDEN / "metadata.json").read_text())
    (destination / "frame.png").write_bytes(
        (JUNK_SHOP_GOLDEN / "frame.png").read_bytes()
    )
    return metadata


def test_dependency_runtime_paths_use_packaged_public_executables(tmp_path: Path) -> None:
    roots = {
        component: str(tmp_path / component)
        for component in validate_suite.validation.DEPLOYED_ROOT_NAMES
    }
    for component in ("ovrtx-bridge-server", "ovphysx-bridge-server"):
        executable = Path(roots[component]) / "bin" / component
        executable.parent.mkdir(parents=True)
        executable.touch()

    paths = validate_suite.validation._dependency_runtime_paths_from_roots(roots)

    assert Path(paths["ovrtx_worker"]) == (
        Path(roots["ovrtx-bridge-server"]) / "bin" / "ovrtx-bridge-server"
    )
    assert Path(paths["ovphysx_server"]) == (
        Path(roots["ovphysx-bridge-server"]) / "bin" / "ovphysx-bridge-server"
    )


def test_golden_evidence_accepts_explicit_operator_approval() -> None:
    evidence = validate_suite.validation._golden_evidence(
        "perf_junk_shop_1280x720", JUNK_SHOP_GOLDEN
    )

    assert evidence["status"] == "pass"


@pytest.mark.parametrize(
    "approval",
    (
        {"approved_by": None, "approved_at_utc": None},
        {"approved_by": "operator"},
        {"approved_at_utc": "2026-07-17T03:50:33Z"},
        {"approved_by": "operator", "approved_at_utc": "not-a-timestamp"},
        [],
    ),
)
def test_golden_evidence_rejects_unapproved_or_malformed_approval(
    tmp_path: Path, approval: object
) -> None:
    metadata = _copy_junk_shop_golden(tmp_path)
    metadata["approval"] = approval
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    evidence = validate_suite.validation._golden_evidence(
        "perf_junk_shop_1280x720", tmp_path
    )

    assert evidence == {
        "status": "unavailable",
        "reason": (
            "golden is not explicitly approved: approval requires a non-empty "
            "approved_by and a valid UTC approved_at_utc"
        ),
    }


def test_golden_evidence_still_rejects_a_digest_mismatch(tmp_path: Path) -> None:
    metadata = _copy_junk_shop_golden(tmp_path)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "frame.png").write_bytes(b"not the approved frame")

    evidence = validate_suite.validation._golden_evidence(
        "perf_junk_shop_1280x720", tmp_path
    )

    assert evidence == {
        "status": "unavailable",
        "reason": "golden artifact digest does not match metadata",
    }


def test_inventory_classifies_every_validation_entry_point() -> None:
    scripts = Path(validate_suite.ROOT / "scripts")
    entry_points = {path.name for path in scripts.glob("run_*.py")}
    performance = {
        script
        for suite in validate_suite.inventory.PERFORMANCE.values()
        if isinstance(suite, dict)
        for script, *_metadata in suite.values()
    }
    classified = {
        *validate_suite.inventory.INTEGRATION_PROBES,
        *validate_suite.inventory.EXCLUDED_PROBES,
        *performance,
    }

    assert entry_points == classified


def test_performance_selection_uses_inventory_order_and_rejects_nonmembers() -> None:
    assert validate_suite._performance_selection(
        "performance-large",
        ("blender-navigation-hdr", "blender-navigation-hdr"),
    ) == ("blender-navigation-hdr",)

    with pytest.raises(validate_suite.validation.ValidationError):
        validate_suite._performance_selection("performance-large", ("golden",))
    with pytest.raises(validate_suite.validation.ValidationError):
        validate_suite._performance_selection(
            "performance-small", ("blender-navigation",)
        )


@pytest.mark.parametrize("suite", ("performance-large", "performance-small"))
def test_main_rejects_invalid_selection_before_runtime_discovery(
    suite: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        validate_suite,
        "_runtime",
        lambda *_args: pytest.fail("runtime discovery must not begin"),
    )

    assert validate_suite.main((suite, "--measurement", "not-a-member")) == 2
    assert "error" in json.loads(capsys.readouterr().out)


def test_blender_integration_uses_shared_executable_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discovered = tmp_path / "Blender 5.1" / "blender"
    probes = []
    monkeypatch.setattr(validate_suite, "_runtime", lambda *_args: {})
    monkeypatch.setattr(validate_suite, "blender_executable", lambda: discovered)
    monkeypatch.setattr(validate_suite, "_pytest", lambda _items: 0)
    monkeypatch.setattr(
        validate_suite,
        "_probe",
        lambda *_args: probes.append(_args[-1]) or True,
    )

    assert (
        validate_suite.main(
            (
                "blender-integration",
                "--runtime-root",
                str(tmp_path / "runtime"),
                "--output-dir",
                str(tmp_path / "output"),
            )
        )
        == 0
    )
    assert probes == [discovered]


def test_blender_integration_probe_forwards_discovered_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}
    monkeypatch.setattr(
        validate_suite.validation,
        "_fixture_manifest_record",
        lambda *_args: ("expected-workload",),
    )
    monkeypatch.setattr(validate_suite.validation, "_load_result", lambda *_args: {})
    monkeypatch.setattr(
        validate_suite.validation,
        "_live_transform_color_presentation",
        lambda *_args, **_kwargs: "scene-linear-hdr",
    )
    monkeypatch.setattr(validate_suite.validation, "_write_json", lambda *_args: None)
    monkeypatch.setattr(
        validate_suite.validation, "_probe_failure", lambda *_args: None
    )
    monkeypatch.setattr(
        validate_suite.validation,
        "_probe_fixture_identity",
        lambda _result: "expected-workload",
    )
    executor = SimpleNamespace(
        run=lambda *_args, **kwargs: captured.update(kwargs) or object()
    )
    blender = tmp_path / "Blender 5.1" / "blender"

    assert validate_suite._probe(
        executor,
        tmp_path,
        {"ovrtx_blender_client": "client", "ovrtx_worker_command": "worker"},
        blender,
    )
    assert captured["env"] == {"BLENDER_COMMAND": str(blender)}


def test_empty_performance_selection_preserves_measurement_record(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert validate_suite._performance("performance-small", {}, None, ()) == 0
    assert json.loads(capsys.readouterr().out) == {
        "suite": "performance-small",
        "measurements": [],
    }


@pytest.mark.parametrize(
    ("requested", "expected"),
    (
        (None, ("blender-navigation-ldr", "blender-navigation-hdr")),
        (("blender-navigation-hdr",), ("blender-navigation-hdr",)),
    ),
)
def test_performance_executes_selected_inventory_members(
    requested: tuple[str, ...] | None,
    expected: tuple[str, ...],
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    commands: list[list[str]] = []
    blender = tmp_path / "Blender 5.1" / "blender"
    monkeypatch.setattr(validate_suite, "blender_executable", lambda: blender)

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(validate_suite.subprocess, "run", run)
    selected = (
        None
        if requested is None
        else validate_suite._performance_selection("performance-large", requested)
    )
    assert validate_suite._performance(
        "performance-large",
        {"ovrtx_blender_client": "client", "ovrtx_worker_command": "worker"},
        tmp_path / "results",
        selected,
    ) == 0

    measurements = validate_suite.inventory.PERFORMANCE["performance-large"]
    measurement_commands = commands[::2]
    report_commands = commands[1::2]
    assert [Path(command[1]).name for command in measurement_commands] == [
        measurements[name][0] for name in expected
    ]
    for name, command in zip(expected, measurement_commands, strict=True):
        assert command[0] == sys.executable
        assert command[command.index("--output") + 1].endswith(
            f"{name}.json"
        )
        assert command[command.index("--native-client-path") + 1] == "client"
        assert command[command.index("--worker-command") + 1] == "worker"
        assert command[command.index("--blender-command") + 1] == str(blender)
        assert (
            command[command.index("--color-presentation") + 1]
            == measurements[name][1]
        )
    assert [Path(command[1]).name for command in report_commands] == [
        "report_navigation.py"
    ] * len(expected)
    assert [Path(command[2]).name for command in report_commands] == [
        f"{name}.json" for name in expected
    ]
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {
        "suite": "performance-large",
        "selected": list(expected),
        "completed": list(expected),
        "measurements": list(expected),
    }


def test_performance_retries_a_cold_cache_failure_after_a_sibling_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    outcomes = iter((1, 0, 0, 0, 0))
    monkeypatch.setattr(
        validate_suite,
        "blender_executable",
        lambda: Path("/prepared/blender"),
    )
    monkeypatch.setattr(
        validate_suite.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=next(outcomes)),
    )

    assert validate_suite._performance(
        "performance-large",
        {"ovrtx_blender_client": "client", "ovrtx_worker_command": "worker"},
        tmp_path / "results",
    ) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["completed"] == [
        "blender-navigation-ldr",
        "blender-navigation-hdr",
    ]
