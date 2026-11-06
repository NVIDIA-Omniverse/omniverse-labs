from types import SimpleNamespace
import json
import pytest
from ovrtx_blender_example.physics_bake_capture import (
    BakeCaptureSettings, BakeCaptureError, _SampleWriter,
)


@pytest.mark.parametrize("values", [
    (3, 1, 5, 24), (1, 3, 2, 24), (1, 1, 2, 0), (1.0, 1, 2, 24),
])
def test_invalid_ranges_fail_before_capture(values):
    with pytest.raises(ValueError):
        BakeCaptureSettings(*values)


@pytest.mark.parametrize("base", [0, -1, float("nan"), float("inf")])
def test_invalid_fps_base(base):
    with pytest.raises(ValueError):
        BakeCaptureSettings(1, 1, 2, 24, base)


def test_schedule_preroll_substeps_and_fractional_fps():
    settings = BakeCaptureSettings(-4, 3, 10, 24, 1.001, 3, 4)
    samples = [step for step in range(settings.total_steps + 1) if settings.sample_step(step)]
    assert samples == [28, 40, 52]
    assert settings.total_steps == 56
    assert settings.timing()["requested_step_seconds"] == [1001, 96000]
    assert abs(settings.timestep_ns / 1e9 - 1001 / 96000) < 1e-9


def test_artifact_complete_roster_integrity_and_bounded_iteration(tmp_path):
    writer = _SampleWriter(tmp_path, ("/A", "/B"))
    poses = [SimpleNamespace(prim_path=p, translate=(1, 2, 3), orient=(0, 0, 0, 1)) for p in writer.paths]
    with pytest.raises(BakeCaptureError, match="incomplete"):
        writer.append(1, 0, poses[:1], {}, 10)
    writer.append(1, 0, poses, {}, 10)
    writer.append(2, 1, poses, {}, 10)
    artifact = writer.finish({"output_paths": ["/A"]})
    assert artifact.manifest()["sample_count"] == 2
    assert [p["simulation_time_ns"] for p in artifact.samples()] == [0, 10]
    path = tmp_path / "samples.jsonl"
    path.write_text(path.read_text().replace('"frame":2', '"frame":3'))
    with pytest.raises(BakeCaptureError, match="samples changed"):
        list(artifact.samples())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({}))
    with pytest.raises(BakeCaptureError, match="manifest changed"):
        artifact.manifest()


def test_abort_discards_partial_samples(tmp_path):
    writer = _SampleWriter(tmp_path, ("/A",))
    writer.abort()
    assert not tuple(tmp_path.iterdir())
