"""Owned HDR conversion storage must not mutate immutable frame snapshots."""
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import engine


def test_hdr_staging_reuses_owned_float_storage_and_preserves_source():
    np = pytest.importorskip("numpy")
    staging = {}
    key = (1, 1, "rgba16f")
    first = struct.pack("<4e", -0.0, 8.0, 0.125, 0.25)
    result = engine._rgba16f_to_float_array(first, staging, key)
    assert result.dtype == np.float32 and result.flags.owndata
    assert result.tolist() == [-0.0, 8.0, 0.125, 1.0]
    second = struct.pack("<4e", 2.0, 4.0, 0.5, 0.0)
    assert engine._rgba16f_to_float_array(second, staging, key) is result
    assert result.tolist() == [2.0, 4.0, 0.5, 1.0]
    assert struct.unpack("<4e", first) == (-0.0, 8.0, 0.125, 0.25)
    assert struct.unpack("<4e", second) == (2.0, 4.0, 0.5, 0.0)


def test_hdr_staging_replaces_storage_on_shape_change_even_with_equal_byte_count():
    pytest.importorskip("numpy")
    staging = {}
    payload = struct.pack("<8e", *range(8))
    previous = engine._rgba16f_to_float_array(payload, staging, (2, 1, "rgba16f"))
    current = engine._rgba16f_to_float_array(payload, staging, (1, 2, "rgba16f"))
    assert current is not previous
    assert len(staging) == 2


def test_hdr_staging_is_per_viewport_and_special_values_survive():
    np = pytest.importorskip("numpy")
    payload = struct.pack("<4e", float("inf"), float("nan"), -0.0, -1.0)
    first = engine._rgba16f_to_float_array(payload, {}, (1, 1, "rgba16f"))
    second = engine._rgba16f_to_float_array(payload, {}, (1, 1, "rgba16f"))
    assert not np.shares_memory(first, second)
    assert np.isinf(first[0]) and np.isnan(first[1]) and np.signbit(first[2])
    assert first[3] == 1.0


def test_ldr_mode_switch_releases_hdr_staging_even_on_texture_cache_hit():
    from ovrtx_blender_example.ovrtx_runtime_client import RenderResult
    result = RenderResult(width=1, height=1, rgba8=b"\0\0\0\xff",
        completed_samples=1, session_completed_samples=1, simulation_time_ns=0)
    staging = {"rgba": object(), "key": (1, 1, "rgba16f")}
    texture = object()
    upload = engine._upload_viewport_texture(SimpleNamespace(), result,
        cached_texture=texture, cached_texture_size=(1, 1, result.frame_format),
        cached_texture_snapshot_index=1, snapshot_index=1, accepts_rgba8=False,
        float_staging=staging)
    assert upload.texture is texture and staging == {}


def test_gpu_prepared_hdr_uses_owned_float_staging_and_preserves_half_plane():
    np = pytest.importorskip("numpy")
    half = struct.pack("<4e", -0.0, 8.0, 0.125, 0.25)
    prepared = struct.pack("<4f", -0.0, 8.0, 0.125, 1.)
    staging = {}
    rgba = engine._rgba16f_to_float_array(half, staging, (1, 1, "rgba16f"),
        prepared_rgba32f=memoryview(prepared))
    assert rgba.tolist() == [-0.0, 8.0, 0.125, 1.]
    assert np.signbit(rgba[0]) and rgba.flags.owndata
    rgba[1] = 999.
    assert struct.unpack("<4e", half)[1] == 8.
    assert struct.unpack("<4f", prepared)[1] == 8.
    assert engine._rgba16f_to_float_array(half, staging, (1, 1, "rgba16f"),
        prepared_rgba32f=prepared) is rgba


def test_prepared_hdr_rejects_wrong_size():
    with pytest.raises(engine.RenderClientError, match="plane size"):
        engine._rgba16f_to_float_array(bytes(8), prepared_rgba32f=bytes(12))
