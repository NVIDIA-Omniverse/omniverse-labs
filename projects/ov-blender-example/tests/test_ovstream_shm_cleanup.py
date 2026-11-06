# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import ovstream_shm_cleanup as cleanup

pytestmark = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='POSIX unlink ownership fixture; Windows uses kernel reference lifetime')


@pytest.fixture
def registry(tmp_path, monkeypatch):
    def dead(pid, signal):
        raise ProcessLookupError()
    monkeypatch.setattr(cleanup.os, "kill", dead)
    pid = 987654
    name = "blender-" + "a" * 32
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(dict(schema=1,pid=pid,epoch="epoch1",streams=[name])))
    region = tmp_path / ("ovstream-" + name)
    region.write_bytes(struct.pack("<8I2Q3I",0x4D53564F,2,1,3,2,2,64,192,128,1,0,1,pid))
    wake = tmp_path / ("sem.ovstream-" + name + "-wake")
    wake.write_bytes(b"fixture")
    sock = tmp_path / ("ovstream-" + name + ".sock")
    sock.write_bytes(b"fixture")
    return path,pid,[region,wake,sock]


def run(registry):
    path,pid,_ = registry
    return cleanup.cleanup_registry(path,expected_pid=pid,shm_directory=path.parent,control_directory=path.parent)


def test_cleanup_exact_dead_worker_resources_only(registry, tmp_path):
    foreign = tmp_path / "ovstream-foreign"
    foreign.write_bytes(b"preserve")
    removed = run(registry)
    assert set(removed) == {str(p) for p in registry[2]}
    assert foreign.read_bytes() == b"preserve"
    assert run(registry) == []


def test_live_worker_is_never_cleaned(registry, monkeypatch):
    monkeypatch.setattr(cleanup.os, "kill", lambda *_:None)
    with pytest.raises(RuntimeError,match="still exists"):
        run(registry)
    assert all(p.exists() for p in registry[2])


def test_foreign_pid_header_rejects_entire_cleanup(registry):
    with registry[2][0].open("r+b") as stream:
        stream.seek(56)
        stream.write(struct.pack("<I",registry[1]+1))
    with pytest.raises(ValueError,match="another producer"):
        run(registry)
    assert all(p.exists() for p in registry[2])


def test_registry_identity_mismatch_rejected(registry):
    path,pid,_ = registry
    data = json.loads(path.read_text())
    data["pid"] = pid+1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="expected worker"):
        run(registry)


def test_missing_region_still_cleans_recorded_partial_creation(registry):
    registry[2][0].unlink()
    assert len(run(registry)) == 2


def test_symlink_never_followed_or_removed(registry, tmp_path):
    target = tmp_path / "foreign"
    target.write_bytes(b"preserve")
    registry[2][1].unlink()
    registry[2][1].symlink_to(target)
    with pytest.raises(ValueError,match="symlink"):
        run(registry)
    assert target.read_bytes() == b"preserve"
    assert registry[2][0].exists()


def test_untrusted_stream_name_rejected(registry):
    path,pid,_ = registry
    path.write_text(json.dumps(dict(schema=1,pid=pid,epoch="a",streams=["../foreign"])))
    with pytest.raises(ValueError,match="stream names"):
        run(registry)
