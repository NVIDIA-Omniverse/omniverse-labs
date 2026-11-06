# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner cleanup retries transient Windows sharing errors and stays consistent.

GAP-CLOSE (OMPE-115792 n-02-06): a generation's freshly exported ``textures``
directory was occasionally still held when the deferred scene close removed
it (WinError 145), the close ended 'failed', and the owners stayed accepted
after their physics snapshots had been cleared.
"""
from __future__ import annotations

import errno
from pathlib import Path
import shutil
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import engine  # noqa: E402
from ovrtx_blender_example import scene_generation  # noqa: E402
from ovrtx_blender_example import scene_generation_sessions as sessions  # noqa: E402

REAL_RMTREE = shutil.rmtree


def sharing_error(winerror: int = 145) -> OSError:
    exc = OSError(errno.ENOTEMPTY, "The directory is not empty")
    exc.winerror = winerror
    return exc


class FlakyRmtree:
    """Fail the first ``failures`` removals of ``target`` (or every one if None)."""

    def __init__(self, target: Path, failures: int | None, error=sharing_error):
        self.target, self.failures, self.error, self.calls = Path(target), failures, error, 0

    def __call__(self, path, *args, **kwargs):
        if Path(path) == self.target:
            self.calls += 1
            if self.failures is None or self.calls <= self.failures:
                raise self.error()
        return REAL_RMTREE(path, *args, **kwargs)


@pytest.fixture
def sleeps(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(scene_generation, "_cleanup_sleep", calls.append)
    return calls


def make_owner(tmp_path: Path, name: str = "generations") -> scene_generation.SceneGenerationOwner:
    owner = scene_generation.SceneGenerationOwner(tmp_path / name)
    textures = owner.work_directory / "generation-000001-0000000000000000" / "stock-export" / "textures"
    textures.mkdir(parents=True)
    (textures / "color_0C0C0C.exr").write_bytes(b"exr")
    owner._current = object()  # noqa: SLF001 - an accepted generation
    return owner


@pytest.mark.parametrize("winerror", [5, 32, 145])
def test_transient_sharing_error_is_retried_with_backoff(monkeypatch, tmp_path, sleeps, winerror):
    owner = make_owner(tmp_path)
    flaky = FlakyRmtree(owner.work_directory, 3, lambda: sharing_error(winerror))
    monkeypatch.setattr(scene_generation.shutil, "rmtree", flaky)

    assert owner.close() is None

    assert flaky.calls == 4
    assert sleeps == list(scene_generation.CLEANUP_RETRY_DELAYS_SECONDS[:3])
    assert not owner.work_directory.exists()
    assert owner.current_generation is None


def test_persistent_sharing_error_is_bounded_and_retains_state(monkeypatch, tmp_path, sleeps):
    owner = make_owner(tmp_path)
    generation = owner.current_generation
    flaky = FlakyRmtree(owner.work_directory, None)
    monkeypatch.setattr(scene_generation.shutil, "rmtree", flaky)

    with pytest.raises(scene_generation.SceneGenerationError, match="cleanup failed.*not empty"):
        owner.close()

    delays = scene_generation.CLEANUP_RETRY_DELAYS_SECONDS
    assert flaky.calls == len(delays) + 1
    assert sleeps == list(delays)
    assert sum(delays) < 5
    assert owner.current_generation is generation


def test_persistent_sharing_error_can_defer_the_residue(monkeypatch, tmp_path, sleeps):
    owner = make_owner(tmp_path)
    monkeypatch.setattr(scene_generation.shutil, "rmtree", FlakyRmtree(owner.work_directory, None))

    assert owner.close(defer_residue=True) == owner.work_directory
    assert owner.current_generation is None
    assert owner.close(defer_residue=True) is None  # already closed


@pytest.mark.skipif(sys.platform != "win32", reason="needs Windows file sharing semantics")
def test_real_windows_sharing_violation_is_retried_until_the_handle_closes(tmp_path):
    owner = make_owner(tmp_path)
    texture = next(owner.work_directory.rglob("color_0C0C0C.exr"))
    stream = texture.open("rb")  # CRT handles do not share delete: unlink -> WinError 32
    closer = threading.Timer(0.15, stream.close)
    closer.start()
    try:
        assert owner.close() is None
    finally:
        closer.cancel()
        stream.close()
    assert not owner.work_directory.exists()


def test_non_transient_error_is_not_retried(monkeypatch, tmp_path, sleeps):
    owner = make_owner(tmp_path)
    flaky = FlakyRmtree(owner.work_directory, None, lambda: OSError("busy"))
    monkeypatch.setattr(scene_generation.shutil, "rmtree", flaky)

    with pytest.raises(scene_generation.SceneGenerationError, match="cleanup failed"):
        owner.close(defer_residue=True)

    assert flaky.calls == 1 and sleeps == []
    assert owner.current_generation is not None


# --------------------------------------------------------------------------- session close
@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(sessions, "_CLOSE_TASK", None)
    monkeypatch.setattr(sessions, "_runtimes", {})
    monkeypatch.setattr(sessions, "_owners", {})
    monkeypatch.setattr(sessions, "_unclosed_owners", [])
    monkeypatch.setattr(sessions, "_deferred_cleanup_paths", [])
    monkeypatch.setattr(sessions, "_physics_snapshots", {})
    monkeypatch.setattr(sessions, "_stop_preparation_worker", lambda **kwargs: set())
    monkeypatch.setattr(engine, "_ENGINE_RUNTIMES", {})
    monkeypatch.setattr(engine, "_WORKER_RESTART", None)
    return sessions


def close_and_wait(defer_native: bool = True) -> dict:
    sessions.close(defer_native=defer_native)
    assert sessions._CLOSE_TASK["completion"].wait(10)
    return sessions.close_status()


def test_transient_failure_during_deferred_close_closes_cleanly(monkeypatch, tmp_path, sleeps, session):
    owner = make_owner(tmp_path)
    session._owners[7] = owner
    session._physics_snapshots[7] = object()
    flaky = FlakyRmtree(owner.work_directory, 2)
    monkeypatch.setattr(scene_generation.shutil, "rmtree", flaky)

    assert close_and_wait() == {"status": "closed"}
    assert session._owners == {} and session._physics_snapshots == {}
    assert not owner.work_directory.exists()
    assert session.deferred_cleanup_paths() == ()
    session._require_session_open()


def test_persistent_residue_is_deferred_and_removed_by_the_next_close(monkeypatch, tmp_path, sleeps, session):
    owner = make_owner(tmp_path)
    session._owners[7] = owner
    session._physics_snapshots[7] = object()
    flaky = FlakyRmtree(owner.work_directory, None)
    monkeypatch.setattr(scene_generation.shutil, "rmtree", flaky)
    monkeypatch.setattr(sessions.shutil, "rmtree", flaky)

    assert close_and_wait() == {"status": "closed"}
    assert session._owners == {}  # no accepted owner without its snapshot
    assert session.deferred_cleanup_paths() == (owner.work_directory,)
    assert owner.work_directory.exists()
    session._require_session_open()

    flaky.failures = flaky.calls  # the other handle has closed
    assert close_and_wait() == {"status": "closed"}
    assert session.deferred_cleanup_paths() == ()
    assert not owner.work_directory.exists()


def test_permanent_failure_never_leaves_an_accepted_owner(monkeypatch, tmp_path, sleeps, session):
    broken, healthy = make_owner(tmp_path, "broken"), make_owner(tmp_path, "healthy")
    session._owners.update({7: broken, 8: healthy})
    session._physics_snapshots.update({7: object(), 8: object()})
    flaky = FlakyRmtree(broken.work_directory, None, lambda: OSError("busy"))
    monkeypatch.setattr(scene_generation.shutil, "rmtree", flaky)

    status = close_and_wait()
    assert status["status"] == "failed" and "cleanup failed" in status["error"]
    # Snapshots were cleared, so neither owner may stay reachable or accepted.
    assert session._owners == {} and session._physics_snapshots == {}
    assert not healthy.work_directory.exists()  # one failure does not skip the others
    assert broken.current_generation is not None and session._unclosed_owners == [broken]
    with pytest.raises(RuntimeError, match="cleanup failed"):
        session._require_session_open()

    # The next close retries the unreachable owner but does not fail on it again.
    assert close_and_wait() == {"status": "closed"}
    assert session._unclosed_owners == [broken]

    flaky.failures = flaky.calls
    assert close_and_wait() == {"status": "closed"}
    assert session._unclosed_owners == []
    assert not broken.work_directory.exists()
    session._require_session_open()


def test_synchronous_close_uses_the_same_path(monkeypatch, tmp_path, sleeps, session):
    owner = make_owner(tmp_path)
    session._owners[7] = owner
    monkeypatch.setattr(scene_generation.shutil, "rmtree", FlakyRmtree(owner.work_directory, 1))

    assert close_and_wait(defer_native=False) == {"status": "closed"}
    assert session._owners == {} and sleeps == [scene_generation.CLEANUP_RETRY_DELAYS_SECONDS[0]]
