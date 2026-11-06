# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""F12 reaches its requested samples when the worker changes mid-job (build 29 review).

Session recovery replaces a lost renderer worker and replays the scene, but the
replacement worker accumulates samples from zero. The final-render job sizes its
chunks from, and reports, the samples the worker says it has accumulated, so the
delivered image holds the full requested samples and its count is the worker's own.
"""

from __future__ import annotations

from pathlib import Path
import sys
import threading

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.ovrtx_runtime_client import RenderClientError, RenderResult  # noqa: E402
from ovrtx_blender_example.render_requests import RenderRequest  # noqa: E402


class _WorkerController:
    """Borrowed controller whose worker accumulates samples like the public SDK worker.

    ``replace_on`` lists the 1-based render calls served by a fresh replacement
    worker (session recovery): its count restarts from zero before the chunk.
    """

    def __init__(self, *, replace_on=(), initial_samples=0, progress=None, stuck=False):
        self.replace_on = set(replace_on)
        self.samples = initial_samples
        self.worker = 1
        self.progress = progress
        self.stuck = stuck
        self.requested: list[int] = []
        self.reported: list[tuple[int, int]] = []
        self.progress_seen: list[int] = []
        self.shutdown_calls = 0

    def ensure(self, _request):
        raise AssertionError("prepared controller must not be ensured again")

    def render(self, _request, *, additional_samples: int) -> RenderResult:
        if self.progress is not None:
            self.progress_seen.append(int(self.progress.get("completed_samples", 0)))
        self.requested.append(additional_samples)
        if len(self.requested) in self.replace_on:
            self.worker += 1
            self.samples = 0
        if self.stuck:
            self.samples = 1
        else:
            self.samples += additional_samples
        self.reported.append((self.worker, self.samples))
        return RenderResult(
            width=1,
            height=1,
            rgba8=bytes((self.worker, 0, 0, 255)),
            completed_samples=self.samples,
            session_completed_samples=100 + len(self.requested),
            simulation_time_ns=0,
            native_timings={"runtime_epoch": self.worker},
        )

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _run(controller, max_samples, progress=None):
    from ovrtx_blender_example import engine as engine_module

    return engine_module._run_final_render_job(
        RenderRequest(max_samples=max_samples),
        threading.Event(),
        progress,
        controller=controller,
        composition=object(),
    )


@pytest.mark.parametrize(
    "max_samples, replace_on, requested",
    (
        # The verifier's case: killed during chunk 2 of 8 samples.
        (8, (2,), [1, 1, 1, 2, 4]),
        # Killed during the last chunk of the shipped default (128 samples).
        (128, (8,), [1, 1, 2, 4, 8, 16, 32, 64, 64]),
        # Two replacements within one job.
        (8, (2, 4), [1, 1, 1, 2, 2, 4]),
    ),
)
def test_final_render_worker_change_reaches_requested_samples(max_samples, replace_on, requested):
    from ovrtx_blender_example import engine as engine_module

    progress = engine_module._final_render_progress()
    controller = _WorkerController(replace_on=replace_on, progress=progress)

    outcome = _run(controller, max_samples, progress)

    assert outcome["status"] == engine_module.FINAL_RENDER_STATUS_COMPLETED
    assert controller.requested == requested
    final_worker, final_samples = controller.reported[-1]
    # The delivered image comes from the last worker and holds every requested sample.
    assert final_worker == 1 + len(replace_on)
    assert outcome["result"].rgba8[0] == final_worker
    assert final_samples == max_samples
    assert outcome["result"].completed_samples == final_samples
    assert outcome["timings"]["sample_chunk_counts"] == [s for _w, s in controller.reported]
    assert outcome["timings"]["accumulation_restarts"] == len(replace_on)
    # Progress never runs past the endpoint and ends on it.
    assert all(0 <= value <= max_samples for value in controller.progress_seen)
    assert progress["completed_samples"] == max_samples
    assert controller.shutdown_calls == 0  # borrowed controller


@pytest.mark.parametrize(
    "initial_samples, max_samples, expected",
    (
        (0, 8, 8),
        # A borrowed worker that had already accumulated: its real count is reported,
        # and progress stays capped at the endpoint.
        (6, 4, 7),
    ),
)
def test_final_render_reports_worker_accumulated_count(initial_samples, max_samples, expected):
    from ovrtx_blender_example import engine as engine_module

    progress = engine_module._final_render_progress()
    controller = _WorkerController(initial_samples=initial_samples, progress=progress)

    outcome = _run(controller, max_samples, progress)

    assert outcome["result"].completed_samples == controller.samples == expected
    assert outcome["result"].session_completed_samples == 100 + len(controller.requested)
    assert outcome["timings"]["accumulation_restarts"] == 0
    assert progress["completed_samples"] == min(expected, max_samples)


def test_final_render_stops_when_the_worker_count_never_advances():
    from ovrtx_blender_example import engine as engine_module

    controller = _WorkerController(stuck=True)

    with pytest.raises(RenderClientError, match="accumulation restarted"):
        _run(controller, 8)
    # One good chunk, then the bounded number of non-advancing ones.
    assert len(controller.requested) == engine_module.FINAL_RENDER_MAX_ACCUMULATION_RESTARTS + 2
