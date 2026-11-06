# SPDX-License-Identifier: Apache-2.0
"""Main-thread engine timers must tolerate an engine freed before they fire."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import engine  # noqa: E402


class _Engine:
    def __init__(self, alive: bool) -> None:
        self.alive = alive
        self.calls = 0

    def as_pointer(self) -> int:
        if not self.alive:
            raise ReferenceError("StructRNA of type OvrtxExampleRenderEngine has been removed")
        return 1

    def absorb(self) -> None:
        self.calls += 1


def test_live_engine_timer_calls_live_engine() -> None:
    target = _Engine(alive=True)
    assert engine._live_engine_timer(target.absorb)() is None
    assert target.calls == 1


def test_live_engine_timer_drops_freed_engine() -> None:
    target = _Engine(alive=True)
    callback = engine._live_engine_timer(target.absorb)
    target.alive = False
    assert callback() is None
    assert target.calls == 0


def test_live_engine_timer_calls_plain_python_owner() -> None:
    calls = []

    class Stub:
        def absorb(self) -> None:
            calls.append(1)

    engine._live_engine_timer(Stub().absorb)()
    assert calls == [1]
