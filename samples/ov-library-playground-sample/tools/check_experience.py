# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native customer-flow checks; API-driven, separate from visible input QA."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
from nvidia.ov_library_lab.app import LibraryLab, ui
from nvidia.ov_library_lab.pacing import precise_timers


async def check(args):
    lab = LibraryLab(args.runtime_root, 'public', 'Performance')
    checks = []
    def passed(name, **data):
        checks.append({'name': name, **data})
        print('PASS', name, data, flush=True)
    async def frames(count=10):
        for _ in range(count):
            frame = await ui.next_frame()
            lab.update(frame.dt)
    try:
        await lab.start()
        from nvidia.ov_library_lab.extensions import value
        if value('ROOM_RESTRICTIONS'):
            from nvidia.ov_library_lab.internal_features import check_public_profile
            await check_public_profile(lab, frames)
        else:
            from nvidia.ov_library_lab.content import ROOMS
            assert len(ROOMS)==8 and lab.overlay is None
            assert not hasattr(lab.gallery, 'graph_cell')
            assert not hasattr(lab.gallery.extra, 'conveyor_check')
            assert lab.settings.profile=='public'
        passed('public distribution has only enabled public experiments')

        # The source viewer must resolve all relocated examples and its fallback guide.
        from nvidia.ov_library_lab.panels import show_example
        from nvidia.ov_library_lab.content import ROOMS
        for key in ROOMS:
            show_example(lab, key)
            await frames(2)
            assert lab.example_window.visible, key
            lab.example_window.destroy()
            lab.example_window = None
        passed('every room opens its relocated example or integration guide')

        lab.go_to('physics')
        for friction in (.05, .65):
            lab.compare_friction(friction)
            deadline = time.monotonic()+20
            while lab.comparison_run:
                assert time.monotonic() < deadline
                await frames(1)
        low, high = (lab.physics_results[x] for x in (.05, .65))
        assert low > high+.05, (low, high)
        passed('two-second comparison measures actual PhysX travel', low_m=low, high_m=high)

        def fail():
            raise RuntimeError('Intentional recovery check')
        lab.queue_action('Recovery check', fail)
        assert lab.activity.text == 'Recovery check...'
        await frames(4)
        assert lab.last_failed_action is not None and 'failed' in lab.activity.text
        calls = []
        lab.last_failed_action = ('Recovery check', lambda: calls.append(True))
        lab.retry_action()
        await frames(4)
        assert calls == [True] and lab.last_failed_action is None
        passed('failed actions stay recoverable; retry executes exactly once')

        lab.queue_action('Old room action', lambda: calls.append('stale'))
        lab.request_room('scene')
        await frames(4)
        assert 'stale' not in calls
        passed('changing rooms cancels a queued old-room action')

        g, w = lab.gallery, lab.gallery.workshops
        w.set_fleet(6)
        g.shared_offset = g.shared_target = 2.4
        w.update_fleet()
        w.lift_cycle = True
        lab.gallery.extra.set_lift(.9)
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in lab.storage.root.glob('*.usda')}
        lab.start_over()
        await frames(10)
        assert lab.current_room == 'lobby'
        assert w.fleet_count == 1 and abs(g.shared_offset+2.4) < .001
        assert not w.lift_cycle and lab.gallery.extra.lift_value == .25
        assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in lab.storage.root.glob('*.usda')}
        passed('start over restores experiments and preserves saved files')

        for quality in ('Balanced', 'Performance'):
            lab.settings.quality = quality
            deadline = time.monotonic()+20
            expected = lab.settings.render_spec()[0]
            while lab.viewport.presented_size != expected:
                assert time.monotonic() < deadline
                await frames(1)
            passed('quality changes resize real RTX output', quality=quality, resolution=expected)

        lab.go_to('rendering')
        lab.set_render_subject('car')
        lab.panel_modes['rendering'] = 'camera'
        lab.request_panel()
        deadline = time.monotonic()+25
        while not lab.observer_view or lab.observer_view.pixels is None:
            assert time.monotonic() < deadline
            await frames(1)
        lab.remember_view()
        reference = lab.reference_pixels.copy()
        w.set_light(angle=75, size=.4)
        await frames(30)
        assert (lab.reference_pixels == reference).all()
        from nvidia.ov_library_lab.panels import start_snapshot
        start_snapshot(lab)
        while not lab.snapshot_task.done():
            await frames(1)
        assert lab.snapshot_task.result().is_file()
        passed('RTX reference remains fixed while a fresh image can be saved')

        # Repeated entry, reset and secondary-camera activation checks resource lifecycle.
        for _ in range(args.cycles):
            for room in ('scene', 'streaming', 'interface', 'physics', 'rendering', 'storage', 'lobby'):
                lab.go_to(room)
                await frames(8)
            lab.start_over()
            await frames(8)
        passed('repeated room changes and session resets', cycles=args.cycles)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(checks, indent=2))
    finally:
        await lab.close()


async def host(args):
    pump = asyncio.create_task(ui.run_async())
    try:
        await check(args)
    finally:
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=5)
    args = parser.parse_args()
    ui.init('Customer-flow verification', width=1600, height=1000, max_fps=120)
    try:
        with precise_timers():
            asyncio.run(host(args))
    finally:
        ui.shutdown()
