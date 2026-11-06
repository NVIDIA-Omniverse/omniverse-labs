# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure live UI cadence and actual presented RTX frames, using wall-clock time."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
from nvidia.ov_library_lab.app import LibraryLab, ui
from nvidia.ov_library_lab.keys import Key


def summary(values):
    return {'median_ms': statistics.median(values)*1000,
            'p95_ms': sorted(values)[int(.95*(len(values)-1))]*1000,
            'max_ms': max(values)*1000}


async def measure(args):
    lab = LibraryLab(args.runtime_root, 'public', 'Balanced')
    report = {'method': 'Live wall-clock native app loop, not recorded or fixed-step video.',
              'ui_max_fps': args.ui_fps, 'runs': []}
    try:
        await lab.start()
        if args.cpu_display:
            lab.viewport.gpu_enabled = False
        report['display'] = 'GPU image upload' if lab.viewport.gpu_enabled else 'CPU pixels'
        report['ui_backend'] = os.environ.get('OMNIUI_BACKEND', 'default')
        report['headless'] = os.environ.get('OMNIUI_HEADLESS','')
        report['nonblocking_desktop'] = lab.nonblocking_desktop
        for quality in args.qualities:
            lab.settings.quality = quality
            for room in ('lobby', 'rendering'):
                lab.go_to(room)
                lab.keys = {Key.RIGHT}
                end = time.perf_counter()+4
                while time.perf_counter() < end:
                    frame = await ui.next_frame()
                    lab.update(frame.dt)
                lab.presented_times.clear()
                updates, loops = [], []
                start = last = time.perf_counter()
                while time.perf_counter()-start < args.seconds:
                    frame = await ui.next_frame()
                    now = time.perf_counter()
                    loops.append(now-last)
                    last = now
                    lab.update(frame.dt)
                    updates.append(time.perf_counter()-now)
                times = lab.presented_times
                intervals = [b-a for a, b in zip(times, times[1:])]
                assert len(times) > 10, 'No sustained RTX image delivery'
                run = {'quality': quality, 'room': room, 'motion_resolution': lab.native.quality_spec[0],
                       'frames': len(times), 'gpu_frames': lab.viewport.gpu_frames, 'delivered_fps': (len(times)-1)/(times[-1]-times[0]),
                       'presentation_intervals': summary(intervals), 'ui_loop': summary(loops),
                       'update_cpu': summary(updates)}
                report['runs'].append(run)
                print(json.dumps(run), flush=True)
        if args.actions:
            lab.keys.clear()
            async def until(predicate,timeout=10.):
                start=time.perf_counter()
                while not predicate():
                    assert time.perf_counter()-start < timeout,'Action did not complete'
                    frame=await ui.next_frame();lab.update(frame.dt)
                return (time.perf_counter()-start)*1000
            lab.go_to('lobby')
            yaw=lab.walker.yaw;frames=lab.native.frames
            lab._on_ui_key(Key.RIGHT,0,True)
            move_ms=await until(lambda:lab.walker.yaw>yaw+.015 and lab.native.frames>=frames+2)
            lab._on_ui_key(Key.RIGHT,0,False)
            lab.request_room('physics');frames=lab.native.frames
            room_ms=await until(lambda:lab.current_room=='physics' and lab.native.frames>=frames+2)
            lab.queue_action('Updating experiment',lab.toggle_physics);frames=lab.native.frames
            physics_ms=await until(lambda:lab.running and lab.native.frames>=frames+2)
            await until(lambda:lab.sim_time>=.5)
            assert lab.native.physics_steps>0
            lab.reset_physics()
            report['actions']={'method':'Application key and queued-button handlers through the live desktop loop; timings include two native frame completions.',
                'turn_to_frames_ms':move_ms,'room_to_frames_ms':room_ms,'physics_start_to_frames_ms':physics_ms}
            print(json.dumps(report['actions']),flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    finally:
        await lab.close()


async def host(args):
    pump = asyncio.create_task(ui.run_async())
    try:
        await measure(args)
    finally:
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--qualities', nargs='+', default=['Studio', 'Balanced', 'Performance'])
    parser.add_argument('--cpu-display', action='store_true', help='Use CPU image transfer for a controlled A/B comparison.')
    parser.add_argument('--ui-fps', type=float, default=60.)
    parser.add_argument('--actions', action='store_true', help='Also measure navigation, room entry and PhysX startup through the live loop.')
    args = parser.parse_args()
    ui.init('Live customer performance check', width=1600, height=1000, max_fps=args.ui_fps)
    try:
        from nvidia.ov_library_lab.pacing import precise_timers, run_event_loop
        with precise_timers():
            run_event_loop(host(args))
    finally:
        ui.shutdown()
