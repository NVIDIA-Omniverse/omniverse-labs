#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Test-only Pillow: taken from the consumer/test environment (scripts/requirements-public-sdk-consumer.txt), never the shipped SDK runtime.
"""Capture real Blender UI edits and reject unchanged viewport evidence.

Requires an X11 display, xdotool, a C compiler with X11 headers, and Pillow in
this orchestrator Python. No NVIDIA SDK is imported by the orchestrator or
Blender. Captures are unmodified window pixels; the GIF is a sequence of those
captures, not a synthesized UI or an uninterrupted screen recording.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

from PIL import Image, ImageChops, ImageStat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--startup-timeout', type=float, default=300)
    parser.add_argument('--edit-timeout', type=float, default=120)
    parser.add_argument('--leave-open', action='store_true')
    parser.add_argument('--extended-materials', action='store_true',
                        help='Also exercise initially omitted OpenPBR inputs and texture wiring')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    private, review = output / 'private', output / 'review'
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    review.mkdir(parents=True, exist_ok=True)
    executable = private / 'capture_x11_window'
    subprocess.run(['cc', str(source / 'scripts/capture_x11_window.c'), '-O2', '-lX11', '-o', str(executable)], check=True)
    env = dict(os.environ, OV_BLENDER_PUBLIC_SDK_OUTPUT=str(private), OV_BLENDER_PUBLIC_SDK_GPU=str(args.gpu))
    if not env.get('DISPLAY'):
        raise RuntimeError('Set DISPLAY to the intended X11 session')
    env.setdefault('OV_BLENDER_PUBLIC_SDK_TIMEOUT_S', '300')
    log = (private / 'blender.log').open('w')
    process = subprocess.Popen([str(source / 'scripts/launch_public_sdk_blender.sh'),
        '--window-geometry', '30', '50', '1440', '900', '--python-exit-code', '17',
        '--python', str(source / 'scripts/probe_public_sdk_ui_edits.py')], env=env,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    report = {'status': 'running', 'blender_pid': process.pid, 'captures': [], 'edits': []}
    command_id = 0
    control = private / 'ui-edits'
    def status():
        if process.poll() is not None:
            raise RuntimeError(f'Blender exited {process.returncode}; see {log.name}')
        path = control / 'status.json'
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if value['errors']:
            raise RuntimeError('; '.join(value['errors']))
        if value['sdk_imports_in_blender']:
            raise RuntimeError('Native SDK imported inside Blender')
        return value
    def wait(predicate, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = status()
            if value and predicate(value):
                return value
            time.sleep(0.25)
        raise TimeoutError('Blender edit did not reach its acceptance condition')
    def send(action, **kwargs):
        nonlocal command_id
        command_id += 1
        path = control / 'command.tmp'
        path.write_text(json.dumps(dict(id=command_id, action=action, **kwargs)))
        path.replace(control / 'command.json')
    window = None
    def capture(label):
        raw = subprocess.check_output([str(executable), window], env=env)
        image = Image.open(io.BytesIO(raw)).convert('RGB')
        if max(hi - lo for lo, hi in image.getextrema()) < 10:
            raise AssertionError('Window capture is blank')
        path = review / (label + '.png')
        image.save(path)
        report['captures'].append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                   'source': 'XGetImage of owned Blender window', 'redactions': 'none'})
        return image
    def difference(before, after):
        # Analyze the viewport only; retain the entire original window capture.
        box = (80, 160, 1120, 790)
        return sum(ImageStat.Stat(ImageChops.difference(before.crop(box), after.crop(box))).mean) / 3
    def same_session(before, after):
        first, second = before['presented'][0]['session'], after['presented'][0]['session']
        assert all(value is not None for value in first.values()), f'Missing lifecycle proof: {first}'
        assert first == second, f'Value edit replaced or reloaded its live session: {first} -> {second}'
    try:
        ready = wait(lambda s: bool(s['presented']) and s['presented'][0]['samples'] >= 32, args.startup_timeout)
        windows = subprocess.check_output(['xdotool', 'search', '--all', '--onlyvisible', '--pid', str(process.pid), '--class', 'Blender'], env=env, text=True).splitlines()
        if len(windows) != 1:
            raise RuntimeError(f'Expected exactly one owned Blender window, got {windows}')
        window = windows[0]
        subprocess.run(['xdotool', 'windowactivate', '--sync', window, 'mousemove', '--window', window, '600', '700', 'key', 'Escape'], env=env, check=True)
        time.sleep(1)
        previous = capture('01-initial')
        previous_hash = ready['presented'][0]['sha256']
        started = time.monotonic()
        subprocess.run(['xdotool', 'mousemove', '--window', window, '600', '415', 'key',
                        'g', 'x', '1', 'period', '5', 'Return', 'r', 'z', '3', '0', 'Return'], env=env, check=True)
        state = wait(lambda s: abs(s['cube_location'][0]-1.5) < 0.001 and s['presented'] and
                     s['presented'][0]['sha256'] != previous_hash and s['presented'][0]['samples'] >= 32)
        same_session(ready, state)
        time.sleep(0.5)
        current = capture('02-moved-rotated')
        delta = difference(previous, current)
        assert delta > 1, f'Transform did not visibly change viewport: {delta}'
        report['edits'].append({'action': 'keyboard translate X 1.5; rotate Z 30 degrees', 'seconds': time.monotonic()-started, 'viewport_mae': delta, 'status': state})
        previous = current
        steps = [('material', {'rgba': [0.025, 0.08, 0.65, 1], 'roughness': 0.22}, '03-material-blue', 1.0),
                 ('light', {'energy': 650, 'color': [0.6, 0.75, 1]}, '04-light-change', 1.0),
                 ('add', {}, '05-added-object', 0.2),
                 ('delete', {}, '06-deleted-object', 0.2),
                 ('geometry', {}, '07-geometry-change', 0.2),
                 ('camera', {'x': 4}, '08-camera-change', 1.0)]
        if args.extended_materials:
            steps += [
                ('material_sockets', {'values': {'IOR': 2.2, 'Coat Weight': 1.0,
                    'Coat Roughness': 0.03, 'Coat IOR': 2.4}}, '09-coat-ior', 0.15),
                ('material_sockets', {'values': {'Emission Color': [0.0, 0.8, 0.05, 1],
                    'Emission Strength': 4.0}}, '10-emission', 1.0),
                ('material_sockets', {'values': {'Emission Strength': 0.0,
                    'Coat Weight': 0.0}}, '11-lobes-disabled', 1.0),
                ('texture', {'connected': True}, '12-texture-connected', 1.0),
                ('texture', {'connected': False}, '13-texture-disconnected', 1.0),
            ]
        for action, values, label, threshold in steps:
            previous_state = state
            previous_hash = state['presented'][0]['sha256']
            started = time.monotonic()
            send(action, **values)
            state = wait(lambda s: s['command_id'] == command_id and s['presented'] and
                         s['presented'][0]['sha256'] != previous_hash and s['presented'][0]['samples'] >= 32,
                         args.edit_timeout)
            same_session(previous_state, state)
            time.sleep(0.5)
            current = capture(label)
            delta = difference(previous, current)
            assert delta > threshold, f'{action} had no meaningful visible effect: {delta}'
            report['edits'].append({'action': action, 'seconds': time.monotonic()-started, 'viewport_mae': delta, 'status': state})
            previous = current
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (private / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        if report['captures']:
            frames = [Image.open(review / capture['file']).convert('P', palette=Image.Palette.ADAPTIVE) for capture in report['captures']]
            frames[0].save(review / 'blender-ui-edits.gif', save_all=True, append_images=frames[1:], duration=1400, loop=0)
            (review / 'manifest.json').write_text(json.dumps({'capture_method': 'Unmodified X11 window captures',
                'gif': 'Sequence of captures, not continuous screen recording', 'captures': report['captures']}, indent=2) + '\n')
            for frame in frames:
                frame.close()
        if not args.leave_open and process.poll() is None:
            try:
                send('quit')
                process.wait(timeout=20)
            except (subprocess.TimeoutExpired, OSError):
                process.terminate()
                process.wait(timeout=10)
        log.close()
    print(json.dumps({'status': report['status'], 'edits': len(report['edits']), 'review': str(review)}))


if __name__ == '__main__':
    main()
