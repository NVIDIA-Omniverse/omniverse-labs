#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real public-SDK publisher -> vendor-free consumer, with exact pixel checks.

Run using the external runtime's Python. No Blender or scene is required; this
isolates frame transport correctness from renderer capability or scene content.
"""
from __future__ import annotations

import argparse
import importlib
import os
import types
import uuid
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def load_reader(path):
    path = Path(path).resolve()
    name = '_ovstream_smoke_' + uuid.uuid4().hex
    package = types.ModuleType(name)
    package.__path__ = [str(path.parent)]
    package.__package__ = name
    sys.modules[name] = package
    try:
        module = importlib.import_module(name + '.' + path.stem)
    except BaseException:
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + '.'):
                sys.modules.pop(key, None)
        raise
    return module


def resource_state(reader, stream_name, *, shm_directory=None, control_directory=None):
    if sys.platform == 'win32':
        helper = importlib.import_module(reader.__package__ + '.win32_shm')
        return helper.existing_resources(stream_name)
    if shm_directory is None:
        raise ValueError('Supply the mounted POSIX SHM namespace for reader/resource observation; this does not configure the SDK producer')
    shared = Path(shm_directory)
    control = Path(control_directory or os.environ.get('TMPDIR') or '/tmp')
    paths = (shared / ('ovstream-' + stream_name),
             shared / ('sem.ovstream-' + stream_name + '-wake'),
             control / ('ovstream-' + stream_name + '.sock'))
    result = {}
    for path in paths:
        try:
            path.lstat()
        except FileNotFoundError:
            result[str(path)] = False
        else:
            result[str(path)] = True
    return result


def producer_resources_present(state):
    """True when every resource the live producer must own exists.

    OVStream 0.5 on POSIX publishes only the SHM region and the control socket;
    the named ``sem.ovstream-*-wake`` semaphore belonged to OVStream 0.4. It
    stays in the observed set so closure still proves it is absent, but it is
    not required while the producer is live. Windows kernel objects are all
    required.
    """
    required = {path: live for path, live in state.items()
                if not Path(str(path)).name.startswith('sem.ovstream-')}
    return bool(required) and all(required.values())


def assert_closed(states):
    if not states or any(not row or any(row.values()) for row in states.values()):
        raise AssertionError('Owned OVStream resources still exist or were not observed')


_CONSUMER = '''
import importlib.util, json, hashlib, sys
spec = importlib.util.spec_from_file_location('smoke_entry', sys.argv[3])
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
module = entry.load_reader(sys.argv[1])
options = json.loads(sys.argv[4])
meta = json.loads(sys.argv[2])
with module.OvstreamShmReader(stream_name=meta['stream_name'], epoch=meta['epoch'], producer_pid=meta['producer_pid'], **options) as reader:
    copied = reader.copy_frame(module.HeldFrame(**meta))
    assert not {'ovstream', 'ovrtx', 'ovphysx', 'ovstage', 'warp'} & sys.modules.keys()
    print(json.dumps(dict(width=copied.width,height=copied.height,sequence=copied.sequence,pitch=copied.source_pitch_bytes,sha256=hashlib.sha256(copied.rgba8).hexdigest(),vendor_modules_loaded=False)))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--shm-directory', type=Path,
        help='Mounted POSIX SHM namespace for observation only; does not redirect the producer')
    parser.add_argument('--control-directory', type=Path)
    args = parser.parse_args()
    if sys.platform != 'win32' and args.shm_directory is None:
        parser.error('--shm-directory is required for POSIX resource observation')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "runtime"))
    import numpy as np
    import warp as wp
    from ovstream_frames import ShmPublisher
    wp.init()
    reports = []
    streams = set()
    live_resources = []
    reader_path = root / 'addon/ovrtx_blender_example/ovstream_shm.py'
    reader_module = load_reader(reader_path)
    options = {}
    if args.shm_directory is not None:
        options['shm_directory'] = str(args.shm_directory)
    if args.control_directory is not None:
        options['control_directory'] = str(args.control_directory)
    with ShmPublisher(device=args.device) as publisher:
        for width, height in [(7, 5), (7, 5), (1280, 720)]:
            pixels = np.arange(width * height * 4, dtype=np.uint8).reshape(height, width, 4)
            source = wp.array(pixels, dtype=wp.uint8, device=args.device)
            metadata = publisher.publish_rgba_cuda(source, width, height)
            streams.add(metadata['stream_name'])
            observed = resource_state(reader_module, metadata['stream_name'], **options)
            assert producer_resources_present(observed), 'Producer resources not found in observed namespace'
            live_resources.append(observed)
            try:
                publisher.publish_rgba_cuda(source, width, height)
            except RuntimeError as exc:
                assert "still held" in str(exc)
            else:
                raise AssertionError("Unacknowledged frame was overwritten")
            result = subprocess.run([
                sys.executable, "-I", "-B", "-c", _CONSUMER,
                str(reader_path), json.dumps(metadata), str(Path(__file__).resolve()), json.dumps(options),
            ], check=True, capture_output=True, text=True, timeout=30)
            report = json.loads(result.stdout)
            assert report["sha256"] == hashlib.sha256(pixels.tobytes()).hexdigest()
            publisher.release(metadata["copy_token"])
            reports.append(report)
    closed_resources = {name: resource_state(reader_module, name, **options) for name in streams}
    assert_closed(closed_resources)
    report = dict(status="passed", device=args.device, results=reports,
                  held_frame_overwrite_rejected=True, cleanup_passed=True,
                  live_resources=live_resources, resources_after_shutdown=closed_resources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
