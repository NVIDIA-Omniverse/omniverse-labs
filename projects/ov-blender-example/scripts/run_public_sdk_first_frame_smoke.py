#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""First-frame preflight through authenticated worker, mapping, owned copy and ACK.

This process stays SDK-free. The --python entrypoint is preserved exactly,
including a Windows venv launcher and relocated paths containing spaces.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
VENDORS = {'ovrtx', 'ovstage', 'ovstream', 'ovphysx', 'warp', 'numpy'}


def local_caches(root):
    # Keep native cache writes in the requested probe workspace. Reuse this
    # owned cache across fresh worker processes; it contains no scene inputs.
    for key, folder in [('OPTIX_CACHE_PATH', 'optix'), ('CUDA_CACHE_PATH', 'cuda'),
                        ('WARP_CACHE_PATH', 'warp'), ('LOCALAPPDATA', 'local'),
                        ('APPDATA', 'appdata'), ('TEMP', 'temp'), ('TMP', 'temp')]:
        path = root / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path.resolve())


def scene_spec(width, height):
    from ovrtx_blender_example.ovrtx_scene_composition import OvrtxSceneComposition
    from ovrtx_blender_example.ovrtx_session import OvrtxSessionSpec
    scene = ROOT / 'tests/fixtures/public_sdk_render.usda'
    composition = OvrtxSceneComposition(str(scene), str(scene), (),
        hashlib.sha256(scene.read_bytes()).hexdigest(), True)
    spec = OvrtxSessionSpec(composition, ('/Render/Camera',), width, height,
                           '/World/Camera', 'composed_scene', output_size_value_route=True)
    return SimpleNamespace(**vars(spec), stage_metadata={
        'upAxis': 'Y', 'metersPerUnit': 1, 'defaultPrim': 'World'})


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    local_caches(args.output.parent / 'native-cache')
    os.environ.update(OV_BLENDER_PUBLIC_SDK_PYTHON=str(Path(args.python).absolute()),
                      OV_BLENDER_PUBLIC_SDK_OUTPUT=str(args.output.resolve() / 'workers'),
                      OV_BLENDER_PUBLIC_SDK_GPU=str(args.gpu))
    sys.path.insert(0, str(ROOT / 'addon'))
    sys.path.insert(0, str(ROOT / 'scripts'))
    from ovrtx_blender_example.direct_sdk_client import PublicSdkRenderClient
    from run_ovstream_raw_frame_smoke import resource_state
    client = PublicSdkRenderClient()
    report = dict(status='running', command=sys.argv, consumer_python=sys.executable,
                  worker_python=args.python, frames=[], descriptors=[])
    frames = []
    copy = client._copy_frame
    def observe(result, width, height, render_var):
        # Capture actual transport identity before opening the named mapping.
        report['descriptors'].append(dict(result['shm']))
        identity = client.runtime.identity
        assert result['shm']['producer_pid'] == identity.pid
        assert result['shm']['epoch'] == identity.session
        assert result['_runtime_pid'] == identity.pid
        pixels = copy(result, width, height, render_var)
        assert client._reader is not None and client._reader._mapping is not None
        return pixels
    client._copy_frame = observe
    try:
        identity = client.start_session(scene_spec(args.width, args.height))
        report['runtime_identity'] = client.runtime.request('health')['runtime_identity']
        report['identity'] = vars(client.runtime.identity)
        if args.require_venv_child:
            assert client.runtime.pid != client.runtime.launcher_pid, 'Probe did not exercise the venv launcher distinction'
        original_worker = client.runtime.identity
        sizes = ((args.width, args.height), (args.width, args.height), (args.width+32, args.height+18))
        for index, (render_var, (width, height)) in enumerate(zip(('LdrColor', 'HdrColor', 'HdrColor'), sizes)):
            if index == 2:
                client.resize_output(identity, width, height)
            frame = client.render_result(identity, selected_sensor_paths=('/Render/Camera',),
                                         render_var=render_var, additional_samples=8)
            frames.append(frame)
            payload = frame.linear_rgba16f if render_var == 'HdrColor' else frame.rgba8
            assert (frame.width, frame.height) == (width, height)
            assert len(payload) == width*height*(8 if render_var == 'HdrColor' else 4)
            assert client.runtime.identity == original_worker
            if render_var == 'HdrColor':
                rgb = [value for pixel in struct.iter_unpack('<4e', payload) for value in pixel[:3]]
                assert all(math.isfinite(v) for v in rgb) and max(rgb) > 1 and max(rgb) > min(rgb)
                assert not frame.rgba8, 'HDR preflight must exercise the optional-LDR-absent contract'
            else:
                rgb = payload[0::4]
                assert max(rgb) > min(rgb)
            assert frame.native_timings['raw_chunk_count'] >= 1
            report['frames'].append(dict(render_var=render_var, width=width, height=height, bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(), minimum=min(rgb), maximum=max(rgb),
                timing=frame.native_timings))
        assert client.runtime.diagnostics()['request_counts']['release_frame'] == 3
        assert report['descriptors'][1]['stream_name'] != report['descriptors'][2]['stream_name']
        report['resize_same_worker'] = dict(before=sizes[1], after=sizes[2], epoch=original_worker.epoch)
        report['frame_resources_after_ack'] = client.runtime.request('health')['frame_resources']
        assert report['frame_resources_after_ack']['held_leases'] == 0
        assert report['frame_resources_after_ack']['active_transfers'] == 0
        assert not VENDORS & sys.modules.keys()
        report.update(status='passed', vendor_modules=[], mapping_copy_and_ack=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            client.shutdown()
            report['runtime'] = client.runtime.diagnostics()
            report['resources'] = {row['stream_name']: resource_state(row['stream_name'])
                                   for row in report['descriptors']}
            assert report['runtime']['closed'] and report['runtime']['returncode'] == 0
            assert not report['runtime']['cleanup_errors']
            assert not any(v for state in report['resources'].values() for v in state.values())
            for frame, result in zip(frames, report['frames']):
                payload = frame.linear_rgba16f if frame.render_var == 'HdrColor' else frame.rgba8
                assert hashlib.sha256(payload).hexdigest() == result['sha256']
            report['owned_pixels_after_worker_death'] = True
        except BaseException as error:
            report.update(status='failed', teardown_error=f'{type(error).__name__}: {error}')
            raise
        finally:
            (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--width', type=int, default=65)
    parser.add_argument('--height', type=int, default=49)
    parser.add_argument('--require-venv-child', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
