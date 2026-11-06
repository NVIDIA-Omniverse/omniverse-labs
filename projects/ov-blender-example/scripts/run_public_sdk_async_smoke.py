#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise native asynchronous tickets through the vendor-free add-on client."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import struct
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    public = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(OV_BLENDER_PUBLIC_SDK_PYTHON=args.python,
                      OV_BLENDER_PUBLIC_SDK_OUTPUT=str(args.output.resolve() / 'workers'),
                      OV_BLENDER_PUBLIC_SDK_GPU=str(args.gpu))
    sys.path.insert(0, str(public / 'addon'))
    from ovrtx_blender_example.direct_sdk_client import PublicSdkRenderClient
    from ovrtx_blender_example.ovrtx_scene_composition import OvrtxSceneComposition
    from ovrtx_blender_example.ovrtx_session import OvrtxSessionSpec
    from ovrtx_blender_example.ovrtx_runtime_client import RenderClientError
    scene = public / 'tests/fixtures/public_sdk_render.usda'
    composition = OvrtxSceneComposition(str(scene), str(scene), (),
        hashlib.sha256(scene.read_bytes()).hexdigest(), True)
    sensor = '/Render/Camera'
    spec = OvrtxSessionSpec(composition, (sensor,), args.width, args.height, '/World/Camera',
                           'composed_scene', output_size_value_route=True)
    spec = SimpleNamespace(**vars(spec), stage_metadata={
        'upAxis': 'Y', 'metersPerUnit': 1, 'defaultPrim': 'World'})
    client = PublicSdkRenderClient()
    report = {'status': 'running', 'frames': []}
    try:
        identity = client.start_session(spec)
        client.render_result(identity, selected_sensor_paths=(sensor,), render_var='LdrColor', additional_samples=4)
        pid, epoch = client.runtime.pid, client.runtime.epoch
        assert client.supports_async_render_sample()
        def begin(render_var='LdrColor'):
            start = time.monotonic()
            ticket = client.begin_render_sample(identity, selected_sensor_paths=(sensor,), render_var=render_var)
            return ticket, time.monotonic() - start
        def consume(ticket):
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                ticket.wait(10)
                frame = client.poll_render_sample(ticket)
                if frame is not None:
                    if frame.render_var == 'HdrColor':
                        assert len(frame.linear_rgba16f) == frame.width * frame.height * 8
                        red = [pixel[0] for pixel in struct.iter_unpack('<4e', frame.linear_rgba16f)]
                        assert max(red) > min(red)
                        # HDR presentation can omit the optional LDR plane.
                        if frame.rgba8:
                            assert len(frame.rgba8) == frame.width * frame.height * 4
                    else:
                        assert len(frame.rgba8) == frame.width * frame.height * 4
                        assert max(frame.rgba8[0::4]) > min(frame.rgba8[0::4])
                    report['frames'].append({'samples': frame.session_completed_samples,
                        'width': frame.width, 'height': frame.height,
                        'time_ns': frame.simulation_time_ns, 'render_var': frame.render_var,
                        'sha256': hashlib.sha256(frame.linear_rgba16f if frame.render_var == 'HdrColor' else frame.rgba8).hexdigest(),
                        'frame_format': frame.frame_format,
                        'rgba8_bytes': len(frame.rgba8), 'linear_rgba16f_bytes': len(frame.linear_rgba16f)})
                    return frame
                time.sleep(.001)
            raise TimeoutError('Native asynchronous sample did not complete')
        first, first_seconds = begin()
        second, second_seconds = begin('HdrColor')
        a, b = consume(first), consume(second)
        assert b.session_completed_samples > a.session_completed_samples
        assert b.simulation_time_ns > a.simulation_time_ns
        cancelled, _ = begin()
        cancelled.cancel()
        try:
            client.poll_render_sample(cancelled)
        except RenderClientError:
            pass
        else:
            raise AssertionError('Cancelled ticket produced a frame')
        next_ticket, _ = begin()
        consume(next_ticket)
        assert (client.runtime.pid, client.runtime.epoch) == (pid, epoch)
        assert not ({'ovrtx', 'ovstage', 'ovstream', 'ovphysx', 'warp'} & sys.modules.keys())
        report.update(status='passed', pid=pid, epoch=epoch,
                      begin_seconds=[first_seconds, second_seconds],
                      cancelled_then_resumed=True, same_worker=True)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        client.shutdown()
        report['runtime'] = client.runtime.diagnostics()
        (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
