#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove frame-coherent uint32 semantics through the owned public-SDK SHM client."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(OV_BLENDER_PUBLIC_SDK_PYTHON=args.python, OV_BLENDER_PUBLIC_SDK_OUTPUT=str(args.output.resolve() / 'workers'), OV_BLENDER_PUBLIC_SDK_GPU=args.gpu)
    sys.path.insert(0, str(root / 'addon'))
    from ovrtx_blender_example.direct_sdk_client import PublicSdkRenderClient
    from ovrtx_blender_example.semantic_output import decode_id_map, save_capture
    from ovrtx_blender_example.ovrtx_scene_composition import OvrtxSceneComposition
    source = root / 'tests/fixtures/public_sdk_semantic.usda'
    composition = OvrtxSceneComposition(str(source), str(source), (), hashlib.sha256(source.read_bytes()).hexdigest(), True)
    sensor = '/Render/Camera'
    request = SimpleNamespace(semantic_output_path=str(args.output / 'frame'), camera_prim_path='/World/Camera', dlss_enabled=False)
    spec = SimpleNamespace(ovrtx_scene_composition=composition, sensor_paths=(sensor,), width=160, height=120, camera_prim_path='/World/Camera', render_var='HdrColor', stage_metadata={'upAxis': 'Y', 'metersPerUnit': 1, 'defaultPrim': 'World'})
    client = PublicSdkRenderClient(request)
    report = dict(status='running', frames=[])
    started = time.monotonic()
    try:
        identity = client.start_session(spec)
        pid, epoch = (client.runtime.pid, client.runtime.epoch)
        results = [client.render_result(identity, selected_sensor_paths=(sensor,), render_var='HdrColor', additional_samples=2)]
        ticket = client.begin_render_sample(identity, selected_sensor_paths=(sensor,), render_var='HdrColor')
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            ticket.wait(10)
            result = client.poll_render_sample(ticket)
            if result is not None:
                results.append(result)
                break
        else:
            raise TimeoutError('Semantic asynchronous sample did not complete')
        labels = [dict(id='class_a', name='class_a', source_ids=['Left', 'Occluder']), dict(id='class_b', name='class_b', source_ids=['Right'])]
        for index, result in enumerate(results):
            capture = result.semantic_capture
            assert capture is not None
            assert len(result.linear_rgba16f) == 160 * 120 * 8
            assert len(capture.ids) == 160 * 120 * 4
            mapping = decode_id_map(capture.id_map)
            assert {'class: class_a;', 'class: class_b;'}.issubset(set(mapping.values()))
            receipt = save_capture(args.output / f'frame.{index:04d}', result, labels, camera='/World/Camera', frame=index)
            report['frames'].append(dict(verified=receipt['verified'], camera=capture.camera_prim_path, width=capture.width, height=capture.height, simulation_time_ns=capture.simulation_time_ns, frame_start_time=capture.frame_start_time, frame_end_time=capture.frame_end_time, ids_sha256=hashlib.sha256(capture.ids).hexdigest(), hdr_sha256=hashlib.sha256(result.linear_rgba16f).hexdigest(), native_id_map=mapping, epoch=capture.epoch, stage_ordinal=capture.ordinal))
        assert results[1].simulation_time_ns > results[0].simulation_time_ns
        assert (client.runtime.pid, client.runtime.epoch) == (pid, epoch)
        assert not {'ovrtx', 'ovstage', 'ovstream', 'ovphysx', 'warp'} & sys.modules.keys()
        report.update(status='passed', same_worker=True, pid=pid, epoch=epoch)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        client.shutdown()
        report['runtime'] = client.runtime.diagnostics()
        report['elapsed_seconds'] = time.monotonic() - started
        (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
if __name__ == '__main__':
    main()
