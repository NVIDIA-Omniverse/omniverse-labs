# SPDX-License-Identifier: Apache-2.0
"""Isolated OVStage/addon CPU probe; excludes GPU rendering and headset timing."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / 'runtime'))
os.environ.setdefault('OMNICLIENT_HUB_MODE', 'disabled')
import numpy as np
import ovstage
from native_pose_fanout import NativePoseFanout
from ovstage_scene_updates import SceneGenerationPopulation
from ovstage_pose_layers import PoseLayerPopulation
from public_sdk_backend import RendererBackend

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

def matrix(x):
    value = np.eye(4)
    value[3, 0] = x
    return value

def rows(path, x):
    return [dict(prim_path=path, attribute_name='omni:xform', value_type='Matrix4d', value=matrix(x).tolist()),
            dict(prim_path=path, attribute_name='omni:resetXformStack', value_type='Bool', value=True)]

reports = []
for count in (10, 1000, 10000):
    directory = args.output / str(count)
    directory.mkdir(exist_ok=True)
    source = directory / 'source.usda'
    source.write_text('''#usda 1.0
def Camera "Camera" {}
def Xform "Moved" {
    def Mesh "Mesh" {
        point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0,1,2]
        def GeomSubset "Faces" {
            uniform token elementType = "face"
            int[] indices = [0]
        }
    }
}
def Scope "Unrelated" {
''' + '\n'.join('def Xform "Object' + str(n) + '" {}' for n in range(count)) + '\n}\n')
    with ovstage.Stage('navigation-scaling') as stage:
        owner = SceneGenerationPopulation(stage)
        owner.initialize(source)
        stage.advance_write_floor(owner.ordinal).wait()
        backend = RendererBackend.__new__(RendererBackend)
        backend.stage, backend.ovstage, backend.np = stage, ovstage, np
        backend.native_stage_id, backend.ordinal = str(count), owner.ordinal
        backend.population_owner = owner
        backend.pose_population = PoseLayerPopulation(owner, directory)
        backend.native_pose_fanout = NativePoseFanout(stage, ovstage)
        backend.desired_values = {}
        backend.simulation_time_ns = 0
        backend.sample_count = 16
        calls = {'renderer_updates': 0, 'renderer_resets': 0}
        def updated(ordinal):
            calls['renderer_updates'] += 1
        def reset(**kwargs):
            calls['renderer_resets'] += 1
        # Exercise the actual backend and native writes; exclude RTX/GPU work
        # so scene rendering cost cannot hide addon overhead in these timings.
        backend.renderer = SimpleNamespace(update_from_stage=updated, reset=reset)
        times = []
        for frame in range(120):
            start = time.perf_counter()
            result = backend.update(rows('/Camera', frame * .01))
            elapsed = (time.perf_counter() - start) * 1000
            assert result['native_pose_route'] == 'native_inputs'
            if frame >= 20:
                times.append(elapsed)
        moved = backend.update(rows('/Moved', 2))
        assert moved['native_pose_route'] == 'native_inputs'
        assert moved['native_pose_fanout_rows'] == 2
        before = dict(calls)
        repeated = backend.update(rows('/Moved', 2))
        assert repeated['native_pose_route'] == 'unchanged' and calls == before
        with ovstage.PathDictionary(stage) as paths:
            worlds = backend.native_pose_fanout._worlds(
                {'/Camera', '/Moved', '/Moved/Mesh'}, backend.ordinal, paths)
        np.testing.assert_allclose(worlds['/Camera'], matrix(1.19))
        np.testing.assert_allclose(worlds['/Moved'], matrix(2))
        np.testing.assert_allclose(worlds['/Moved/Mesh'], matrix(2))
        assert calls['renderer_resets'] == 0 and owner.path == source.resolve()
        assert not list(directory.glob('native-poses*.usda'))
        reports.append(dict(unrelated_objects=count, samples=len(times),
            camera_cpu_p50_ms=float(np.median(times)), camera_cpu_p95_ms=float(np.percentile(times, 95)),
            cached_identities=len(backend.native_pose_fanout.identities),
            hierarchy_entries=len(backend.native_pose_fanout.children),
            population_replacements=0, **calls))
        print(json.dumps(reports[-1]), flush=True)
(args.output / 'result.json').write_text(json.dumps({'scope': __doc__, 'cases': reports}, indent=2))
print('NAVIGATION_NATIVE_SCALING_PASS', flush=True)
