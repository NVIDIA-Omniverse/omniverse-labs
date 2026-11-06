# SPDX-License-Identifier: Apache-2.0
"""Explicit native DLSS A/B using a synthetic scene; caller owns outputs.

Tests initialized global Token presets and an owned XR source product without
an HMD. CPU render-call timings are not GPU timing or headset latency.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
root = Path(__file__).parents[1]
sys.path[:0] = [str(root/'runtime'), str(root/'addon'), str(root/'scripts')]
from public_sdk_backend import RendererBackend
from xr_product_source import XrProductSource
from xr_worker import XrWorker
from ovrtx_blender_example.xr_settings import XrSettings
from public_sdk_light_fixture import _generated_light_probe_fixture
import numpy as np
import warp as wp
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
fixture = _generated_light_probe_fixture(args.output/'synthetic.usda', 960, 600)
path = Path(fixture['usd_path'])
s = path.read_text().replace('float inputs:intensity = 50000','float inputs:intensity = 500')
s = s.replace('def Cube "LitCube"', 'def Mesh "LitCube"').replace('double size = 1.6', 'point3f[] points = [(-1,0,0),(1,0,0),(1,0,2),(-1,0,2)]\nint[] faceVertexCounts = [3,3]\nint[] faceVertexIndices = [0,2,1,0,3,2]\nuniform token subdivisionScheme = "none"\nnormal3f[] normals = [(0,-1,0),(0,-1,0),(0,-1,0),(0,-1,0)]')
s = s.replace('float inputs:angle = 0.53', 'float inputs:angle = 0.53\nquatd xformOp:orient = (0.7071068,0.7071068,0,0)\nuniform token[] xformOpOrder = ["xformOp:orient"]')
path.write_text(s)
images, reports = {}, {}
for quality in ('quality', 'performance'):
    owner = RendererBackend(args.output/quality)
    source = None
    try:
        owner.dispatch('load', dict(scene_path=str(path), sensor_paths=[fixture['render_product_path']], width=960,height=600, settings={'/rtx/post/dlss/execMode': quality}))
        worker = XrWorker.__new__(XrWorker)
        worker.owner = owner
        worker.settings_effective = XrSettings(dlss_quality=quality)
        source = XrProductSource(worker)
        selected = source.path
        def frame():
            products = owner.renderer.step({selected}, delta_time=1/60, ordinal=owner.ordinal)
            result = products[selected].frames[-1]
            buffer = next(v for k,v in result.render_vars.items() if k.rsplit('/',1)[-1]=='LdrColor')
            mapped = buffer.map(device=owner.rtx.Device.CUDA)
            try:
                return wp.from_dlpack(mapped).numpy()[..., :3].copy()
            finally:
                mapped.unmap()
        for _ in range(24): frame()
        elapsed = []
        for _ in range(24):
            start=time.perf_counter()
            image=frame()
            elapsed.append(1000*(time.perf_counter()-start))
        control=frame()
        observed=worker.eye_values([dict(prim_path=selected,attribute_name='omni:rtx:post:dlss:execMode',value_type='Token',value=quality)])
        assert observed[0]['value'] == quality
        assert image.shape == (600,960,3) and float(image.mean())>1
        images[quality]=image
        reports[quality]={'observed':quality,'mean_pixel':float(image.mean()), 'no_edit_mae':float(np.abs(image.astype(float)-control.astype(float)).mean()),'cpu_render_and_readback_ms':float(np.mean(elapsed))}
    finally:
        if source: source.close()
        owner.dispatch('unload',{})
mae=float(np.abs(images['quality'].astype(float)-images['performance'].astype(float)).mean())
proof={'presets':reports, 'quality_performance_pixel_mae':mae,'changed_above_noise':mae>max(.1,3*max(row['no_edit_mae'] for row in reports.values())), 'scope':'Synthetic initialized native RenderProducts; no connected OVXR context; CPU timings include readback; no house FPS or latency claim'}
(args.output/'proof.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof))
