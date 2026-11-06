# SPDX-License-Identifier: Apache-2.0
"""Explicit installed public SDK initial/live light probe; caller owns outputs."""
import argparse
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "runtime"))
sys.path.insert(0, str(root / "scripts"))
from public_sdk_backend import RendererBackend
from public_sdk_light_fixture import _generated_light_probe_fixture
import warp as wp

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
fixture = _generated_light_probe_fixture(args.output / "lights.usda", 320, 200)
path = Path(fixture["usd_path"])
s = path.read_text().replace("float inputs:intensity = 50000", "float inputs:intensity = 500\nfloat inputs:exposure = 0\nfloat inputs:diffuse = 1\nfloat inputs:specular = 1")
s = s.replace('def Cube "LitCube"', 'def Mesh "LitCube"').replace('double size = 1.6', 'point3f[] points = [(-1,0,0), (1,0,0), (1,0,2), (-1,0,2)]\nint[] faceVertexCounts = [3,3]\nint[] faceVertexIndices = [0,2,1,0,3,2]\nuniform token subdivisionScheme = "none"\nnormal3f[] normals = [(0,-1,0), (0,-1,0), (0,-1,0), (0,-1,0)]')
s = s.replace('float inputs:angle = 0.53', 'float inputs:angle = 0.53\nquatd xformOp:orient = (0.7071068, 0.7071068, 0, 0)\nuniform token[] xformOpOrder = ["xformOp:orient"]')
path.write_text(s)
owner = RendererBackend(args.output)
try:
    owner.dispatch("load", dict(scene_path=str(path), sensor_paths=[fixture["render_product_path"]], width=320, height=200, settings={}))
    stage_id = owner.native_stage_id
    def pixels():
        for _ in range(32):
            products = owner.renderer.step(set(owner.sensors), delta_time=1/60, ordinal=owner.ordinal)
        frame = products[owner.sensors[0]].frames[-1]
        buffer = next(v for k,v in frame.render_vars.items() if k.rsplit("/",1)[-1] == "LdrColor")
        mapped = buffer.map(device=owner.rtx.Device.CUDA)
        try:
            return wp.from_dlpack(mapped).numpy()[..., :3].copy()
        finally:
            mapped.unmap()
    def write(name, value):
        owner.update([dict(prim_path="/World/KeyLight", attribute_name=name, value_type="Float", value=value)])
    baseline = pixels()
    write("inputs:exposure", 2.0)
    exposure = pixels()
    write("inputs:exposure", 0.0)
    write("inputs:intensity", 2000.0)
    baked = pixels()
    write("inputs:diffuse", 0.0)
    diffuse_off = pixels()
    import numpy as np
    result = dict(initial_mean=float(baseline.mean()), exposure_mean=float(exposure.mean()), baked_mean=float(baked.mean()), diffuse_off_mean=float(diffuse_off.mean()), exposure_baked_mean_abs_delta=float(np.abs(exposure.astype(float)-baked.astype(float)).mean()), same_session=stage_id==owner.native_stage_id)
    result["exposure_honored"] = bool(np.abs(exposure.astype(float)-baseline.astype(float)).mean() > .001)
    result["diffuse_honored"] = bool(np.abs(diffuse_off.astype(float)-baked.astype(float)).mean() > .001)
    assert result["same_session"]
    assert result["initial_mean"] > 1.0
    assert float(np.abs(exposure.astype(float)-baseline.astype(float)).mean()) > .5
    assert result["exposure_baked_mean_abs_delta"] < .1
    assert float(np.abs(diffuse_off.astype(float)-baked.astype(float)).mean()) > .5
    write("inputs:diffuse", 1.0)
    owner.update([dict(prim_path="/World/KeyLight", attribute_name="visibility", value_type="Token", value="invisible")])
    hidden = pixels()
    owner.update([dict(prim_path="/World/KeyLight", attribute_name="visibility", value_type="Token", value="inherited")])
    restored = pixels()
    result["visibility_hide_mean"] = float(hidden.mean())
    result["visibility_restore_mean"] = float(restored.mean())
    assert float(np.abs(hidden.astype(float)-restored.astype(float)).mean()) > .5
    assert float(np.abs(restored.astype(float)-baked.astype(float)).mean()) < .1
    # Visibility change is exercised independently of the diffuse-off control.
    write("inputs:diffuse", 1.0)
    write("inputs:specular", 0.0)
    result["specular_off_mean"] = float(pixels().mean())
    write("inputs:specular", 1.0)
    result["specular_reset_mean"] = float(pixels().mean())
    assert abs(result["specular_reset_mean"]-result["specular_off_mean"]) > .5
    (args.output / "proof.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
finally:
    owner.dispatch("unload", {})
