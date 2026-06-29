#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test the public OVRTX facade against the Metal backend."""

from __future__ import annotations

try:
    import numpy as np
except ModuleNotFoundError as exc:
    print(f"SKIP: Python dependency missing: {exc.name}")
    raise SystemExit(77) from exc

import ovrtx


PRODUCT = "/Render/Products/Product"

USD = """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)

def Xform "World"
{
    def Mesh "Triangle"
    {
        point3f[] points = [(-1, -1, 0), (1, -1, 0), (0, 1, 0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        normal3f[] primvars:normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1)] (
            interpolation = "vertex"
        )
        color3f[] primvars:displayColor = [(0.9, 0.25, 0.1)] (
            interpolation = "constant"
        )
        uniform token subdivisionScheme = "none"
    }

    def Camera "Camera"
    {
        matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 4, 1) )
        float focalLength = 35
        float horizontalAperture = 20.955
        float verticalAperture = 15.2908
        float horizontalApertureOffset = 0.75
        float2 clippingRange = (0.1, 100)
    }
}

def Scope "Render"
{
    def Scope "Products"
    {
        def RenderProduct "Product"
        {
            int2 resolution = (96, 64)
            rel camera = </World/Camera>
            rel orderedVars = [<../Vars/Color>]
        }
    }

    def Scope "Vars"
    {
        def RenderVar "Color"
        {
            string sourceName = "LdrColor"
        }
    }
}
"""


def main() -> int:
    renderer = ovrtx.Renderer(ovrtx.RendererConfig())
    renderer.open_usd_from_string(USD)
    outputs = renderer.step(render_products={PRODUCT}, delta_time=1.0 / 60.0)
    frame = outputs[PRODUCT].frames[0]

    with frame.render_vars["LdrColor"].map(ovrtx.Device.CPU) as mapped:
        assert mapped.__dlpack_device__() == (1, 0)
        pixels = np.from_dlpack(mapped)

    assert pixels.shape == (64, 96, 4), pixels.shape
    assert pixels.dtype == np.uint8, pixels.dtype
    assert np.any(pixels[..., :3]), "OVRTX Metal render produced an empty color image"

    try:
        frame.render_vars["LdrColor"].map(ovrtx.Device.CUDA)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Metal backend should reject CUDA render-var mapping")

    nu = renderer._nu
    assert nu is not None
    assert nu.mesh_count >= 1
    assert nu.get_mesh_name(0).endswith("/World/Triangle")

    moved = np.eye(4, dtype=np.float32)
    moved[0, 3] = 0.5
    renderer.write_attribute(
        ["/World/Triangle"],
        "xformOp:transform",
        moved.reshape(1, 4, 4),
        semantic=ovrtx.Semantic.XFORM_MAT4x4,
        prim_mode=ovrtx.PrimMode.CREATE_NEW,
    )
    assert not renderer._native_stage_dirty
    got = nu.get_mesh_transform(0)
    assert np.allclose(got, moved, atol=1.0e-5), got

    renderer.write_attribute(
        ["/World/Triangle"],
        "primvars:displayColor",
        np.asarray([[[0.1, 0.8, 0.2]]], dtype=np.float32),
    )
    assert not renderer._native_stage_dirty

    renderer.write_attribute(
        ["/World/Triangle"],
        "visibility",
        np.asarray(["invisible"]),
        prim_mode=ovrtx.PrimMode.CREATE_NEW,
    )
    assert not renderer._native_stage_dirty

    outputs.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
