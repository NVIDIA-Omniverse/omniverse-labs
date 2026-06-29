# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tests for camera_sensors.rst Python code examples."""

from pathlib import Path

import numpy as np
import ovrtx
import pytest
from PIL import Image

SCENE_PATH = str((Path(__file__).parent / "../../../tests/data/simple_camera.usda").resolve())

USDA = f"""#usda 1.0
(
    subLayers = [
        @{SCENE_PATH}@
    ]
)

def "Render" {{
    def RenderProduct "Camera" {{
        int2 resolution = (1920, 1080)
        rel camera = </Camera0>
        rel orderedVars = [<LdrColor>, <HdrColor>]

        def RenderVar "LdrColor" {{
            string sourceName = "LdrColor"
        }}

        def RenderVar "HdrColor" {{
            string sourceName = "HdrColor"
        }}
    }}
}}
"""


def test_step_and_map_camera_outputs(output_dir):
    """Test stepping and mapping both LdrColor and HdrColor outputs (camera_sensors.rst)."""
    renderer = ovrtx.Renderer()
    renderer.open_usd_from_string(USDA)

    # Warm up
    for _ in range(5):
        renderer.step(render_products={"/Render/Camera"}, delta_time=1.0 / 60)

    # [snippet:doc-step-and-map-camera-outputs]
    products = renderer.step(
        render_products={"/Render/Camera"},
        delta_time=1.0 / 60,
    )

    for product_name, product in products.items():
        for frame in product.frames:
            # LdrColor: uint8 sRGB image
            var = frame.render_vars["LdrColor"].map(device=ovrtx.Device.CPU)
            ldr_pixels = np.from_dlpack(var)  # shape: (H, W, 4), dtype: uint8
            assert ldr_pixels.shape == (1080, 1920, 4)
            assert ldr_pixels.dtype == np.uint8
            Image.fromarray(ldr_pixels).save(output_dir / "test_camera_sensors.LdrColor.png")

            # HdrColor: float16 linear image
            var = frame.render_vars["HdrColor"].map(device=ovrtx.Device.CPU)
            hdr_pixels = np.from_dlpack(var)  # shape: (H, W, 4), dtype: float16
            assert hdr_pixels.shape == (1080, 1920, 4)
            assert hdr_pixels.dtype == np.float16
    # [/snippet:doc-step-and-map-camera-outputs]

    del renderer


def test_step_async_returns_operation(output_dir):
    """``step_async`` returns an ``Operation[PendingFetch[RenderProductSetOutputs]]``.

    Replaces the 0.2.0 ``RendererResult`` return type — see CHANGELOG 0.3.0.
    """
    renderer = ovrtx.Renderer()
    renderer.open_usd_from_string(USDA)

    # [snippet:doc-step-async]
    # step_async() returns an Operation. wait() resolves to a PendingFetch,
    # whose fetch() produces the RenderProductSetOutputs that step() would
    # have returned synchronously.
    op = renderer.step_async(
        render_products={"/Render/Camera"},
        delta_time=1.0 / 60,
    )
    pending = op.wait()
    products = pending.fetch()
    # [/snippet:doc-step-async]

    assert isinstance(op, ovrtx.Operation)
    assert isinstance(pending, ovrtx.PendingFetch)
    assert isinstance(products, ovrtx.RenderProductSetOutputs)

    del renderer


def test_map_camera_output_cuda():
    """Map a render output as CUDA memory."""
    renderer = ovrtx.Renderer()
    renderer.open_usd_from_string(USDA)
    products = renderer.step(render_products={"/Render/Camera"}, delta_time=1.0 / 60)

    # [snippet:doc-map-render-output-cuda]
    mapping = products["/Render/Camera"].frames[0].render_vars["LdrColor"].map(
        device=ovrtx.Device.CUDA
    )
    try:
        tensor = mapping.tensor
        assert tensor.__dlpack_device__()[0] == 2  # DLPack kDLCUDA
    finally:
        mapping.unmap()
    # [/snippet:doc-map-render-output-cuda]

    del renderer


def test_renderer_result_no_longer_exported():
    """The 0.2.0 ``RendererResult`` export was removed; importing it must fail."""
    with pytest.raises(ImportError):
        from ovrtx import RendererResult  # noqa: F401
