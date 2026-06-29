# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tests using ovrtx-test-base.usda."""

from pathlib import Path

import numpy as np
import ovrtx
from PIL import Image

TEST_BASE_PATH = str((Path(__file__).parent / "../data/ovrtx-test-base.usda").resolve())
LOGO_ANIMATED_PATH = str(
    (Path(__file__).parent / "../data/ovrtx-test-base-logo-animated.usda").resolve()
)


def test_base(renderer, output_dir):
    """Render LdrColor from /World/Camera using the test base scene."""
    renderer.open_usd(TEST_BASE_PATH)

    # Warm up
    for _ in range(5):
        renderer.step(render_products={"/Render/Camera"}, delta_time=1.0 / 60)

    # Render one frame
    products = renderer.step(
        render_products={"/Render/Camera"},
        delta_time=1.0 / 60,
    )

    # Save the LdrColor output
    for product in products.values():
        for frame in product.frames:
            var = frame.render_vars["LdrColor"].map(device=ovrtx.Device.CPU)
            pixels = np.from_dlpack(var)
            assert pixels.dtype == np.uint8
            assert pixels.shape[2] == 4  # RGBA
            img = Image.fromarray(pixels)
            img.save(output_dir / "base.Camera.LdrColor.0001.png")


def test_bind_material(renderer, output_dir):
    """Bind the glass material to the logo and render."""
    renderer.open_usd(TEST_BASE_PATH)

    # [snippet:doc-bind-material]
    # Bind /World/Looks/srf_glass to /World/logo
    renderer.write_array_attribute(
        prim_paths=["/World/logo/logo/logo"],
        attribute_name="material:binding",
        tensors=[["/World/Looks/srf_glass"]],
    )
    # [/snippet:doc-bind-material]

    # [snippet:doc-warmup]
    # Warm up - step enough frames for texture streaming to finish loading
    # high-res mips and for path tracing to converge to a good quality image.
    WARMUP_FRAMES = 40
    for _ in range(WARMUP_FRAMES):
        renderer.step(render_products={"/Render/Camera"}, delta_time=1.0 / 60)
    # [/snippet:doc-warmup]

    # Render one frame
    products = renderer.step(
        render_products={"/Render/Camera"},
        delta_time=1.0 / 60,
    )

    # Save the LdrColor output
    for product in products.values():
        for frame in product.frames:
            var = frame.render_vars["LdrColor"].map(device=ovrtx.Device.CPU)
            pixels = np.from_dlpack(var)
            assert pixels.dtype == np.uint8
            assert pixels.shape[2] == 4  # RGBA
            img = Image.fromarray(pixels)
            img.save(output_dir / "bind_material.Camera.LdrColor.0001.png")


def test_settings_rtpt_maxBounces(renderer, output_dir):
    """Test omni:rtx:rtpt:maxBounces render setting at different values."""
    renderer.open_usd(TEST_BASE_PATH)

    for max_bounces in [2, 3, 23]:
        # [snippet:doc-set-render-setting]
        # Set a render setting on the RenderProduct prim
        renderer.write_attribute(
            prim_paths=["/Render/Camera"],
            attribute_name="omni:rtx:rtpt:maxBounces",
            tensor=np.array([max_bounces], dtype=np.int32),
        )
        # [/snippet:doc-set-render-setting]

        # Reset and warm up so the setting takes effect
        renderer.reset()
        for _ in range(40):
            renderer.step(render_products={"/Render/Camera"}, delta_time=1.0 / 60)

        # Render one frame
        products = renderer.step(
            render_products={"/Render/Camera"},
            delta_time=1.0 / 60,
        )

        # Save the LdrColor output
        for product in products.values():
            for frame in product.frames:
                var = frame.render_vars["LdrColor"].map(device=ovrtx.Device.CPU)
                pixels = np.from_dlpack(var)
                assert pixels.dtype == np.uint8
                assert pixels.shape[2] == 4  # RGBA
                img = Image.fromarray(pixels)
                img.save(
                    output_dir
                    / f"settings_rtpt_maxBounces.Camera.LdrColor.maxBounces-{max_bounces}.0001.png"
                )


def test_update_from_usd_time_async(renderer):
    """Asynchronously evaluate a time-sampled attribute at two distinct times."""
    # The animated sublayer authors xformOp:translate time samples on /World/logo
    # spanning X=-100 at timecode 0 to X=+100 at timecode 60, with
    # timeCodesPerSecond=60 (so timecode 60 corresponds to 1 second).
    renderer.open_usd(LOGO_ANIMATED_PATH)
    renderer.reset()

    def _translate_x_at(t_seconds: float) -> float:
        # [snippet:doc-update-from-usd-time-async]
        # update_from_usd_time_async takes seconds, not USD timecodes — the
        # runtime converts via the stage's timeCodesPerSecond metadata.
        # .wait() resolves once time-sampled attributes have been re-evaluated.
        renderer.update_from_usd_time_async(t_seconds).wait()
        # [/snippet:doc-update-from-usd-time-async]

        # Drive the render pipeline once so time-sampled attribute writes are
        # consumed before we read the composed transform back.
        renderer.step(render_products={"/Render/Camera"}, delta_time=1.0)

        # Read the composed local matrix and pull out the translate row (USD
        # uses row-major with the translation in row 3).
        tensor = renderer.read_attribute(
            attribute_name="omni:xform",
            prim_paths=["/World/logo"],
        )
        matrix = np.from_dlpack(tensor).reshape(4, 4)
        return float(matrix[3, 0])

    # 0s → timecode 0 → X=-100; 1s → timecode 60 → X=+100.
    x_at_start = _translate_x_at(0.0)
    x_at_end = _translate_x_at(1.0)

    assert abs(x_at_end - x_at_start) > 1.0, (
        f"expected time-sampled translate to change across time, "
        f"got x(0s)={x_at_start}, x(1s)={x_at_end}"
    )


def test_operation_status_while_loading(renderer):
    """Poll ``Operation.query_status()`` on a long-running USD reference op."""
    # Start from a clean stage so the reference add actually does work.
    renderer.reset_stage()

    # [snippet:doc-operation-status]
    # Enqueue a long-running op and poll its progress without waiting.
    op = renderer.add_usd_reference_async(TEST_BASE_PATH, "/LoadedBase")
    saw_counter = False
    while True:
        status = op.query_status()
        assert status.state in (ovrtx.EventStatus.PENDING, ovrtx.EventStatus.COMPLETED)
        assert isinstance(status.counters, list)
        for counter in status.counters:
            assert isinstance(counter, ovrtx.OperationCounter)
            saw_counter = True
        if status.state != ovrtx.EventStatus.PENDING:
            break
    op.wait()
    # [/snippet:doc-operation-status]

    # The scene has shaders/textures/materials to load, so we expect at least
    # one counter to have surfaced while the op was running. Use a soft check
    # so the test doesn't flake if a host is fast enough to skip the PENDING
    # window entirely.
    if not saw_counter:
        print("note: USD reference load completed before any counter was observed")


def test_settings_rtpt_maxSpecularAndTransmissionBounces(renderer, output_dir):
    """Test omni:rtx:rtpt:maxSpecularAndTransmissionBounces with glass material."""
    renderer.open_usd(TEST_BASE_PATH)

    # Bind glass material to the logo so specular/transmission bounces are visible
    renderer.write_array_attribute(
        prim_paths=["/World/logo/logo/logo"],
        attribute_name="material:binding",
        tensors=[["/World/Looks/srf_glass"]],
    )

    for bounces in [2, 3, 23]:
        # Set maxSpecularAndTransmissionBounces on the RenderProduct prim
        renderer.write_attribute(
            prim_paths=["/Render/Camera"],
            attribute_name="omni:rtx:rtpt:maxSpecularAndTransmissionBounces",
            tensor=np.array([bounces], dtype=np.int32),
        )

        # Reset and warm up so the setting takes effect
        renderer.reset()
        for _ in range(40):
            renderer.step(render_products={"/Render/Camera"}, delta_time=1.0 / 60)

        # Render one frame
        products = renderer.step(
            render_products={"/Render/Camera"},
            delta_time=1.0 / 60,
        )

        # Save the LdrColor output
        for product in products.values():
            for frame in product.frames:
                var = frame.render_vars["LdrColor"].map(device=ovrtx.Device.CPU)
                pixels = np.from_dlpack(var)
                assert pixels.dtype == np.uint8
                assert pixels.shape[2] == 4  # RGBA
                img = Image.fromarray(pixels)
                img.save(
                    output_dir
                    / f"settings_rtpt_maxSpecularAndTransmissionBounces.Camera.LdrColor.maxSpecularAndTransmissionBounces-{bounces}.0001.png"
                )
