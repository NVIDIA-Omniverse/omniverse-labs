# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the literal actor samples, not merely nonempty native payloads."""
import sys

import moderngl
import numpy as np
from PIL import Image
import pytest

from ovd_ui.renderer import ViewportRenderer
from ovd_ui.overlays import OverlayRenderer
from ovd_ui.render_settings import DisplayMode
from test_selected_deformables import selected_capture, open_frame


class ImageSink:
    def destroy(self):
        pass

    def set_image_data(self, *_):
        pass


@pytest.fixture
def render_context(request):
    context = moderngl.create_standalone_context(require=330, **({} if sys.platform == "win32" else {"backend": "egl"}))
    request.node.user_properties.extend((key, context.info[key]) for key in ("GL_RENDERER", "GL_VERSION"))
    yield context
    assert context.error == "GL_NO_ERROR"
    context.release()


def pixel(matrix, point, width=640, height=384):
    clip = matrix @ np.array([*point, 1.])
    ndc = clip[:2] / clip[3]
    # Both production shaders flip clip Y for the OvUI image convention.
    return int((ndc[0] + 1) * width / 2), int((1 - ndc[1]) * height / 2)


def recorded_owners(session):
    # Writer handles are literal fixture identities. DOM instance allocation
    # order differs across platforms and must not define the picking oracle.
    page = session.evidence_metadata(0, 512)
    assert page.next_offset is None
    identities = {row.api_handle: row for row in page.rows}
    assert [identities[handle].class_name for handle in (10, 20, 21)] == [
        "PxDeformableVolume", "PxDeformableSurface", "PxDeformableSurface"]
    return tuple(identities[handle].instance_id for handle in (10, 20, 21))


def test_selected_mesh_motion_reaches_raster_and_picking(selected_capture, tmp_path, render_context):
    session = open_frame(selected_capture(), tmp_path / "index")
    renderer = ViewportRenderer(render_context, ImageSink())
    renderer.camera.look_at([0, 0, 18], [0, 0, 0])
    volume, left, right = recorded_owners(session)
    expected = [(-5.8, 2.2, 0), (2.2, 2.2, 0), (-5.8, -2.8, .2), (2.2, -2.8, .2)]
    owners = [left, right, volume, volume]
    try:
        images = []
        for frame in (1, 2):
            session.seek(session.scenes()[0].id, frame)
            renderer.set_snapshot(session.geometry_snapshot())
            renderer.render(640, 384, grid_contrast=0)
            matrix = renderer._view_projection()
            rgba = renderer.framebuffer.read(components=4)
            images.append(rgba)
            Image.frombytes("RGBA", (640, 384), rgba).save(tmp_path / f"selected-geometry-{frame}.png")
            shifts = [3, -3, 3, 3] if frame == 2 else [0] * 4
            for position, dx, owner in zip(expected, shifts, owners):
                moved = [position[0] + dx, *position[1:]]
                assert renderer.pick(*pixel(matrix, moved)) == owner
                if frame == 2:
                    assert renderer.pick(*pixel(matrix, position)) is None
        assert images[0] != images[1]
    finally:
        renderer.close()


def test_selected_velocity_arrows_rasterize_at_actor_sample_locations(selected_capture, tmp_path, render_context):
    session = open_frame(selected_capture(), tmp_path / "index")
    viewport = ViewportRenderer(render_context, ImageSink())
    viewport.camera.look_at([0, 0, 18], [0, 0, 0])
    viewport.render(640, 384, grid_contrast=0)
    overlay = OverlayRenderer(render_context)
    framebuffer = render_context.simple_framebuffer((640, 384), components=4)
    matrix = viewport._view_projection()
    volume, left, right = recorded_owners(session)
    try:
        images = []
        for frame in (1, 2):
            session.seek(session.scenes()[0].id, frame)
            overlay.set_snapshot(session.overlay_snapshot(4))
            overlay.set_mode(4, DisplayMode.ALL)
            with render_context.scope(framebuffer=framebuffer):
                framebuffer.clear(0, 0, 0, 0)
                render_context.viewport = (0, 0, 640, 384)
                overlay.draw(matrix, viewport.eye(), viewport.distance, session.scenes()[0].instance_id)
            rgba = framebuffer.read(components=4)
            images.append(rgba)
            Image.frombytes("RGBA", (640, 384), rgba).save(tmp_path / f"selected-velocity-{frame}.png")
            pixels = np.frombuffer(rgba, dtype=np.uint8).reshape(384, 640, 4)
            dx = 3 * (frame - 1)
            # Midpoints of literal arrows: surface +Y/-Y, volume simulation +Y.
            for point in [(-6 + dx, 2.2, 0), (2 - dx, 1.85, 0), (2 + dx, -2.9, 0)]:
                x, y = pixel(matrix, point)
                assert np.any(pixels[y-2:y+3, x-2:x+3, :3]), point
            # Collision vertices have no velocity arrows.
            x, y = pixel(matrix, (-6 + dx, -2.9, 0))
            assert not np.any(pixels[y-2:y+3, x-2:x+3, :3])
            with render_context.scope(framebuffer=framebuffer):
                framebuffer.clear(0, 0, 0, 0)
                overlay.draw(matrix, viewport.eye(), viewport.distance, session.scenes()[0].instance_id, picking=True)
            encoded = np.frombuffer(framebuffer.read(components=4), dtype="<u4").reshape(384, 640)
            for point, owner in [((-6 + dx, 2.2, 0), left), ((2 - dx, 1.85, 0), right), ((2 + dx, -2.9, 0), volume)]:
                x, y = pixel(matrix, point)
                assert owner + 1 in encoded[y-2:y+3, x-2:x+3]
        assert images[0] != images[1]
    finally:
        overlay.close()
        framebuffer.release()
        viewport.close()
