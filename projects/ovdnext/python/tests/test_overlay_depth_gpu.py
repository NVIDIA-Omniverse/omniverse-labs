# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real OpenGL proof of OvUI's deliberate on-top debug-arrow behavior."""
import os
import sys
from types import SimpleNamespace

import moderngl
import numpy as np
import pytest

from ovd_ui.render_settings import RenderingSettings
from ovd_ui.renderer import ViewportRenderer
from ovd_ui.palette import ColorPalette, DEFAULT_PALETTE_PATH


class TextureConsumer:
    def destroy(self):
        pass

    def set_image_data(self, *_):
        pass


def box_snapshot(identity=100, extent=1.):
    return SimpleNamespace(
        object_ids=np.array([identity], dtype="i4"),
        shape_ids=np.array([identity + 100], dtype="i4"),
        geometry_ids=np.array([identity + 200], dtype="i4"),
        scene_ids=np.array([1], dtype="i4"),
        kinds=np.array([2], dtype="u1"),
        resource_ids=np.array([0], dtype="i4"),
        transforms=np.eye(4, dtype="f4")[None],
        dimensions=np.full((1, 3), extent, dtype="f4"),
        bounds=np.array([[[-extent] * 3, [extent] * 3]], dtype="f4"),
        sleeping=np.array([False]), meshes=[], diagnostics=[])


def overlay_snapshot(category):
    # Arrow, contact normal, contact point, and a query triangle are all
    # enclosed by the opaque unit box, with separate screen-space locations.
    positions=np.array([
        [[-.6, .45, 0], [.6, .45, 0], [0, 0, 0]],
        [[-.6, -.45, 0], [.6, -.45, 0], [0, 0, 0]],
        [[0, -.7, 0], [0, 0, 0], [0, 0, 0]],
        [[-.2, -.2, 0], [.2, -.2, 0], [0, .2, 0]],
    ], dtype="f4")
    colors=np.repeat(np.array([
        [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 1, 1],
    ], dtype="f4")[:, None], 3, axis=1)
    query=box_snapshot(identity=20, extent=.3)
    query.transforms[:, 0, 3]=.55
    query.bounds[:, :, 0]+=.55
    return SimpleNamespace(
        ready=True, positions=positions, colors=colors,
        kinds=np.array([4, 2, 1, 3], dtype="u1"),
        categories=np.array([category, 32, 32, 64], dtype="u4"),
        object_ids=np.array([10, 11, 12, 13], dtype="i4"),
        related_ids=np.zeros((4, 2), dtype="i4"),
        scene_ids=np.ones(4, dtype="i4"),
        query_geometry=query)


@pytest.fixture
def renderer():
    options={} if sys.platform == "win32" else {"backend": "egl"}
    context=moderngl.create_standalone_context(require=330, **options)
    result=ViewportRenderer(context, TextureConsumer())
    try:
        if os.environ.get("GALLIUM_DRIVER") == "llvmpipe":
            assert "llvmpipe" in context.info["GL_RENDERER"]
        result.set_snapshot(box_snapshot())
        result.camera.position=np.array([0., 0., 4.])
        result.camera.yaw=result.camera.pitch=0.
        result.camera.distance=4.
        yield result
    finally:
        result.close()
        context.release()


@pytest.mark.parametrize("anti_aliasing", [False, True])
@pytest.mark.parametrize("category", [2, 4, 8, 16, 64])
@pytest.mark.parametrize("contact_kind", [2, 4])
def test_only_debug_arrows_show_through_opaque_geometry(renderer, category, anti_aliasing, contact_kind):
    """Depth-testing arrows or disabling depth for contacts/queries breaks this."""
    renderer.set_settings(RenderingSettings(anti_aliasing=anti_aliasing))
    if anti_aliasing:
        assert renderer.samples >= 2
    renderer.render(256, 256)
    baseline=np.frombuffer(renderer.framebuffer.read(components=3), dtype="u1").reshape(-1, 3)
    baseline_depth=renderer.framebuffer.read(attachment=-1, components=1, dtype="f4")
    renderer.pick(128, 128)
    baseline_ids=renderer.pick_buffer.read(components=4)

    snapshot=overlay_snapshot(category)
    # Structured OVD contacts use lines; legacy PVD contact normals are arrows.
    snapshot.kinds[1]=contact_kind
    renderer.set_overlays(snapshot)
    renderer.render(256, 256)
    pixels=np.frombuffer(renderer.framebuffer.read(components=3), dtype="u1").reshape(-1, 3)
    changed=np.any(pixels != baseline, axis=1)
    red=(pixels[:, 0] > 200) & (pixels[:, 1] < 40) & (pixels[:, 2] < 40)
    assert np.count_nonzero(red & changed) > 25, "Opaque geometry must not bury debug arrows"
    # Every changed pixel must move toward arrow red, even at MSAA edges.
    assert np.all(pixels[changed, 0] >= baseline[changed, 0])
    assert np.all(pixels[changed, 1:] <= baseline[changed, 1:])
    assert renderer.framebuffer.read(attachment=-1, components=1, dtype="f4") == baseline_depth
    # Visual on-top arrows deliberately leave the occluded picking pass alone.
    assert renderer.pick(128, 128) == 100
    assert renderer.pick_buffer.read(components=4) == baseline_ids

    # Confirm that contact/query fixtures really render when no body occludes
    # them; otherwise an accidentally empty batch could make the proof pass.
    renderer.hidden={100}
    renderer.render(256, 256)
    exposed=np.frombuffer(renderer.framebuffer.read(components=3), dtype="u1").reshape(-1, 3)
    green=(exposed[:, 0] < 40) & (exposed[:, 1] > 100) & (exposed[:, 2] < 40)
    cyan=(exposed[:, 0] < 40) & (exposed[:, 1] > 200) & (exposed[:, 2] > 200)
    assert np.count_nonzero(green) > 25
    assert np.count_nonzero(cyan) > 25
    renderer.pick(128, 128)
    exposed_ids=set(np.frombuffer(renderer.pick_buffer.read(components=4), dtype="<u4"))
    assert {11, 12, 13, 14, 21} <= exposed_ids

    renderer.hidden.clear()
    renderer.set_overlays(None)
    renderer.render(256, 256)
    assert renderer.framebuffer.read(components=3) == baseline.tobytes()
    assert renderer.context.error == "GL_NO_ERROR"


@pytest.mark.parametrize("anti_aliasing", [False, True])
@pytest.mark.parametrize("depth_enabled", [False, True])
def test_arrow_scope_restores_caller_depth_state(renderer, anti_aliasing, depth_enabled):
    """A normal line drawn after the arrow pass must inherit the original depth state."""
    renderer.set_settings(RenderingSettings(anti_aliasing=anti_aliasing))
    renderer.render(256, 256)
    target=renderer.msaa_buffer or renderer.framebuffer
    snapshot=overlay_snapshot(16)
    snapshot.kinds[1:]=0  # Only the red arrow participates in this state probe.
    renderer.overlays.set_snapshot(snapshot)
    flags=moderngl.BLEND | (moderngl.DEPTH_TEST if depth_enabled else 0)
    with renderer.context.scope(framebuffer=target, enable_only=flags):
        renderer.overlays.draw(renderer._view_projection(), renderer.eye(), renderer.distance,
                               None, framebuffer=target)
        # Same hidden segment, now an ordinary green line. Do not reset depth:
        # its pixels prove the nested arrow scope restored the caller's state.
        snapshot.kinds[0]=2
        snapshot.colors[0]=[0, 1, 0, 1]
        renderer.overlays.draw(renderer._view_projection(), renderer.eye(), renderer.distance, None)
    if renderer.msaa_buffer is not None:
        renderer.context.copy_framebuffer(renderer.framebuffer, target)
    pixels=np.frombuffer(renderer.framebuffer.read(components=3), dtype="u1").reshape(-1, 3)
    green=(pixels[:, 0] < 40) & (pixels[:, 1] > 200) & (pixels[:, 2] < 40)
    if depth_enabled:
        assert not np.any(green)
    else:
        assert np.count_nonzero(green) > 25
    assert renderer.context.error == "GL_NO_ERROR"


@pytest.mark.parametrize("anti_aliasing", [False, True])
@pytest.mark.parametrize("contact_kind", [2, 4])
def test_visible_contacts_draw_after_on_top_debug_arrows(renderer, anti_aliasing, contact_kind):
    """OvUI deliberately lets depth-tested visible contacts cover debug arrows."""
    renderer.set_settings(RenderingSettings(anti_aliasing=anti_aliasing))
    renderer.hidden={100}
    snapshot=overlay_snapshot(16)
    snapshot.query_geometry=None
    snapshot.kinds[2:]=0
    snapshot.kinds[1]=contact_kind
    snapshot.positions[1]=snapshot.positions[0]
    renderer.set_overlays(snapshot)
    renderer.render(256, 256)
    pixels=np.frombuffer(renderer.framebuffer.read(components=3), dtype="u1").reshape(-1, 3)
    green=(pixels[:, 0] < 40) & (pixels[:, 1] > 100) & (pixels[:, 2] < 40)
    assert np.count_nonzero(green) > 25
    assert renderer.context.error == "GL_NO_ERROR"


@pytest.mark.parametrize("anti_aliasing", [False, True])
def test_arrow_and_line_batches_reuse_warm_gpu_storage(renderer, monkeypatch, anti_aliasing):
    """Sharing a cache key makes unequal line batches orphan twice per frame."""
    renderer.set_settings(RenderingSettings(anti_aliasing=anti_aliasing))
    renderer.hidden={100}
    snapshot=overlay_snapshot(16)
    snapshot.query_geometry=None
    snapshot.kinds[2:]=0
    renderer.set_overlays(snapshot)
    renderer.render(256, 256)
    if anti_aliasing:
        assert renderer.samples >= 2
    baseline=renderer.framebuffer.read(components=3)
    pixels=np.frombuffer(baseline, dtype="u1").reshape(-1, 3)
    # Include partially covered MSAA line pixels over the dark background.
    red=(pixels[:, 0] > 120) & (pixels[:, 1] < 60) & (pixels[:, 2] < 60)
    green=(pixels[:, 0] < 60) & (pixels[:, 1] > 120) & (pixels[:, 2] < 60)
    assert np.count_nonzero(red) > 25
    assert np.count_nonzero(green) > 25

    # Observe actual GL allocations without replacing any upload or draw work.
    buffers=tuple(buffer for buffer, _ in renderer.overlays.buffers.values())
    orphaned=[]
    real_orphan=moderngl.Buffer.orphan
    def track_orphan(buffer, *args, **kwargs):
        if buffer in buffers:
            orphaned.append((buffer.glo, args, kwargs))
        return real_orphan(buffer, *args, **kwargs)
    monkeypatch.setattr(moderngl.Buffer, "orphan", track_orphan)

    for _ in range(3):
        renderer.render(256, 256)
        assert renderer.framebuffer.read(components=3) == baseline
    assert not orphaned, "Unchanged arrow and ordinary line batches must not resize each other's storage"
    assert tuple(buffer for buffer, _ in renderer.overlays.buffers.values()) == buffers
    assert renderer.context.error == "GL_NO_ERROR"


@pytest.mark.parametrize("anti_aliasing", [False, True])
@pytest.mark.parametrize("category", [2, 4, 8, 16, 64])
@pytest.mark.parametrize("use_palette,selected_rgb", [(False, (204, 240, 125)), (True, (255, 120, 120))])
def test_selected_on_top_arrow_uses_selection_tint(renderer, category, anti_aliasing, use_palette, selected_rgb):
    """Skipping selection tint for the split arrow batch leaves red GPU pixels."""
    renderer.set_settings(RenderingSettings(anti_aliasing=anti_aliasing))
    if use_palette:
        renderer.set_palette(ColorPalette(DEFAULT_PALETTE_PATH))
    renderer.render(256, 256)
    body_pixels=np.frombuffer(renderer.framebuffer.read(components=3), dtype="u1").reshape(-1, 3).astype(float)
    snapshot=overlay_snapshot(category)
    snapshot.query_geometry=None
    snapshot.kinds[1:]=0
    original_colors=snapshot.colors.copy()
    renderer.set_overlays(snapshot)
    renderer.render(256, 256)
    if anti_aliasing:
        assert renderer.samples >= 2
    baseline=renderer.framebuffer.read(components=3)
    pixels=np.frombuffer(baseline, dtype="u1").reshape(-1, 3)
    # Infer coverage from the red arrow's zero green channel. This preserves
    # the literal tint expectation even when a driver only partially covers
    # every shaft pixel under MSAA; no raster-position assumption is needed.
    coverage=(body_pixels[:, 1]-pixels[:, 1])/np.maximum(body_pixels[:, 1], 1)
    arrow_pixels=coverage > .5
    assert np.count_nonzero(arrow_pixels) > 25
    baseline_depth=renderer.framebuffer.read(attachment=-1, components=1, dtype="f4")
    renderer.pick(128, 128)
    baseline_ids=renderer.pick_buffer.read(components=4)

    renderer.selected={10}
    renderer.render(256, 256)
    selected_pixels=np.frombuffer(renderer.framebuffer.read(components=3), dtype="u1").reshape(-1, 3)
    # Literal RGBs independently encode the fallback tint or red + #787878.
    coverage=coverage[arrow_pixels, None]
    expected=body_pixels[arrow_pixels]*(1-coverage)+np.array(selected_rgb)*coverage
    np.testing.assert_allclose(selected_pixels[arrow_pixels], expected, atol=2)
    assert renderer.framebuffer.read(attachment=-1, components=1, dtype="f4") == baseline_depth
    # On-top display is not on-top picking: the arrow cannot steal the body hit.
    arrow_indices=np.flatnonzero(arrow_pixels)
    arrow_y,arrow_x=divmod(int(arrow_indices[len(arrow_indices)//2]), 256)
    assert renderer.pick(arrow_x, arrow_y) == 100
    assert renderer.pick_buffer.read(components=4) == baseline_ids
    np.testing.assert_array_equal(snapshot.colors, original_colors)

    renderer.selected.clear()
    renderer.render(256, 256)
    assert renderer.framebuffer.read(components=3) == baseline
    assert renderer.context.error == "GL_NO_ERROR"
