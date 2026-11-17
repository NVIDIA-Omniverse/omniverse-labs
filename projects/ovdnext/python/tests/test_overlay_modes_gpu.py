# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual OpenGL raster/picking proof: NVIDIA by default, explicit llvmpipe in CI."""
import os
from pathlib import Path
import sys

import moderngl
import numpy as np
import pytest
from _pvd_native import Session
from capture_semantics import dynamic_geometry_owner

from ovd_ui.camera import Camera
from ovd_ui.overlays import OverlayRenderer
from ovd_ui.render_settings import DisplayMode

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('category',[1,2])
def test_native_bounds_and_axes_draw_and_pick_the_same_selection(tmp_path,category):
    session=Session(str(ROOT/'test/data/multi_geom.ovd'),str(tmp_path))
    scene=session.scenes()[0]
    publication=session.presentation(scene.id,2,overlay_mask=category)
    snapshot=publication.overlay_snapshot()
    geometry=publication.geometry_snapshot()
    spans=geometry.bounds[:,1]-geometry.bounds[:,0]
    bounded=np.all(np.isfinite(geometry.bounds),axis=(1,2))&(np.min(spans,axis=1)>1e-4)
    # Choose a body with the requested native family, not an arbitrary bounded
    # particle/shape that has no bounds or axes overlay rows.
    covered=np.isin(geometry.object_ids,snapshot.object_ids[snapshot.categories==category])
    identity=dynamic_geometry_owner(session,2,2)
    assert np.any(bounded & covered & (geometry.object_ids==identity))
    bounds=geometry.bounds[geometry.object_ids==identity][0]
    camera=Camera()
    camera.frame_bounds(bounds)
    matrix=camera.projection(256,256)@camera.view_matrix()
    del session,publication
    options={} if sys.platform=='win32' else {'backend':'egl'}
    context=moderngl.create_standalone_context(require=330,**options)
    renderer=OverlayRenderer(context)
    framebuffer=context.simple_framebuffer((256,256),components=4)
    try:
        expected_renderer = 'llvmpipe' if os.environ.get('GALLIUM_DRIVER') == 'llvmpipe' else 'NVIDIA'
        assert expected_renderer in context.info['GL_RENDERER']
        renderer.set_snapshot(snapshot)
        def draw(mode,selected,picking):
            renderer.set_mode(category,mode)
            with context.scope(framebuffer=framebuffer):
                framebuffer.clear(0,0,0,0)
                context.viewport=(0,0,256,256)
                renderer.draw(matrix,camera.position,camera.distance,scene.instance_id,
                              selected=selected,picking=picking)
            pixels=framebuffer.read(components=4)
            return pixels,renderer.primitive_count
        for mode,selected in ((DisplayMode.ALL,{identity}),(DisplayMode.SELECTED,{identity}),
                              (DisplayMode.UNSELECTED,{identity}),(DisplayMode.NONE,{identity}),
                              (DisplayMode.SELECTED,set()),(DisplayMode.UNSELECTED,set())):
            color,count=draw(mode,selected,False)
            encoded,pick_count=draw(mode,selected,True)
            assert count==pick_count
            ids={int(value)-1 for value in np.frombuffer(encoded,dtype='<u4') if value}
            if mode==DisplayMode.SELECTED and selected:
                assert ids=={identity}
                assert any(color)
            elif mode==DisplayMode.UNSELECTED and selected:
                assert identity not in ids
            elif mode==DisplayMode.NONE or mode==DisplayMode.SELECTED:
                assert not ids and not any(color)
            else:
                assert identity in ids
        assert context.error=='GL_NO_ERROR'
    finally:
        renderer.close()
        framebuffer.release()
        context.release()
