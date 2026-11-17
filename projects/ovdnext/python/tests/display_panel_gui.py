# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned, hidden native Display-layout regression; optional --screenshot PATH.

Run with the production native modules/DLL search path supplied by the host.
No desktop input is injected, and all preferences/capture output are temporary.
"""
import argparse
from contextlib import chdir
import ctypes
import os
from pathlib import Path
import tempfile

import numpy as np
import omni.ui as ui
from _pvd_native import Session
from ovd_ui import style
from ovd_ui.application import Application
from ovd_ui.camera import CameraController
from ovd_ui.renderer import ViewportRenderer
from ovd_ui.design_tokens import rgba as design_rgba, STAGE_GRID_RGBA
from ovd_ui.widgets.file_browser import CaptureFileDialog

ROOT=Path(__file__).resolve().parents[2]


async def settle(app):
    for _ in range(8):
        await ui.next_frame()
        app._update_split_grips()
        for slider in getattr(app,"view_sliders",()):slider.update()
        app.transparency_slider.update()
        if app.presentation_dirty:app._refresh()
        app._sync_workspace()
        app.workspace.poll()
        size=int(app.viewport.computed_width),int(app.viewport.computed_height)
        if min(size)>0:
            app.renderer.render(*size,light=app.theme=='light',background=design_rgba('background',app.theme),
                background_gradient=(design_rgba('stage_center',app.theme),design_rgba('stage_edge',app.theme)),
                ground_color=design_rgba('elevated' if app.theme=='light' else 'surface',app.theme),grid_contrast=.025,
                grid_rgba=STAGE_GRID_RGBA[app.theme],scene_length=app.navigation.scene_length,
                gizmo_scale=app.gizmo_scale*app.master_scale)


async def verify(app,scrolls,screenshot):
    await settle(app)
    actor=int(app.renderer.snapshot.object_ids[app.renderer.snapshot.kinds!=4][0])
    app._tree_selection([app.object_model.canonical_item(actor)])
    await settle(app)
    camera=app.renderer.camera
    before=camera.position.copy(),camera.target.copy()
    viewport=app.viewport_region
    viewport_width=viewport.computed_width
    app.show_view_options()
    await settle(app)
    assert ui.Workspace.get_window("Viewport options") is None,"Display must not create a floating window"
    assert app.view_panel.visible,"Display must open in the inspector column"
    assert app.view_option_controls and app.view_sliders,"Display controls must be built on opening"
    controls=app.view_option_controls.copy()
    panel=app.view_panel
    print('Display initial extents', app.window.width, app.window.height,
        app.inspector_column.computed_height, flush=True)
    scroll=next(s for s in reversed(scrolls) if
        s.screen_position_x>=panel.screen_position_x and s.screen_position_y>=panel.screen_position_y)
    for width,height in ((320,220),(420,380),(260,180),(320,280)):
        app.inspector_column.width=ui.Pixel(width)
        grip=app.view_grip
        delta=(height-panel.computed_height)/grip.direction
        grip.press(0,0,0,0)
        grip.move(0,delta)
        grip.release(0,delta,0,0)
        await settle(app)
        assert abs(app.inspector_column.computed_width-width)<1
        assert abs(panel.computed_height-height)<1,(height,panel.computed_height)
        assert viewport.computed_width>100 and viewport.computed_height>100
        assert app.property_frame.computed_width>0 and app.property_frame.computed_height>0
        assert viewport.screen_position_x+viewport.computed_width<=panel.screen_position_x+1
        assert app.property_frame.screen_position_y<panel.screen_position_y
        assert panel.screen_position_x>=app.inspector_column.screen_position_x
        assert panel.screen_position_x+panel.computed_width<=app.inspector_column.screen_position_x+width+1
        assert panel.screen_position_y+panel.computed_height<=app.inspector_column.screen_position_y+app.inspector_column.computed_height+1
        assert scroll.scroll_y_max>0,"All Display controls must remain reachable by scrolling"
        scroll.scroll_y=scroll.scroll_y_max
        await settle(app)
        assert scroll.scroll_y>0
        scroll.scroll_y=0
        await settle(app)
    # The hidden backend may not own native focus. Never acquire desktop focus
    # just to exercise routing; report that coverage explicitly when unavailable.
    if app.window.focused:
        x=panel.screen_position_x+panel.computed_width/2
        y=panel.screen_position_y+panel.computed_height/2
        for focus in (lambda:app._set_viewport_keyboard_focus(True),app._focus_objects):
            focus()
            app._workspace_pressed(x,y,0,0)
            assert not app.viewport_keyboard_focus and not app.object_keyboard_focus
        print("Display panel input releases viewport/object keyboard focus",flush=True)
    else:print("Focus routing not exercised: hidden test window is unfocused",flush=True)
    if screenshot:
        from omni.ui import testing
        assert testing.capture_screenshot(str(Path(screenshot).resolve()))
        app.toggle_theme()
        await settle(app)
        path=Path(screenshot)
        assert testing.capture_screenshot(str(path.with_name(path.stem+"-light"+path.suffix).resolve()))
    app.show_view_options()
    await settle(app)
    assert not panel.visible
    assert abs(viewport.computed_width-viewport_width)<1,"Closing Display must not change viewport width"
    app.show_view_options()
    await settle(app)
    assert panel.visible and all(app.view_option_controls[key] is value for key,value in controls.items())
    np.testing.assert_array_equal(camera.position,before[0])
    np.testing.assert_array_equal(camera.target,before[1])
    assert not app.error_message,app.error_message
    print("Native embedded Display: placement, resize, scroll, reopen, and unchanged camera passed",flush=True)
    source=ROOT/"test/data/2026_04_28_15_24_52_000000a751246bf6.ovd"
    assert app._display(source,Session(str(source),str(app.workflow.settings.path.parent/"contacts")))
    controls[32].set_value(1)
    await settle(app)
    assert app.frame==2 and app.contact_hint.visible and app.contact_pre_button.enabled
    if screenshot:
        from omni.ui import testing
        path=Path(screenshot)
        assert testing.capture_screenshot(str(path.with_name(path.stem+'-contacts'+path.suffix)))
    button=app.contact_pre_button
    assert button.computed_width>0 and button.computed_height>0
    dpi=ui.Workspace.get_dpi_scale()  # Widgets use logical units; injection uses pixels.
    ui._ui._inject_mouse_move((button.screen_position_x+button.computed_width/2)*dpi,
                              (button.screen_position_y+button.computed_height/2)*dpi)
    await ui.next_frame()
    ui._ui._inject_mouse_button(0,True)
    await ui.next_frame()
    ui._ui._inject_mouse_button(0,False)
    await settle(app)
    assert app.frame==1 and not app.contact_hint.visible and not app.playback.playing,(
        app.frame,app.contact_hint.visible,app.playback.playing,ui.Workspace.get_dpi_scale(),
        button.screen_position_x,button.screen_position_y,button.computed_width,button.computed_height)
    print("Native contact guidance: explicit button selects matching Pre-sim frame",flush=True)
    snapshot=app.presentation.overlay_snapshot()
    first,second=np.unique(snapshot.related_ids[snapshot.categories==32],axis=0)
    common=set(first)&set(second)
    single=set(first)-common
    for mode,selected,count in ((2,common,36),(2,single,18),(2,set(),0),
                                (3,single,18),(3,set(),36),(0,common,0),(1,set(),36)):
        owned=app.presentation
        controls[32].set_value(mode)
        await settle(app)
        if mode in (2,3):assert app.presentation is owned,"Mode changes must reuse contacts"
        # Selecting another object also requests the existing inspector refresh.
        app._tree_selection([app.object_model.canonical_item(int(actor)) for actor in selected])
        await settle(app)
        overlays=app.renderer.overlays
        rendered=overlays.primitive_count if overlays.snapshot is not None else 0
        assert rendered==count,(mode,selected,count,rendered)
    assert not app.error_message,app.error_message
    print("Native Contacts dropdown: rendered either-actor selection and None/All/Selected/Unselected passed",flush=True)
    controls[32].set_value(0)
    source=ROOT/"test/data/articulation.ovd"
    assert app._display(source,Session(str(source),str(app.workflow.settings.path.parent/"frames")))
    controls[16].set_value(1)
    controls[256].set_value(1)
    await settle(app)
    snapshot=app.presentation.overlay_snapshot()
    joints=snapshot.categories==16
    joint=int(snapshot.object_ids[joints][0])
    parent=int(snapshot.related_ids[joints][0,0])
    for bit,actor,count in ((16,joint,6),(256,parent,3)):
        other=256 if bit==16 else 16
        controls[other].set_value(0)
        controls[bit].set_value(2)
        app._tree_selection([app.object_model.canonical_item(actor)])
        await settle(app)
        assert app.renderer.overlays.primitive_count==count,(bit,app.renderer.overlays.primitive_count)
        assert app.renderer.overlays.draw_count==1,"Debug lines must stay batched"
    old=app.presentation.overlay_snapshot()
    controls['com_scale'].set_value(2.)
    controls['axes_scale'].set_value(3.)  # Disabled family: save without adding rows.
    await settle(app)
    current=app.presentation.overlay_snapshot()
    assert current.family_scales==app.overlay_scales
    np.testing.assert_allclose(current.positions[:,:2].mean(axis=1),old.positions[:,:2].mean(axis=1),atol=1e-6)
    assert app.renderer.overlays.primitive_count==3
    assert not app.error_message,app.error_message
    print("Native family controls: joint/COM selection, independent saved sizes, fixed origins and batched lines passed",flush=True)
    master=next(slider for slider in app.view_sliders if slider.model is controls['master_scale'])
    factors=app.overlay_scales.copy()
    master.position_model.set_value(.75)
    await settle(app)
    scaled=app.presentation.overlay_snapshot()
    assert app.overlay_scales==factors and abs(scaled.master_scale-4.47213595)<1e-6
    np.testing.assert_allclose(scaled.positions[:,:2].mean(axis=1),current.positions[:,:2].mean(axis=1),atol=2e-6)
    # Compare analytic COM lengths; rescaling rounded world-space deltas amplifies
    # endpoint cancellation for actors far from the origin. Allow two endpoint ULPs.
    length=np.linalg.norm(scaled.positions[:,1]-scaled.positions[:,0],axis=1)
    tolerance=2*np.spacing(np.abs(scaled.positions[:,:2])).max()+2e-6
    np.testing.assert_allclose(length,.1*factors[256]*scaled.master_scale,rtol=0,atol=tolerance)
    assert app.renderer.overlays.draw_count==1 and app.renderer.overlays.primitive_count==3
    master.position_model.set_value(.501)
    await settle(app)
    assert app.master_scale==1. and master.position_model.as_float==.5
    assert not app.error_message,app.error_message
    if screenshot:
        scroll.scroll_y += master.widget.screen_position_y-scroll.screen_position_y
        await settle(app)
        from omni.ui import testing
        path=Path(screenshot)
        assert testing.capture_screenshot(str(path.with_name(path.stem+'-controls'+path.suffix)))
    print("Native master slider: rendered multiplication, neutral snap, unchanged family factors and batching passed",flush=True)
    with tempfile.TemporaryDirectory(prefix='capture-sort-') as directory:
        folder=Path(directory).resolve()  # Windows TEMP can use an 8.3 user name.
        for name,stamp in (('a-old.ovd',1000000),('z-new.ovd',2000000)):
            path=folder/name
            path.write_bytes(b'capture')
            os.utime(path,(stamp,stamp))
        chosen=[]
        dialog=CaptureFileDialog(chosen.append,initial_path=folder)
        try:
            dialog.show()
            await dialog._task
            await settle(app)
            dialog.sort_combo.model.get_item_value_model().set_value(1)
            await settle(app)
            assert [item.entry.path.name for item in dialog.model.items]==['z-new.ovd','a-old.ovd']
            assert dialog.tree.computed_width>0 and dialog.sort_combo.computed_width>0
            if screenshot:
                path=Path(screenshot)
                assert testing.capture_screenshot(str(path.with_name(path.stem+'-picker'+path.suffix)))
            dialog.tree.selection=[dialog.model.items[0]]
            dialog._double_clicked()
            assert chosen==[folder/'z-new.ovd'],(chosen,dialog.snapshot())
            assert not dialog.visible
        finally:dialog.destroy()
    print("Native file picker: Modified sorting and selected capture opening passed",flush=True)


def main(screenshot=None):
    # Resolve caller-owned output before isolating ImGui's relative ini writes.
    screenshot=str(Path(screenshot).resolve()) if screenshot is not None else None
    with tempfile.TemporaryDirectory(prefix='display-output-') as output:
        with chdir(output):
            _run_display(screenshot)


def _run_display(screenshot):
    # Suppress Windows crash/critical-error dialogs; hide only this test process.
    if os.name=="nt":ctypes.windll.kernel32.SetErrorMode(3)
    old_preferences={name:os.environ.get(name) for name in
        ('LOCALAPPDATA','XDG_DATA_HOME','XDG_CONFIG_HOME')}
    scroll_factory=ui.ScrollingFrame
    scrolls=[]
    def scrolling(*args,**kwargs):
        result=scroll_factory(*args,**kwargs)
        scrolls.append(result)
        return result
    with tempfile.TemporaryDirectory(prefix="display-panel-") as temporary:
        os.environ.update({name:str(Path(temporary)/'preferences') for name in old_preferences})
        app=None
        errors=[]
        try:
            ui.init("OvdNext isolated Display regression",width=1600,height=1000,max_fps=60)
            if os.name=="nt":
                from ctypes import wintypes
                user=ctypes.windll.user32
                user.GetWindowRect.argtypes=[ctypes.c_void_p,ctypes.POINTER(wintypes.RECT)]
                user.GetClientRect.argtypes=[ctypes.c_void_p,ctypes.POINTER(wintypes.RECT)]
                user.GetWindowTextW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_int]
                user.SetWindowPos.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,
                    ctypes.c_int,ctypes.c_int,ctypes.c_uint]
                owned=[]
                @ctypes.WINFUNCTYPE(ctypes.c_bool,ctypes.c_void_p,ctypes.c_void_p)
                def hide(hwnd,_):
                    process=ctypes.c_ulong()
                    ctypes.windll.user32.GetWindowThreadProcessId(ctypes.c_void_p(hwnd),ctypes.byref(process))
                    if process.value==os.getpid():
                        ctypes.windll.user32.ShowWindow(ctypes.c_void_p(hwnd),0)
                        title=ctypes.create_unicode_buffer(256)
                        user.GetWindowTextW(hwnd,title,len(title))
                        if title.value=='OvdNext isolated Display regression':owned.append(hwnd)
                    return True
                ctypes.windll.user32.EnumWindows(hide,0)
                # Native creation may clamp to the current monitor work area.
                # Size only this already-hidden window for exact layout tests.
                from omni.ui import standalone
                dpi_scale=ui.Workspace.get_dpi_scale()
                target_pixels=(round(1600*dpi_scale),round(1000*dpi_scale))
                standalone.set_window_size(1600,1000)
                assert len(owned)==1,owned
                outer,client=wintypes.RECT(),wintypes.RECT()
                assert user.GetClientRect(owned[0],ctypes.byref(client))
                if (client.right,client.bottom)!=target_pixels:
                    assert user.GetWindowRect(owned[0],ctypes.byref(outer))
                    width=target_pixels[0]+(outer.right-outer.left)-(client.right-client.left)
                    height=target_pixels[1]+(outer.bottom-outer.top)-(client.bottom-client.top)
                    # Avoid interactive max-track clipping on a smaller host
                    # desktop. No activation, Z-order, position or desktop change.
                    assert user.SetWindowPos(owned[0],None,0,0,width,height,0x0400|0x0010|0x0004|0x0002)
                    assert user.GetClientRect(owned[0],ctypes.byref(client))
                    assert (client.right,client.bottom)==target_pixels
                print(f"Display regression DPI: {dpi_scale}; client pixels: {target_pixels}",flush=True)
            style.startup("dark")
            ui.ScrollingFrame=scrolling
            app=Application(width=1600,height=1000)
            app.renderer=ViewportRenderer()
            app.renderer.set_settings(app.render_settings)
            app.navigation=CameraController(app.renderer.camera)
            app.build()
            source=ROOT/"test/data/multi_geom.ovd"
            assert app._display(source,Session(str(source),str(Path(temporary)/"capture")))
            async def run():
                try:await verify(app,scrolls,screenshot)
                except Exception as error:errors.append(error)
                finally:
                    try:app.close()
                    finally:ui.shutdown()
            ui.run(run())
        finally:
            ui.ScrollingFrame=scroll_factory
            try:
                if app is not None:app.close()
            finally:
                ui.shutdown()
                for name,value in old_preferences.items():
                    if value is None:os.environ.pop(name,None)
                    else:os.environ[name]=value
        if errors:raise errors[0]


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot")
    main(parser.parse_args().screenshot)
