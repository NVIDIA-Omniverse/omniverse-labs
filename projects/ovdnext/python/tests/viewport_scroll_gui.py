# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native window-hit-test regression, using the same event injection as Inspector.

This is an automated event-dispatch test, not screenshot-first desktop evidence.
No direct Application._scroll or widget callback invocation substitutes for input.
"""
import json
import omni.ui as ui
from ovd_ui.application import Application
from ovd_ui.camera import Camera,CameraController
from ovd_ui import style
from gui_runtime import isolate_gui_output
from test_application_presentation import Renderer


async def verify():
    app=Application.__new__(Application)
    app.navigation=CameraController(Camera())
    app.renderer=Renderer()
    app.dirty=False
    app.mouse_anchor=None
    app.viewport_hovered=False
    app.window=ui.Window('Viewport scroll ownership',width=800,height=600,
        flags=ui.WINDOW_FLAGS_NO_TITLE_BAR|ui.WINDOW_FLAGS_NO_MOVE|ui.WINDOW_FLAGS_NO_RESIZE|1<<13,
        fill_app_window=True)
    with app.window.frame:
        app._build_viewport_input()
    overlay=ui.Window('Profile table',width=320,height=350,
        flags=ui.WINDOW_FLAGS_NO_TITLE_BAR|ui.WINDOW_FLAGS_NO_RESIZE|ui.WINDOW_FLAGS_NO_MOVE)
    overlay.setPosition(200,100)
    wheel_events=[]
    with overlay.frame:
        panel=ui.InvisibleButton(explicit_hover=True,scroll_only_window_hovered=True,
            mouse_wheel_fn=lambda x,y,*_:wheel_events.append(y))
    async def frames(count=3):
        for _ in range(count):await ui.next_frame()
    async def wheel(x,y):
        ui._ui._inject_mouse_move(x,y)
        await frames()
        ui._ui._inject_mouse_scroll(0,1)
        await frames()
    results=[]
    for theme in ('dark','light'):
        style.set_theme(theme)
        await frames()
        # NO_BRING_TO_FRONT_ON_FOCUS lets us focus the workspace while keeping
        # the floating profiler above it. Hovering that panel must not focus it.
        app.window.focus()
        await frames()
        assert app.window.focused and not overlay.focused
        before=app.navigation.camera.distance
        count=len(wheel_events)
        await wheel(350,250)
        assert app.window.focused,'Wheel-first did not retain workspace focus'
        assert len(wheel_events)==count+1,'Foreground profiler did not get its wheel'
        assert app.navigation.camera.distance==before,'Background camera received a profiler wheel'
        # Positive control: viewport hover alone permits zoom, without a click.
        await wheel(650,500)
        after=app.navigation.camera.distance
        assert after!=before,'The real viewport must still receive its wheel'
        results.append({'theme':theme,'before':before,'after':after,'panel_wheel':wheel_events[-1]})
    overlay.destroy();app.window.destroy()
    await frames(2)
    print(json.dumps({'wheel_first':True,'results':results}),flush=True)


if __name__=='__main__':
    errors=[]
    async def run():
        try:await verify()
        except Exception as error:errors.append(error)
        finally:ui.shutdown()
    isolate_gui_output()
    ui.init('Viewport scroll ownership',width=800,height=600,max_fps=60)
    style.startup('dark')
    ui.run(run())
    if errors:raise errors[0]
