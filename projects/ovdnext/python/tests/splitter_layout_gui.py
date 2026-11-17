# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real OVUI draw/layout regression for earlier and later sibling resize targets."""
from gui_runtime import isolate_gui_output, hide_owned_window
import json
import omni.ui as ui
from ovd_ui import style
from ovd_ui.application import Application
from ovd_ui.widgets.splitter import SplitGrip
from ovd_ui.style import layout as fl


async def verify():
    app=Application.__new__(Application)
    saved={"left":[],"right":[]}
    queued=[]
    during=[]
    def dispatch():
        while queued:queued.pop(0)()
    window=ui.Window("Splitter layout regression",width=1000,height=500,
        flags=ui.WINDOW_FLAGS_NO_TITLE_BAR|ui.WINDOW_FLAGS_NO_RESIZE)
    with window.frame:
        with ui.HStack(spacing=4):
            left=ui.VStack(width=260)
            with left:ui.Label("Earlier sibling panel",word_wrap=True)
            app.browser_grip=SplitGrip(left,saved["left"].append)
            ui.Spacer()
            right_host=ui.Frame(width=fl.pvd_splitter_hit_width)
            # The build callback runs during native Frame::_drawContent after
            # the earlier sibling draw, before the later sibling draw. It
            # exercises the same callback timing as a splitter mouse event.
            driver=ui.Frame(width=1,build_fn=dispatch)
            right=ui.VStack(width=320)
            with right:ui.Label("Later sibling panel",word_wrap=True)
            with right_host:
                app.inspector_grip=SplitGrip(right,saved["right"].append,direction=-1)
    async def frames(count=1):
        for _ in range(count):
            await ui.next_frame()
            app._update_split_grips()
    await frames(4)
    assert (left.computed_width,right.computed_width)==(260,320)
    for theme,expected in (("dark",(300,240)),("light",(260,320))):
        style.set_theme(theme)
        await frames(3)
        before=(left.computed_width,right.computed_width)
        delta=(expected[0]-before[0],before[1]-expected[1])
        def drag():
            for grip,amount in zip((app.browser_grip,app.inspector_grip),delta):
                grip.press(100,150,0,0)
                grip.move(100+amount/2,150)
                grip.move(100+amount,150)
            during.append((left.computed_width,right.computed_width))
        queued.append(drag);driver.rebuild()
        await frames(3)
        assert during[-1]==before,"Mouse callback must not mutate layout during draw"
        actual=(left.computed_width,right.computed_width)
        assert actual==expected,(theme,before,actual,expected)
        assert len(saved["left"])==len(saved["right"])
        count=len(saved["left"])
        def release():
            app.browser_grip.release(100+delta[0],150,0)
            app.inspector_grip.release(100+delta[1],150,0)
            assert len(saved["left"])==count,"Persist only after completed layout"
        queued.append(release);driver.rebuild()
        await frames(3)
        assert saved["left"][-1]==expected[0] and saved["right"][-1]==expected[1]
        assert len(saved["left"])==count+1
        await frames(4)
        assert len(saved["left"])==count+1,"Idle frames must not persist repeatedly"
    # Widget constraints can exceed requested width. Persist what layout chose,
    # rather than the unfulfilled request or the previous cached width.
    with right:ui.Label("A fixed-width child",width=350)
    await frames(3)
    def constrained():
        app.inspector_grip.press(100,150,0,0)
        app.inspector_grip.release(250,150,0)
    queued.append(constrained);driver.rebuild()
    await frames(3)
    assert right.computed_width==350 and saved["right"][-1]==350
    # Destruction after an input callback prevents pending apply and save.
    before=(left.computed_width,right.computed_width,tuple(saved["left"]),tuple(saved["right"]))
    def cancelled():
        app.browser_grip.press(100,150,0,0)
        app.browser_grip.release(200,150,0)
        app._destroy_split_grips()
    queued.append(cancelled);driver.rebuild()
    await frames(3)
    assert before==(left.computed_width,right.computed_width,tuple(saved["left"]),tuple(saved["right"]))
    print(json.dumps({"themes":["dark","light"],"saved_widths":saved,"during_draw":during,
                      "constraint_width":right.computed_width,"cancelled":True}),flush=True)
    window.destroy()
    await frames(2)


if __name__=="__main__":
    errors=[]
    async def run():
        try:await verify()
        except Exception as error:errors.append(error)
        finally:ui.shutdown()
    isolate_gui_output()
    ui.init("Splitter layout regression",width=1000,height=500,max_fps=60)
    hide_owned_window()
    style.startup("dark")
    ui.run(run())
    if errors:raise errors[0]
