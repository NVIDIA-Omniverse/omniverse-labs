# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native OVUI dispatch regression, separate from strict Inspector screenshots.

Input injection is confined to this test; PlotView uses only public callbacks.
"""
from pathlib import Path
import tempfile
import time

import omni.ui as ui
from _pvd_native import Session
from ovd_ui import style
from ovd_ui.analysis import AnalysisTasks
from ovd_ui.widgets.analysis_views import PlotView
from gui_runtime import isolate_gui_output

ROOT = Path(__file__).resolve().parents[2]


async def verify():
    view = None
    last_poll = 0.

    async def frames(count=4):
        nonlocal last_poll
        for _ in range(count):
            await ui.next_frame()
            now = time.monotonic()
            if view and now - last_poll >= 1 / 30:
                view.poll()
                # Bound the product's clock sample inside poll(), not its start.
                last_poll = time.monotonic()

    async def key(code, modifiers=()):
        for modifier in modifiers:
            ui._ui._inject_key_event(modifier, True)
        ui._ui._inject_key_event(code, True)
        await frames(1)
        ui._ui._inject_key_event(code, False)
        for modifier in reversed(modifiers):
            ui._ui._inject_key_event(modifier, False)
        await frames()

    async def pointer(x, y, button=None):
        ui._ui._inject_mouse_move(x, y)
        await frames()
        if button is not None:
            ui._ui._inject_mouse_button(button, True)
            await frames(1)
            ui._ui._inject_mouse_button(button, False)
            await frames()

    with tempfile.TemporaryDirectory() as temporary:
        path = ROOT / "test/data/cloth_physx34.pxd2"
        session = Session(str(path), temporary)
        scene_id = session.scenes()[0].id
        session.seek(scene_id, 7)
        owned = session.presentation(frame=7)
        owned_generation = owned.generation
        jumps = []
        tasks = AnalysisTasks()
        view = PlotView(tasks.submit_frame_timing(path, scene_id, 1, 250),
                        "Graph keyboard native delivery", jumps.append)
        view.window.setPosition(50, 40)
        background = ui.Window("Graph focus owner", width=80, height=40)
        background.setPosition(1170, 720)
        with background.frame:
            ui.Button("Other")
        deadline = time.monotonic() + 15
        while not view.finished:
            assert time.monotonic() < deadline, (
                f"Graph analysis timed out: progress={view.job.progress}; "
                f"done={view.job.future.done()}; status={view.progress.text}")
            await frames(1)
        assert view.model is not None, view.progress.text
        series = view.series
        for theme in ("dark", "light"):
            style.set_theme(theme)
            view.window.focus()
            await frames()
            x = view.canvas.screen_position_x + 250
            y = view.canvas.screen_position_y + 150
            await pointer(x + 30, y + 20)
            await pointer(x, y, 0)
            original_flags = view.window.flags
            original = view.model.view
            ui._ui._inject_key_event(262, True)
            await frames(2)
            first = view.keyboard.position
            assert first == (251, 150), (theme, first)
            assert view.window.focused and not view.editing_fields
            assert view.window.flags == original_flags | (1 << 16)
            # A frame count is not a clock: llvmpipe can take >400ms for
            # nine frames under load. Check each pre-deadline poll instead.
            repeat_at = view.keyboard.next_repeat
            deadline = time.monotonic() + 2
            while view.keyboard.position == first:
                assert time.monotonic() < deadline, (
                    "Held arrow did not repeat", vars(view.keyboard))
                await frames(1)
                if last_poll < repeat_at:
                    assert view.keyboard.position == first, "Repeated before the 400ms delay"
            repeated = view.keyboard.position
            assert repeated[0] > first[0] and (repeated[0] - first[0]) % 5 == 0
            assert repeated[1] == first[1]
            ui._ui._inject_key_event(262, False)
            await frames(10)
            assert view.keyboard.position == repeated and view.keyboard.held is None
            await key(257)
            assert view.keyboard.active
            await key(264)
            await key(335)
            assert not view.keyboard.active and view.model.view != original
            assert view.model.history_index == 1
            await key(79, (340,))
            assert view.model.view == original and view.model.history_index == 0
            await key(73, (340,))
            assert view.model.history_index == 1
            await key(79, (340,))
            history = list(view.model.history)
            await key(257)
            await key(262)
            await key(256)
            assert not view.keyboard.visible and view.model.history == history
            assert view.model.history_index == 0 and view.window.flags == original_flags
            # Actual popup dispatch must not leave an active rectangle behind it.
            await key(257)
            await pointer(x, y, 1)
            assert ui.Menu.get_current() is not None and not view.keyboard.visible
            await key(262)
            assert not view.keyboard.visible and view.model.history == history
            await key(256)
            view.window.focus()
            await frames()
            await key(257)
            assert view.keyboard.active
            background.focus()
            await frames()
            assert not view.keyboard.visible and view.window.flags == original_flags
            view.window.focus()
            await frames()
            await key(257)
            assert view.keyboard.active
            view.window.width -= 20
            await frames(8)
            assert not view.keyboard.visible and view.model.history == history
            view.window.width += 20
            await frames(8)
            assert view.series is series and not jumps
            assert session.scenes()[0].current_frame == 7
            assert owned.generation == owned_generation and owned.frame == 7
            # Reset only graph history for the next independent theme pass.
            view.model.history = [original]
            view.model.history_index = 0
        await key(257)
        assert view.keyboard.active
        view.destroy()
        assert not view.keyboard.visible and not view.keyboard.active
        assert not jumps and session.scenes()[0].current_frame == 7
        tasks.close()
        background.destroy()
        await frames(2)
        print("Native graph keyboard delivery passed in both themes", flush=True)


if __name__ == "__main__":
    errors = []

    async def run():
        try:
            await verify()
        except Exception as error:
            errors.append(error)
        finally:
            ui.shutdown()

    isolate_gui_output()
    ui.init("Graph keyboard dispatch", width=1280, height=800, max_fps=60)
    style.startup("dark")
    ui.run(run())
    if errors:
        raise errors[0]
