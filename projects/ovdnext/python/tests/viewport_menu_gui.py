# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native OVUI event-dispatch regression, separate from strict Inspector images.

Uses Inspector's input injection, never a direct mouse/menu callback. The real
capture/selection model is paired with a renderer sink to isolate hit ownership.
"""
import json
import tempfile
from pathlib import Path

import omni.ui as ui
from ovd_ui import style
from ovd_ui.camera import Camera, CameraController
from gui_runtime import isolate_gui_output
from test_application_presentation import app as app_fixture, open_capture
from test_application_object_actions import choose, selection_and_publication


async def verify():
    app = app_fixture.__wrapped__()
    with tempfile.TemporaryDirectory() as directory:
        assert app._display(*open_capture(Path(directory), "test_scene.ovd"))
        choose(app, "object:1")
        app.renderer.camera, app.renderer.size = Camera(), (800, 600)
        app.navigation = CameraController(app.renderer.camera)
        picks = []
        app.renderer.pick = lambda x, y: picks.append((x, y)) or 1
        app.mouse_anchor = None
        app.window = ui.Window("Viewport menu ownership", width=800, height=600,
            flags=ui.WINDOW_FLAGS_NO_TITLE_BAR | ui.WINDOW_FLAGS_NO_MOVE |
                  ui.WINDOW_FLAGS_NO_RESIZE | (1 << 13), fill_app_window=True)
        with app.window.frame:
            app.viewport = ui.ZStack()
            with app.viewport: app._build_viewport_input()
        overlay = ui.Window("Foreground panel", width=220, height=250,
            flags=ui.WINDOW_FLAGS_NO_TITLE_BAR | ui.WINDOW_FLAGS_NO_MOVE | ui.WINDOW_FLAGS_NO_RESIZE)
        overlay.setPosition(100, 100)
        foreground = []
        with overlay.frame:
            ui.InvisibleButton(explicit_hover=True,
                mouse_pressed_fn=lambda *args: foreground.append(args))

        async def frames(count=3):
            for _ in range(count): await ui.next_frame()

        async def move(x, y):
            ui._ui._inject_mouse_move(x, y)
            await frames()

        async def click(x, y, button):
            await move(x, y)
            ui._ui._inject_mouse_button(button, True)
            await frames()
            ui._ui._inject_mouse_button(button, False)
            await frames()

        def camera(): return app.navigation.camera.view_matrix().tobytes()

        results = []
        for theme in ("dark", "light"):
            style.set_theme(theme)
            await frames()
            app.window.focus()
            await frames()
            assert app.window.focused and not overlay.focused
            before = camera(), selection_and_publication(app)
            # A foreground panel must win even before its first focus click.
            count = len(foreground)
            await click(200, 200, 1)
            assert len(foreground) == count + 1
            assert not (app.viewport_menu and app.viewport_menu.shown) and not picks, theme
            assert (camera(), selection_and_publication(app)) == before
            # Positive control: an uncovered right-click opens the real popup.
            await click(450, 200, 1)
            assert app.viewport_menu.shown and ui.Menu.get_current() is app.viewport_menu
            assert not picks
            before = camera(), selection_and_publication(app)
            # Drag directly over a menu row; no camera or background picker.
            await move(470, 224)
            ui._ui._inject_mouse_button(1, True)
            await frames()
            await move(540, 228)
            ui._ui._inject_mouse_button(1, False)
            await frames()
            assert (camera(), selection_and_publication(app)) == before and not picks
            # Hide selected triggers via real left-click dispatch then closes.
            await click(480, 224, 0)
            assert {1, 2, 3} <= app.interaction.hidden and not picks
            assert (camera(), selection_and_publication(app)) == before
            await click(450, 200, 1)
            assert app.viewport_menu.shown
            await click(480, 256, 0)
            assert not app.interaction.hidden and not picks
            assert (camera(), selection_and_publication(app)) == before
            # Ordinary viewport left-click still reaches the native identity.
            await click(700, 500, 0)
            assert len(picks) == 1
            picks.clear()
            results.append({"theme": theme, "foreground_before_focus": True,
                            "popup_drag_blocked": True, "popup_actions": True})
        app._close_viewport_menu()
        overlay.destroy()
        app.window.destroy()
        await frames()
        print(json.dumps({"viewport_menu_dispatch": results}), flush=True)


if __name__ == "__main__":
    errors = []
    async def run():
        try: await verify()
        except Exception as error: errors.append(error)
        finally: ui.shutdown()
    isolate_gui_output()
    ui.init("Viewport menu ownership", width=800, height=600, max_fps=60)
    style.startup("dark")
    ui.run(run())
    if errors: raise errors[0]
