# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native events exercise Application.build; only GPU rendering is a sink."""
import os
import tempfile

import omni.ui as ui
from ovd_ui import style
from ovd_ui.application import Application
from ovd_ui.camera import Camera, CameraController
from gui_runtime import isolate_gui_output
from test_application_presentation import Renderer


async def verify(app):
    async def frames():
        for _ in range(3):
            await ui.next_frame()

    async def move(point):
        ui._ui._inject_mouse_move(*point)
        await frames()

    async def button(index, pressed):
        ui._ui._inject_mouse_button(index, pressed)
        await frames()

    def camera():
        return app.navigation.camera.view_matrix().tobytes()

    def center(widget):
        return (widget.screen_position_x + widget.computed_width / 2,
                widget.screen_position_y + widget.computed_height / 2)

    await frames()
    viewport = center(app.viewport_region)
    overlay = ui.Window("Camera drag foreground", width=120, height=90,
        flags=ui.WINDOW_FLAGS_NO_TITLE_BAR | ui.WINDOW_FLAGS_NO_MOVE | ui.WINDOW_FLAGS_NO_RESIZE)
    overlay.setPosition(viewport[0] - 200, viewport[1] - 90)
    clicks = []
    picks = []
    app.renderer.pick = lambda *args: picks.append(args)
    with overlay.frame:
        foreground = ui.Button("Foreground", clicked_fn=lambda: clicks.append("foreground"))
    await frames()
    points = (center(foreground), center(app.theme_button.button),
              center(app.browser_column), center(app.inspector_column))

    try:
        for theme in ("dark", "light"):
            style.set_theme(theme)
            app.theme = theme
            app.window.focus()
            await frames()
            for owner in (1, 2, 0):
                await move(viewport)
                await button(owner, True)
                assert app.mouse_anchor and app.mouse_anchor[2] == owner
                for point in points:
                    before = camera()
                    await move(point)
                    assert camera() != before, (theme, owner, "camera stopped at panel", point)
                # Releasing a drag on an actual button must not click it.
                await move(center(app.theme_button.button))
                await button(owner, False)
                before = camera()
                await move(viewport)
                assert camera() == before and app.mouse_anchor is None
                assert app.theme == theme and not clicks and app.viewport_menu is None

                # Escape over a panel cancels without restarting on re-entry.
                await button(owner, True)
                await move(points[0])
                ui._ui._inject_key_event(256, True)
                await frames()
                ui._ui._inject_key_event(256, False)
                before = camera()
                await move(viewport)
                await button(owner, False)
                assert camera() == before and app.mouse_anchor is None

                # A UI-origin drag never becomes camera input on re-entry.
                before = camera()
                await move(center(app.theme_button.button))
                await button(owner, True)
                await move(viewport)
                await button(owner, False)
                assert camera() == before and app.mouse_anchor is None

            # An unrelated release over a panel cannot end an active RMB drag.
            await move(viewport); await button(1, True)
            await move(points[2]); await button(2, True); await button(2, False)
            assert app.mouse_anchor and app.mouse_anchor[2] == 1 and app.viewport_keyboard_focus
            before = camera()
            await move(viewport)
            assert camera() != before
            await button(1, False)

            # Owner release and movement in one native dispatch must not move
            # the camera while a secondary button remains held.
            for owner in (0, 1):
                await move(viewport); await button(owner, True); await button(2, True)
                before = camera()
                ui._ui._inject_mouse_move(*points[2])
                ui._ui._inject_mouse_button(owner, False)
                await frames()
                await move(viewport); await button(2, False)
                assert camera() == before and app.mouse_anchor is None and not picks

            # Focus transfer cancels even if focus returns before the next move.
            await button(1, True)
            overlay.focus(); await frames()
            app.window.focus(); await frames()
            before = camera()
            await move(points[2]); await button(1, False)
            assert camera() == before and app.mouse_anchor is None

            # Ordinary controls still work, including a previously unfocused window.
            await move(center(foreground)); await button(0, True); await button(0, False)
            assert clicks == ["foreground"]
            clicks.clear()
            await move(center(app.theme_button.button)); await button(0, True); await button(0, False)
            assert app.theme != theme
        print("camera drag ownership: dark, light passed", flush=True)
    finally:
        overlay.destroy()


if __name__ == "__main__":
    errors = []
    isolate_gui_output()
    with tempfile.TemporaryDirectory() as temporary:
        os.environ["LOCALAPPDATA"] = os.environ["XDG_DATA_HOME"] = temporary
        ui.init("Camera drag regression", width=1600, height=1000, max_fps=60)
        style.startup("dark")
        app = Application(width=1600, height=1000)
        app.renderer = Renderer()
        app.renderer.provider = ui.ByteImageProvider()
        app.renderer.camera, app.renderer.size = Camera(), (1600, 1000)
        app.renderer.close = lambda: None
        app.navigation = CameraController(app.renderer.camera)
        app.build()
        app.window.set_focused_changed_fn(app._window_focused)

        async def run():
            try:
                await verify(app)
            except Exception as error:
                errors.append(error)
            finally:
                try:
                    app.close()
                finally:
                    ui.shutdown()

        ui.run(run())
    if errors:
        raise errors[0]
