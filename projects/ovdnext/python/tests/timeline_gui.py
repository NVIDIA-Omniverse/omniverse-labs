# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated public-OVUI timeline QA harness (Inspector screenshots and gestures).

DISPLAY=:102 OVUIINSPECT_PORT=9915 OVUIINSPECT_ENABLE_STATE=1 \
PYTHONPATH=python:_ci/deps/ovui/skills/omniverse-ui-inspector \
_ci/venv/bin/python python/tests/timeline_gui.py
"""
import omni.ui as ui
import os
if os.environ.get('OVUIINSPECT_ENABLED') == '1':
    import ovuiinspect
else:
    ovuiinspect = None
import sys

from ovd_ui import style
from ovd_ui.playback import Playback
from ovd_ui.widgets.timeline import SoftGlassTimeline
from gui_runtime import isolate_gui_output


class TimelineHarness:
    def __init__(self, verify=False):
        self.verify, self.running = verify, True
        self.error = None
        self.theme = "dark"
        self.playback = Playback()
        self.playback.reset(201, ovd=True)
        self.playback.clip.start, self.playback.clip.stop = 20, 181
        self.playback.seek(70)
        self.model = ui.SimpleIntModel(70)
        self.suppress = False
        self.requests = []
        self.ready = True

    def changed(self, value):
        if self.suppress:
            return
        self.requests.append(value.as_int)
        selected = self.playback.seek(value.as_int) if self.ready else self.playback.frame
        self.suppress = True
        self.model.set_value(selected)
        self.suppress = False
        self.label.text = f"Published frame {selected} · request {value.as_int} · Post-sim"

    def toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        style.set_theme(self.theme)

    def toggle_ready(self):
        self.ready = not self.ready
        self.ready_button.text = "Producer ready" if self.ready else "Producer unpublished"

    def publish_model_probe(self, frame):
        # Exercise the widget's publication display without the harness's
        # normal seek/parity clamp. This does not simulate a native receiver.
        self.suppress = True
        self.model.set_value(frame)
        self.suppress = False
        self.label.text = f"Model frame {frame} · publication display probe"

    async def verify_layout(self):
        for _ in range(5):
            await ui.next_frame()
        assert self.ruler.widget.computed_width == 650
        assert self.narrow.widget.computed_width == 30
        assert self.narrow.input.computed_width == 30
        # Frame updates retain the static ruler and replace only the playhead.
        # Consecutive equal-width labels must still move the native marker.
        ruler, cursor = self.ruler._ruler, self.ruler.cursor
        self.suppress = True
        for frame in (0, 9, 10, 99, 100, 101, 102, 150, 152, 153, 200, 201, -1, 70):
            self.model.set_value(frame)
            # The next completed UI frame must show this publication. Waiting
            # two frames masks a lazy-build layer that trails live playback.
            await ui.next_frame()
            assert self.ruler._ruler is ruler and self.ruler.cursor is cursor
            visible = 0 <= frame <= 200
            assert (self.ruler._marker is not None) == (self.ruler._badge is not None) == visible
            if visible:
                scale = self.ruler._scale()
                expected = self.ruler.widget.screen_position_x + max(1, min(647, scale.position(frame)-1))
                assert abs(self.ruler._marker_shape.screen_position_x-expected) <= 1
                assert self.ruler._badge_label.text == str(frame)
                assert self.ruler._badge_content.computed_width == max(18, len(str(frame))*6+12)
        self.suppress = False
        assert self.requests == []
        self.ruler.width = ui.Pixel(230)
        for _ in range(5):
            await ui.next_frame()
        assert self.ruler.widget.computed_width == 230
        assert self.ruler.input.computed_width == 230
        assert self.ruler.get_inspector_state()["ticks"] == [0, 100, 200]
        assert self.requests == [] and self.model.as_int == 70
        # A smaller rebuilt ruler must retain working playhead updates,
        # including hide/show and label shrink after switching theme.
        self.suppress = True
        for frame in (150, 152, 201, 70, 72):
            self.model.set_value(frame)
            await ui.next_frame()
            if frame <= 200:
                expected = self.ruler.widget.screen_position_x + self.ruler._scale().position(frame)-1
                assert abs(self.ruler._marker_shape.screen_position_x-expected) <= 1
                assert self.ruler._badge_label.text == str(frame)
                assert self.ruler._badge_content.computed_width == max(18, len(str(frame))*6+12)
            for _ in range(5):
                await ui.next_frame()
            if frame == 152:
                self.toggle_theme()
                for _ in range(5):
                    await ui.next_frame()
        self.model.set_value(70)
        self.suppress = False
        # Native mouse callbacks happen during drawing, unlike the model
        # publications above. Check the real hide/restore button path as well.
        self.publish_model_probe(152)
        for _ in range(5):
            await ui.next_frame()
        for button, frame in ((self.outside_button, 201), (self.restore_button, 70)):
            ui._ui._inject_mouse_move(button.screen_position_x + button.computed_width/2,
                                      button.screen_position_y + button.computed_height/2)
            ui._ui._inject_mouse_button(0, True)
            await ui.next_frame()
            ui._ui._inject_mouse_button(0, False)
            await ui.next_frame()
            await ui.next_frame()  # First layout after the release callback.
            assert self.model.as_int == frame
            assert (self.ruler._marker is not None) == (frame == 70)
            if frame == 70:
                expected = self.ruler.widget.screen_position_x + self.ruler._scale().position(frame)-1
                assert abs(self.ruler._marker_shape.screen_position_x-expected) <= 1
                assert self.ruler._badge_content.computed_width == 24
        assert self.requests == []
        self.ruler.width = ui.Pixel(650)
        for _ in range(5):
            await ui.next_frame()
        assert self.ruler.widget.computed_width == 650
        assert self.ruler.input.computed_width == 650
        self.ruler.min, self.ruler.max = 70, 100
        self.ruler.set_domain(0, 100)
        for _ in range(5):
            await ui.next_frame()
        assert self.ruler.get_inspector_state()["clip"] == (70, 100)
        assert self.requests == [] and self.model.as_int == 70
        self.running = False
        print("Timeline layout checks passed", flush=True)

    def get_inspector_state(self):
        return {"timeline": self.ruler.get_inspector_state(),
                "requested_width": str(self.ruler.widget.width),
                "frame": self.playback.frame, "requests": self.requests[-15:],
                "ready": self.ready, "theme": self.theme,
                "single": self.single.get_inspector_state(),
                "narrow": self.narrow.get_inspector_state()}

    async def run(self):
        flags = ui.WINDOW_FLAGS_NO_TITLE_BAR | ui.WINDOW_FLAGS_NO_RESIZE | ui.WINDOW_FLAGS_NO_MOVE
        self.window = ui.Window("Timeline QA", width=1000, height=440, flags=flags)
        self.window.position_x = self.window.position_y = 0
        with self.window.frame:
            with ui.VStack(spacing=16):
                ui.Spacer(height=8)
                ui.Label("Soft Glass timeline · capture 0–200 · clip 20–180", height=30)
                with ui.HStack(height=30, spacing=8):
                    ui.Button("Theme", width=90, clicked_fn=self.toggle_theme)
                    ui.Button("Narrow", width=90, clicked_fn=lambda: setattr(self.ruler, "width", ui.Pixel(230)))
                    ui.Button("Wide", width=90, clicked_fn=lambda: setattr(self.ruler, "width", ui.Pixel(650)))
                    self.ready_button = ui.Button("Producer ready", width=180, clicked_fn=self.toggle_ready)
                    self.outside_button = ui.Button("Outside domain", width=125,
                        clicked_fn=lambda: self.publish_model_probe(201))
                    self.restore_button = ui.Button("Restore model", width=125,
                        clicked_fn=lambda: self.publish_model_probe(70))
                    ui.Button("Quit", width=70, clicked_fn=lambda: setattr(self, "running", False))
                    ui.Spacer()
                with ui.HStack(height=44, spacing=12):
                    ui.Spacer(width=20)
                    self.ruler = SoftGlassTimeline(self.model, min=20, max=180, width=650)
                    self.ruler.set_domain(0, 200)
                    ui.IntField(self.model, name="mono", width=70, height=32)
                    ui.Label("CLIP BOUNDARY", width=140)
                    ui.Spacer()
                self.label = ui.Label("Published frame 70 · Post-sim", height=24)
                self.model.add_value_changed_fn(self.changed)
                with ui.HStack(height=44, spacing=12):
                    ui.Spacer(width=20)
                    self.single = SoftGlassTimeline(ui.SimpleIntModel(7), min=7, max=7, width=230)
                    ui.Label("One frame", width=100)
                    self.narrow = SoftGlassTimeline(ui.SimpleIntModel(123456789), min=123456789,
                        max=123456789, width=30)
                    ui.Label("Narrow badge must stay clipped", width=240)
                    ui.Spacer()
                ui.Label("Left click/drag: seek. Model callback applies post-sim parity.\n"
                         "Toggle producer state to verify rejected seeks retain the complete prior playhead.", height=50)
                ui.Spacer()
        try:
            if self.verify:
                await self.verify_layout()
            while self.running:
                if ovuiinspect is not None:
                    ovuiinspect.drain_pending(ui._ui, application=self)
                await ui.next_frame()
        except Exception as error:
            self.error = error
        finally:
            for ruler in (self.ruler, self.single, self.narrow):
                ruler.destroy()
            ui.shutdown()


if __name__ == "__main__":
    isolate_gui_output()
    ui.init("Timeline QA", width=1000, height=440, max_fps=60)
    style.startup("dark")
    application = TimelineHarness(verify="--verify" in sys.argv)
    if ovuiinspect is not None:
        ovuiinspect.attach_application(application)
    ui.run(application.run())
    if application.error:
        raise application.error
