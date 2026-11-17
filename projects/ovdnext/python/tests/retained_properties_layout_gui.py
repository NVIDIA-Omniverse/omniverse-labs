# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native layout regression; separate from mouse/keyboard application QA."""
from capture_semantics import test_scene_roles, multi_geom_roles, unique_instance_id, dynamic_geometry_owner
from gui_runtime import isolate_gui_output, hide_owned_window
from pathlib import Path
import struct
import tempfile

import omni.ui as ui
from _pvd_native import Session
from ovd_ui import style
from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.widgets.properties import PropertyInspector
from test_retained_properties import ValueView

ROOT = Path(__file__).resolve().parents[2]


class ObservedInspector(PropertyInspector):
    """Observe the real widgets built by each row without replacing their layout."""
    def __init__(self):
        super().__init__(lambda *_: None, lambda *_: None, {})
        self.labels, self.calls = {}, []

    def _build_property_row(self, adapter, identity, prop):
        factory = ui.Label
        labels = []
        def label(*args, **kwargs):
            result = factory(*args, **kwargs)
            labels.append(result)
            return result
        ui.Label = label
        try:result=super()._build_property_row(adapter, identity, prop)
        finally:ui.Label = factory
        self.labels[prop.name] = labels
        self.calls.append(prop.name)
        return result


async def verify():
    with tempfile.TemporaryDirectory() as temporary:
        source = Session(str(ROOT / "test/data/multi_geom.ovd"), temporary)
        adapter = None
        identity=dynamic_geometry_owner(source,2,30)
        selected = [identity]
        def publication(frame):
            result = PvdDataAdapter(source.presentation(frame=frame, inspect_ids=selected))
            result.schema[result.instances[identity].class_id] = [result.property(identity, name)
                for name in ("environmentID", "globalPose")]
            return result
        adapter = publication(30)
        target = ObservedInspector()
        window = ui.Window("Retained properties regression", width=640, height=650,
            flags=ui.WINDOW_FLAGS_NO_TITLE_BAR | ui.WINDOW_FLAGS_NO_RESIZE)
        with window.frame:
            with ui.HStack():
                outer = ui.Frame(width=320, height=620)
                with outer:
                    with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                        host = ui.Frame(build_fn=lambda: target.build(adapter, selected, retain=True))
                ui.Spacer()
        async def frames(count=5):
            for _ in range(count):await ui.next_frame()
        await frames()
        assert target.calls == ["environmentID", "globalPose"], target.calls
        scalar = target.labels["environmentID"][-1]
        old_pose = target.labels["globalPose"]
        row_frames = {key: value[0] for key, value in target._rows.items()}
        height = host.computed_content_height
        adapter = publication(31)
        host.rebuild()
        await frames()
        assert target.calls == ["environmentID", "globalPose"]
        assert host.computed_content_height == height and height > 100
        adapter = publication(32)
        host.rebuild()
        await frames()
        assert target.calls == ["environmentID", "globalPose"]
        assert target.labels["environmentID"][-1] is scalar and scalar.computed_height > 0
        assert target.labels["globalPose"] == old_pose
        assert any("7.629400" in label.text for label in target.labels["globalPose"])
        assert {key: value[0] for key, value in target._rows.items()} == row_frames
        retained = target.labels["globalPose"]
        sizes = []
        for theme, width in [("dark",240),("light",360),("dark",240)]:
            style.set_theme(theme)
            outer.width = ui.Pixel(width)
            await frames(10)
            assert outer.computed_width == width
            assert target.labels["globalPose"] == retained
            value = next(label for label in retained if "7.629400" in label.text)
            assert 0 < value.computed_width < width
            assert value.screen_position_x + value.computed_width <= outer.screen_position_x + width
            sizes.append(value.computed_height)
        assert sizes[0] > sizes[1] and sizes[0] == sizes[2], sizes
        assert target.calls == ["environmentID", "globalPose"]
        # A changed value can grow/shrink the same wrapped Label; checking only
        # equal-length native poses misses stale row-height layout caches.
        pose = adapter.property(identity, "globalPose")
        reader = adapter.session
        heights = []
        for coordinate in (1., 1e20, 1.):
            adapter.session = ValueView(reader, pose.id,
                struct.pack("<7f", 0., 0., 0., 1., coordinate, coordinate, coordinate))
            host.rebuild()
            await frames()
            assert target.labels["globalPose"] == retained
            heights.append(host.computed_content_height)
        assert heights[0] < heights[1] and heights[0] == heights[2], heights
        adapter.session = reader
        target.preferences["float_precision"] = 9
        host.rebuild()
        await frames()
        assert len(target.calls) == 4 and target.labels["environmentID"][-1] is not scalar
        selected = []
        host.rebuild()
        await frames()
        assert not target._rows and target._rows_key is None
        assert 0 < host.computed_content_height < height
        target.destroy()
        window.destroy()
        await frames(2)
        print("Native retained property checks passed", sizes, flush=True)


if __name__ == "__main__":
    errors = []
    async def run():
        try:await verify()
        except Exception as error:errors.append(error)
        finally:ui.shutdown()
    isolate_gui_output()
    ui.init("Retained properties regression", width=640, height=650, max_fps=60)
    hide_owned_window()
    style.startup("dark")
    ui.run(run())
    if errors:raise errors[0]
