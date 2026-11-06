# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused experiments and explanations for the right-hand learning panel."""
import os
import asyncio
import time
from pathlib import Path
import webbrowser

import omni.ui as ui
from .native_view import NativeView

from .content import ROOMS
from .extensions import call, value


def slider(lab, label, value, minimum, maximum, callback, key=None, model=None):
    from .style import MUTED
    lab._label(label, 12, MUTED)
    fresh = model is None
    model = ui.SimpleFloatModel(value) if fresh else model
    with ui.HStack(height=28, spacing=8):
        ui.FloatSlider(model, min=minimum, max=maximum, format='%.2f')
        ui.FloatDrag(model, min=minimum, max=maximum, step=.01, width=64, format='%.2f')
    if fresh:
        model.add_value_changed_fn(lambda m: callback(m.as_float))
    if key:
        lab.models[key] = model
    return model


def tabs(lab, key, options):
    from .style import MUTED, rgba
    selected = lab.panel_modes.setdefault(key, options[0][0])
    lab._label('EXPLORE', 11, MUTED)
    with ui.HStack(height=39, spacing=0):
        for value, name in options:
            with ui.VStack(spacing=0):
                lab.tab_buttons[value] = ui.Button(name, height=36, name='ActiveTab' if value == selected else 'Tab',
                          clicked_fn=lambda v=value: select_tab(lab, key, v))
                ui.Rectangle(height=3, style={'background_color':rgba(172,221,244) if value == selected else rgba(49,64,75)})
    return selected


def select_tab(lab, key, value):
    if lab.panel_modes.get(key) == value:
        return
    lab.panel_modes[key] = value
    if key == 'physics':
        from .physics_workshop import configure
        configure(lab, value)
    lab.request_panel()


async def snapshot(lab):
    view = lab.observer_view
    if not view:
        return
    path = lab.runtime_root/'inspection'
    path.mkdir(parents=True, exist_ok=True)
    file = path/f'inspection-{time.time_ns()}.png'
    lab.gallery.workshops.snapshot_status = 'Capturing camera image...'
    from PIL import Image
    if lab.sync_render:
        Image.fromarray(lab.native.render('Inspection')).save(file)
        lab.gallery.workshops.snapshot_status = 'Saved PNG in runtime / inspection.'
        return file
    # Wait for a newly presented inspection frame without blocking input.
    old = view.pixels
    deadline = time.monotonic()+60
    while view.pixels is None or view.pixels is old:
        if time.monotonic() > deadline:
            raise TimeoutError('Inspection camera did not produce an image. Try again.')
        await ui.next_frame()
    Image.fromarray(view.pixels.copy()).save(file)
    if not file.is_file():
        raise RuntimeError('The inspection camera image was not written')
    lab.gallery.workshops.snapshot_status = 'Saved PNG in runtime / inspection.'
    lab.notify('Inspection image saved.')
    return file


def start_snapshot(lab):
    if lab.snapshot_task and not lab.snapshot_task.done():
        return
    async def save():
        try:
            return await snapshot(lab)
        except Exception:
            lab.gallery.workshops.snapshot_status = 'Capture failed. Try Save PNG again.'
            lab.report_error('Saving the inspection image')
    lab.snapshot_task = asyncio.ensure_future(save())
    lab.notify('Capturing inspection image...', 60)


def build(lab, key):
    from .room_info import make_button, show
    from .style import rgba
    room=ROOMS[key]
    accent=rgba(*(int(c*255) for c in room['color']))
    with ui.HStack(height=47*lab.settings.text_scale,spacing=12):
        lab._label(room['name'],36,accent)
        with ui.VStack(width=36*lab.settings.text_scale):
            ui.Spacer(height=3*lab.settings.text_scale)
            lab.info_button=make_button(lab)
    # A background completion can rebuild the panel while the learning page is
    # open. Preserve that state so show() keeps the saved controls scroll.
    lab.showing_room_info=getattr(lab,'room_info_key',None)==key
    lab.info_frame=ui.Frame(visible=False)
    lab.controls_frame=ui.Frame()
    with lab.controls_frame:
        with ui.VStack(spacing=9):
            build_controls(lab,key)
    if getattr(lab,'room_info_key',None)==key:show(lab,True)


def build_controls(lab, key):
    from .style import rgba, MUTED, GREEN
    room, g = ROOMS[key], lab.gallery
    e, w = g.extra, g.workshops
    accent = rgba(*(int(c*255) for c in room['color']))
    lab._label(room['title'], 23)
    ui.Separator(height=2)
    if not lab.settings.allowed(key):
        from .experience import ROOM_RESTRICTIONS
        lab._label('EXPERIMENT UNAVAILABLE IN THIS BUILD', 13, MUTED)
        lab._label(ROOM_RESTRICTIONS.get(key, 'This experiment is not enabled.'), 17)
        lab._label('The equipment illustrates the intended scenario. No library experiment is running here.', 14, MUTED)
        lab._section('WHAT THIS INTEGRATION DEMONSTRATES')
        lab._label(room['use'], 14)
        ui.Button('Back to atrium', height=38, name='Primary', clicked_fn=lambda: lab.request_room('lobby'))
        return
    if call('build_panel', lab, key, accent):
        footer(lab, key)
        return

    if key == 'newton':
        from .extended_panels import build_newton
        build_newton(lab, accent)

    elif key == 'package':
        from .extended_panels import build_package
        build_package(lab, accent)

    elif key == 'physics':
        from .physics_panel import build as build_physics
        build_physics(lab, accent)

    elif key == 'rendering':
        car = w.render_subject == 'car'
        lab._section('SCENE SUBJECT / SWITCHES IMMEDIATELY')
        with ui.HStack(height=37, spacing=6):
            for value, label in (('lidar', 'Robot sensors'), ('car', 'Concept car'), ('robots', 'Robot arms')):
                lab.subject_buttons[value] = lab.choice(label, lambda: w.render_subject, value,
                    lambda v=value: lab.set_render_subject(v))
        if w.render_subject == 'lidar':
            from .lidar_panel import build as build_lidar
            build_lidar(lab, accent)
            footer(lab, key)
            return
        mode = tabs(lab, key, [('surface', 'Materials'), ('light', 'Lighting'), ('camera', 'Camera')])
        if mode == 'surface':
            if car:
                lab._label('Review a finish under the same studio lights.', 18)
                lab._label('Start with a paint color and finish. Watch the highlight widen from gloss to matte; tires, glass and cabin keep their own materials.', 14, MUTED)
                lab._section('PAINT COLOR / LIVE PREVIEW')
                with ui.HStack(height=33, spacing=5):
                    for name in ('Carmine', 'Ocean', 'Pearl'):
                        lab.choice(name, lambda: w.car.color, name, lambda n=name: w.car.paint(n))
                lab._section('FINISH')
                def finish(name):
                    w.car.finish(name)
                    lab.roughness = w.car.roughness
                with ui.HStack(height=36, spacing=5):
                    for title, value in [('Gloss', (.22,.32,1.)), ('Satin', (.48,.18,.35)), ('Matte', (.78,.08,0.))]:
                        lab.choice(title, lambda: (w.car.roughness,w.car.metallic,w.car.clearcoat), value, lambda n=title: finish(n))
                lab._label('Use Lighting to move the reflection across the bodywork. Camera lets you keep a reference for comparison.', 13, MUTED)
            else:
                lab._label('Compare coated castings with the bare-metal robot. Roughness spreads highlights across the surface.', 14)
                with ui.HStack(height=34, spacing=5):
                    for title, value in [('Polished', .07), ('Brushed', .55), ('Matte', .95)]:
                        lab.finish_buttons[title] = lab.choice(title, lambda: round(lab.roughness, 2), value, lambda v=value: lab.set_roughness(v))
            with ui.CollapsableFrame('Fine-tune material / optional', collapsed=True):
                with ui.VStack(spacing=10):
                    slider(lab, 'SURFACE ROUGHNESS', lab.roughness, .03, 1., lab.set_roughness, 'roughness')
                    metal = w.car.metallic if car else lab.stage.GetPrimAtPath('/World/Looks/inspection_alloy/Surface').GetAttribute('inputs:metallic').Get()
                    slider(lab, 'METAL RESPONSE / paint' if car else 'METAL RESPONSE / right-hand robot', metal, 0., 1., w.set_metallic, 'metallic')
                    if car:
                        slider(lab, 'CLEARCOAT', w.car.clearcoat, 0., 1., w.car.set_clearcoat, 'clearcoat')
        elif mode == 'light':
            lab._label('Move the key light across the cell. A larger emitter produces broader highlights and softer shadows; position changes where glare appears.', 14)
            with ui.HStack(height=34, spacing=6):
                for emitter in ('Rectangle', 'Disk', 'Sphere'):
                    lab.choice(emitter, lambda: w.emitter_type, emitter, lambda n=emitter: w.set_emitter(n))
            slider(lab, 'LIGHT ANGLE / degrees', w.light_angle, -85, 85, lambda v: w.set_light(angle=v), 'light_angle')
            slider(lab, 'EMITTER SIZE / metres', w.light_size, .25, 4., lambda v: w.set_light(size=v), 'light_size')
            slider(lab, 'KEY LIGHT INTENSITY / live', w.light_power, 500., 18000., lambda v: w.set_light(power=v), 'light_power')
            lab._section('LIGHTING PRESET / LIVE PREVIEW')
            with ui.VStack(height=66, spacing=4):
                lab.choice('Sharp highlights', lambda: (w.light_angle,w.light_size), (-25.,.3), lambda: w.set_light(angle=-25., size=.3))
                lab.choice('Broad illumination', lambda: (w.light_angle,w.light_size), (35.,3.5), lambda: w.set_light(angle=35., size=3.5))
            lab._label('Use the Camera tab to keep a reference image, change lighting, and compare the camera result.', 13, MUTED)
            lab.toggle('Key light enabled', lambda: lab.key_on, lambda value: lab.toggle_light())
            lab._section('LIGHT COLOR / LIVE PREVIEW')
            with ui.HStack(height=34, spacing=7):
                lab.choice('Warm', lambda: lab.warm_light, True, lab.toggle_light_color)
                lab.choice('Cool', lambda: lab.warm_light, False, lab.toggle_light_color)
        else:
            from .sensor_views import OUTPUTS, export
            lab._label('See what a camera can measure. These views come directly from OVRTX, from the same scene and viewpoint.', 15)
            entries = list(OUTPUTS.items())
            for row in (entries[:2], entries[2:]):
                with ui.HStack(height=32, spacing=8):
                    for label, output in row:
                        def select_output(o=output):
                            lab.native.sensor_output = o
                            lab.reference_pixels = None
                            lab.request_panel()
                        lab.choice(label, lambda: lab.native.sensor_output, output, select_output)
            lab.observer_view = NativeView(lab, 'Inspection', height=200)
            explanations = {
                'LdrColor': 'Color / rendered inspection image.',
                'DistanceToCameraSD': 'Distance from camera / BLUE 0 m  →  GREEN 10 m  →  RED 20 m+. Black means no valid return. Use for robot perception and range checks.',
                'NormalSD': 'Surface orientation / red X, green Y, blue Z. Use surface directions for inspection and geometry analysis.',
                'SemanticSegmentation': 'Object classes / CYAN car, AMBER robots, GRAY building. Labels are authored on the USD objects; OVRTX determines which object each pixel sees.'}
            lab._label(explanations[lab.native.sensor_output], 14)
            lab._section('CAMERA OPTICS / LIVE PREVIEW')
            def lens(fisheye):
                lab.native.sensor_fisheye = fisheye
                lab.reference_pixels = None
                lab.request_panel()
            with ui.HStack(height=34, spacing=7):
                lab.choice('Perspective', lambda: lab.native.sensor_fisheye, False, lambda: lens(False))
                lab.choice('Wide fisheye', lambda: lab.native.sensor_fisheye, True, lambda: lens(True))
            with ui.HStack(height=34, spacing=7):
                ui.Button('Keep reference', clicked_fn=lab.remember_view)
                ui.Button('Clear reference', enabled=lab.reference_pixels is not None,
                    clicked_fn=lambda: (setattr(lab, 'reference_pixels', None), lab.request_panel()))
            if lab.reference_pixels is not None:
                lab._label('REFERENCE / captured inspection pixels', 11, MUTED)
                lab.reference_view = NativeView(lab, 'Reference', height=150)
                lab.reference_view.update(lab.reference_pixels)
                lab._label('The live camera above responds to your changes; this reference stays fixed.', 13, MUTED)
            if not lab.native.sensor_fisheye:
                slider(lab, 'FOCAL LENGTH / mm', w.camera_zoom, 18, 65, w.set_camera_zoom, 'zoom')
                with ui.HStack(height=34, spacing=7):
                    for title, mm in [('18 mm', 18.), ('35 mm', 35.), ('65 mm', 65.)]:
                        lab.choice(title, lambda: w.camera_zoom, mm, lambda f=mm: w.set_camera_zoom(f))
            with ui.HStack(height=34, spacing=7):
                ui.Button('View from left', clicked_fn=lambda: w.camera_angle(False))
                ui.Button('View from right', clicked_fn=lambda: w.camera_angle(True))
            with ui.HStack(height=38):
                ui.Button('Save PNG', name='Primary', clicked_fn=lambda: start_snapshot(lab))
                ui.Button('Save raw data', clicked_fn=lambda: lab.queue_action('Saving sensor data',
                    lambda: export(lab.native, lab.runtime_root/'inspection'/('sensor-'+str(time.time_ns())))))
        lab.stats = lab._label('', 13, accent)
        if car:
            lab._label('Car Concept / Eric Chadwick, DGG / CC BY 4.0', 10, MUTED)

    elif key == 'scene':
        lab._label('Create multiple robots from one reusable asset. A shared cargo edit reaches every instance, while each robot keeps its own transform.', 14)
        lab._section('FLEET SIZE / UPDATES IMMEDIATELY')
        with ui.HStack(height=34, spacing=6):
            for n in (1, 3, 6):
                lab.choice(f'{n} AMR'+('s' if n > 1 else ''), lambda: w.fleet_count, n, lambda count=n: w.set_fleet(count))
        lab._section('TRY A SHARED SCENE UPDATE')
        with ui.HStack(height=37, spacing=7):
            ui.Button('Dispatch fleet', name='Primary', clicked_fn=lab.move_shared)
            ui.Button('Load / unload all', clicked_fn=lab.recolor_shared)
        lab.toggle('Automatic shuttle', lambda: lab.spin, lambda value: setattr(lab,'spin',value))
        lab._label('Vehicle travel is scripted. ovstage demonstrates shared scene updates and instancing.', 13, MUTED)
        slider(lab, 'SCRIPTED TRAVEL SPEED / m/s at single-fleet scale', g.shared_speed, .3, 2., lambda v: setattr(g, 'shared_speed', v), 'shared_speed')
        lab._label('LIVE OBSERVER  /  SAME STAGE, SECOND CAMERA', 10, accent)
        lab.observer_view = NativeView(lab, 'Fleet', height=172)
        lab.stats = lab._label('', 13, accent)

    elif key == 'storage':
        from .storage_panel import build as build_storage
        build_storage(lab, accent)

    elif key == 'interface':
        mode = tabs(lab, key, [('operate', 'Operate'), ('compose', 'Compose'), ('models', 'Models'), ('anchored', 'In scene')])
        if mode == 'anchored':
            lab._label('Controls beside the equipment. The floating slider and this panel share the same OVUI value model. Change either and watch both update.', 15)
        elif mode == 'compose':
            lab._label('Choose the widgets for this operator panel. The application assembles the layout immediately; each control is wired to the scene.', 14)
            for widget, title in [('jog', 'Raise / lower button'), ('quality', 'Quality selector'), ('light', 'Work-light toggle')]:
                with ui.HStack(height=27):
                    model = ui.SimpleBoolModel(w.ui_widgets[widget])
                    ui.CheckBox(model, width=27)
                    lab._label(title, 13)
                    model.add_value_changed_fn(lambda m, k=widget: (w.ui_widgets.__setitem__(k, m.as_bool), lab.operator_frame.rebuild()))
        elif mode == 'models':
            lab._label('The slider and number field below share one value model. Edit either one: both widgets and the scene update together. The cycle also writes to that model.', 14)
        else:
            lab._label('This is a live operator panel made with ready-made UI widgets. Raise the load, change its QC state and control the work light.', 14)
        lab.operator_frame = ui.Frame()
        with lab.operator_frame:
            operator(lab, mode)
        lab.operator_frame.set_build_fn(lambda: operator(lab, mode))
        lab.stats = lab._label('', 13, accent)

    elif key == 'streaming':
        from .native_view import StreamView
        lab._label('RTX camera  >  ovstream server  >  client', 12, accent)
        lab.stream_view = StreamView(lab)
        lab.stream_controls['connect'] = ui.Button('Connect camera feed', height=38, name='Primary', clicked_fn=lambda: lab.queue_action('Connecting camera feed', lab.stream_view.connect))
        lab.stream_controls['pause'] = lab.toggle('Pause incoming video', lambda: w.stream_paused, lambda value: setattr(w,'stream_paused',value))
        lab.stream_hint = lab._label('', 13, MUTED)
        lab._section('CLIENT COMMANDS / AVAILABLE WHEN CONNECTED')
        with ui.HStack(height=35, spacing=7):
            lab.stream_controls['dispatch'] = ui.Button('Dispatch AMR', clicked_fn=lambda: lab.stream_view.send('dispatch'))
            lab.stream_controls['cargo'] = ui.Button('Load / unload', clicked_fn=lambda: lab.stream_view.send('cargo'))
        lab._section('TARGET FRAME RATE / APPLIES IMMEDIATELY')
        with ui.HStack(height=30, spacing=7):
            lab.choice('5 fps target', lambda: w.stream_fps, 5, lambda: setattr(w, 'stream_fps', 5))
            lab.choice('15 fps target', lambda: w.stream_fps, 15, lambda: setattr(w, 'stream_fps', 15))
        lab.stats = lab._label('Ready to connect', 13, accent)
        lab._label('Pause the feed while the robot moves: the client holds its last image, while the scene keeps running. Resume to receive fresh frames.', 14)
        # Startup prepared this connection. Entering on foot must not queue an
        # action (which clears held movement keys) or initialize native resources.
        if lab.stream_transport is not None:
            lab.stream_view.connect()

    footer(lab, key)


def footer(lab, key):
    from .style import MUTED
    room = ROOMS[key]
    ui.Separator(height=2)
    with ui.HStack(height=28*lab.settings.text_scale, spacing=8):
        ui.Button('Reset room', name='Utility',
                  clicked_fn=lambda: lab.queue_action('Resetting room', lab.reset_room))
        route = [*[key for key in ROOMS if lab.settings.allowed(key)], 'lobby']
        next_key = route[route.index(key)+1]
        ui.Button('Atrium' if next_key == 'lobby' else 'Next room >', name='Utility',clicked_fn=lambda: lab.request_room(next_key))


def operator(lab, mode):
    from .style import MUTED, GREEN
    e, w = lab.gallery.extra, lab.gallery.workshops
    with ui.VStack(spacing=9):
        lab._label('YOUR OPERATOR PANEL', 11, GREEN)
        if not getattr(lab, 'fork_model', None):
            lab.fork_model = ui.SimpleFloatModel(e.lift_value)
            lab.fork_model.add_value_changed_fn(lambda m: e.set_lift(m.as_float))
        lab.lift_model = slider(lab, 'FORK HEIGHT / shared value 0-1', e.lift_value, 0, 1, e.set_lift, model=lab.fork_model)
        if w.ui_widgets['jog']:
            ui.Button('Raise / lower forks', height=36, name='Primary', clicked_fn=lambda: lab.lift_model.set_value(.85 if e.lift_value < .5 else .15))
            lab.toggle('Cycle fork height', lambda: w.lift_cycle, lambda value: setattr(w,'lift_cycle',value))
        if w.ui_widgets['quality']:
            with ui.HStack(height=29, spacing=7):
                lab._label('QC status', 12, MUTED)
                combo = ui.ComboBox(e.part_color, 'Awaiting inspection', 'Hold for review', 'Approved', width=218)
                lab.qc_combo = combo
                combo.model.add_item_changed_fn(lambda model, item: e.set_quality(model.get_item_value_model().as_int))
        if w.ui_widgets['light']:
            with ui.HStack(height=25):
                model = ui.SimpleBoolModel(e.work_light_on)
                ui.CheckBox(model, width=27)
                lab._label('Work light', 13)
                model.add_value_changed_fn(lambda m: e.toggle_work_light() if m.as_bool != e.work_light_on else None)


def show_example(lab, key):
    from .style import STYLE, MUTED
    examples = Path(__file__).resolve().parents[3]/'scripts/examples'
    filename = {'physics': 'physics_drop.py', 'rendering': 'render_materials.py',
                'scene': 'shared_stage.py', 'storage': 'storage_revisions.py', 'interface': 'operator_ui.py',
                'streaming': 'stream_roundtrip.py',
                **value('EXAMPLES', {}), 'newton': 'gantry_newton.py',
                'package': 'asset_package.py'}.get(key)
    if key == 'rendering' and lab.gallery.workshops.render_subject == 'lidar':
        filename = 'lidar_points.py'
    path = examples/(filename or 'README.md')
    if getattr(lab, 'example_window', None):
        lab.example_window.destroy()
    width = min(790, ui.Workspace.get_main_window_width()-40)
    height = min(650, ui.Workspace.get_main_window_height()-60)
    window = lab.example_window = ui.Window('Build with this library', width=width, height=height,
        position_x=max(20, (ui.Workspace.get_main_window_width()-width)/2),
        position_y=max(30, (ui.Workspace.get_main_window_height()-height)/2))
    window.frame.style = STYLE
    code = path.read_text(encoding='utf-8') if path.is_file() else 'See docs/DEVELOPMENT.md for the library integration.'
    def copy_code():
        from omni.ui import _ui
        _ui._set_clipboard_text(code)
        lab.notify('Example copied to clipboard.')
    with window.frame:
        with ui.VStack(spacing=10):
            lab._label('From experiment to your application', 23)
            interpreter = key+'-env' if key in ('newton','package') else 'libraries-env'
            lab._label(f'Run this example with {interpreter}/Scripts/python.exe in your runtime. See scripts/examples/README.md for commands and expected results.', 14, MUTED)
            with ui.HStack(height=34, spacing=8):
                lab._label(path.name, 15)
                lab.example_copy = ui.Button('Copy code', width=130, clicked_fn=copy_code)
            with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED):
                font = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts/consola.ttf'
                style = {'font_size': 15*lab.settings.text_scale}
                if font.is_file():
                    style['font'] = str(font).replace('\\', '/')
                ui.Label(code, word_wrap=False, alignment=ui.Alignment.LEFT_TOP, style=style)
            ui.Button('Back to the experiment', height=36, clicked_fn=lambda: setattr(window, 'visible', False))
