# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A guided native comparison, with manual experiment controls kept optional."""
import omni.ui as ui
from .style import MUTED


def build(lab, accent):
    from .panels import tabs, slider
    from .physics_workshop import configure, nudge
    mode = tabs(lab, 'physics', [('drop', 'Drop'), ('stack', 'Stack'), ('slide', 'Slide')])

    def playback():
        with ui.HStack(height=39, spacing=8):
            lab.release_button = ui.Button('Pause' if lab.running else 'Run experiment', name='Primary',
                clicked_fn=lambda: lab.queue_action('Updating experiment', lab.toggle_physics))
            ui.Button('Reset', width=78, clicked_fn=lab.reset_physics)

    def speed():
        slider(lab, 'PLAYBACK SPEED', lab.physics_speed, .2, 1.,
               lambda v: setattr(lab, 'physics_speed', v), 'physics_speed')

    if mode == 'slide':
        lab._label('Compare stopping distance on the receiving belt.', 18)
        lab._label('Measure low friction first, then high friction to compare. Each starts the same six cartons at 1.15 m/s and measures two simulated seconds.', 14, MUTED)
        with ui.VStack(height=82, spacing=6):
            for friction, title in ((.05, 'Run low friction'), (.65, 'Run high friction')):
                lab.comparison_buttons[friction] = ui.Button(title, name='Primary' if friction == .05 else '',
                    clicked_fn=lambda f=friction: lab.queue_action('Measuring carton travel', lambda: lab.compare_friction(f)))
        lab._section('MEASURED TRAVEL / AVERAGE OF SIX CARTONS')
        for friction, title in ((.05, 'Low friction'), (.65, 'High friction')):
            value = lab.physics_results.get(friction)
            lab._label(title+('  /  ready to measure' if value is None else f'  /  {value:.2f} m'), 17, accent)
        if all(f in lab.physics_results for f in (.05, .65)):
            low, high = (lab.physics_results[f] for f in (.05, .65))
            lab._label(f'The higher-friction surface reduced travel by {max(0., low-high):.2f} m in this run.', 14)
        with ui.CollapsableFrame('Manual experiment / optional', collapsed=True):
            with ui.VStack(spacing=9):
                with ui.HStack(height=34, spacing=7):
                    lab.choice('Low friction', lambda: lab.friction, .05, lambda: configure(lab, friction=.05))
                    lab.choice('High friction', lambda: lab.friction, .65, lambda: configure(lab, friction=.65))
                playback()
                speed()
    else:
        lab._label('Choose a transfer height, then release the cartons onto the receiving belt.' if mode == 'drop'
                   else 'Settle a pallet load, then nudge its top layer to see how contacts support the stack.', 15)
        if mode == 'drop':
            lab._section('TRANSFER HEIGHT / SETS UP THE NEXT RUN')
            with ui.HStack(height=34, spacing=7):
                lab.choice('Low / 1.0 m', lambda: lab.high_drop, False, lambda: lab.set_drop_height(False))
                lab.choice('High / 2.1 m', lambda: lab.high_drop, True, lambda: lab.set_drop_height(True))
            lab._section('GRAVITY / RESETS THE EXPERIMENT')
            with ui.HStack(height=34, spacing=7):
                lab.choice('Earth gravity', lambda: lab.moon, False, lambda: lab.set_gravity(False))
                lab.choice('Low gravity', lambda: lab.moon, True, lambda: lab.set_gravity(True))
        lab._section('RUN THE EXPERIMENT')
        playback()
        if mode == 'stack':
            lab.stack_nudge = ui.Button('Nudge the top cartons', height=35, enabled=lab.sim_started,
                      clicked_fn=lambda: lab.queue_action('Nudging the load', lambda: nudge(lab)))
            lab._label('Run the experiment to settle the stack, then nudge the top cartons.', 13, MUTED)
        lab._label('Left: static layout. Right: actual rigid-body simulation. The belt surface is also the collision surface.', 13, MUTED)
        with ui.CollapsableFrame('Slow motion / optional', collapsed=True):
            with ui.VStack(spacing=8):
                speed()
    from .physics_inspection import show, set_mass
    with ui.CollapsableFrame('Inspect physics / shapes and test load', collapsed=True):
        with ui.VStack(spacing=10):
            lab.toggle('Show collision outlines', lambda: lab.gallery.show_colliders,
                       lambda enabled: show(lab.gallery, enabled))
            lab._section('MASS PER CARTON / RESETS THE EXPERIMENT')
            with ui.HStack(height=34, spacing=6):
                for mass in (.5, 2., 10.):
                    lab.choice(f'{mass:g} kg', lambda: lab.carton_mass, mass, lambda m=mass: set_mass(lab, m))
    lab.stats = lab._label('', 13, accent)
