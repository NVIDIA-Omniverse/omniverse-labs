# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clear actions, native-operation feedback and concise capability explanations."""
import time
import omni.ui as ui
from .style import MUTED
from .package_flow import describe, PROFILES


def build_newton(lab, accent):
    room = lab.gallery.newton_room
    lab.extended_ui = widgets = {}
    lab._section('1  /  CHOOSE A MOTION PROFILE')
    lab._label('Move the same 35 kg load with two drive profiles. Watch the rigid suspension link swing as the trolley accelerates and stops.', 15)
    with ui.HStack(height=38, spacing=7):
        for mode, title in (('gentle','Gentle / 4.5 s'),('fast','Fast / 1.25 s')):
            widgets[mode] = lab.choice(title, lambda: room.mode, mode, lambda m=mode: setattr(room,'mode',m))
    lab._label('Selection sets up your next run. It does not start the motion.', 13, MUTED)
    lab._section('2  /  RUN AND OBSERVE')
    with ui.HStack(height=40, spacing=7):
        widgets['run'] = ui.Button('Run transfer', name='Primary', clicked_fn=room.run)
        def playback():
            if room.result and not room.busy:
                if room.elapsed >= 8.:
                    room.elapsed = 0.
                room.playing = not room.playing
        widgets['play'] = ui.Button('Pause', width=102, clicked_fn=playback)
    widgets['status'] = lab._label(room.status, 14, accent)
    widgets['live'] = lab._label('', 15)
    lab._section('RESULT  /  COMPARE LOAD SWAY')
    widgets['comparison'] = lab._label('Run both profiles to compare peak load sway.', 15)
    widgets['error'] = lab._label('', 13, MUTED)
    refresh(lab)


def build_package(lab, accent):
    room = lab.gallery.package_room
    lab.extended_ui = widgets = {}
    lab._label('Package a robot once. Deliver the same asset and its dependencies to another workstation.', 17)
    widgets['step'] = lab._label('', 19, accent)
    widgets['hint'] = lab._label('', 15)
    lab._label('Source robot / changes apply to the left station', 13, MUTED)
    with ui.VStack(height=64, spacing=2):
        for value, title in PROFILES.items():
            widgets[value] = lab.choice(title, lambda: room.profile, value, lambda v=value: room.choose(v))
    widgets['source'] = lab._label('', 13, MUTED)
    def advance():
        action = describe(room)['next']
        if action:
            room.submit(action)
    widgets['primary'] = ui.Button('', height=44, name='Primary', clicked_fn=advance)
    widgets['target'] = lab._label('', 14)
    widgets['result'] = lab._label('', 15)
    widgets['status'] = lab._label(room.status, 14, accent)
    with ui.CollapsableFrame('More / published revisions', collapsed=True) as history:
        widgets['history'] = history
        with ui.VStack(spacing=9):
            lab._label('Each publication is a reusable, versioned asset. Select an older revision to install it again.', 14)
            with ui.HStack(height=34, spacing=7):
                widgets['previous'] = ui.Button('< Older revision', clicked_fn=lambda: room.select(-1))
                widgets['next'] = ui.Button('Newer revision >', clicked_fn=lambda: room.select(1))
            widgets['publish'] = ui.Button('Publish another revision', height=36, clicked_fn=lambda: room.submit('publish'))
            widgets['install'] = ui.Button('Install selected revision', height=36, clicked_fn=lambda: room.submit('install'))
            widgets['verify'] = ui.Button('Check installed files again', height=35, clicked_fn=lambda: room.submit('verify'))
    with ui.CollapsableFrame('Optional / edit and repair the installed asset', collapsed=True) as repair:
        widgets['repair'] = repair
        with ui.VStack(spacing=9):
            lab._label('After installation, remove the payload on the right. The file check detects the edit. Restore a published revision to repair it.', 14)
            widgets['edit'] = ui.Button('1  Remove installed payload', height=37, clicked_fn=lambda: room.submit('edit'))
            widgets['restore'] = ui.Button('2  Restore selected revision', height=37, clicked_fn=lambda: room.submit('restore'))
            widgets['repair_hint'] = lab._label('', 13, MUTED)
    widgets['error'] = lab._label('', 13, MUTED)
    refresh(lab)


def refresh(lab):
    widgets = getattr(lab,'extended_ui',{})
    if not widgets:
        return
    key = lab.current_room
    room = lab.gallery.newton_room if key == 'newton' else lab.gallery.package_room
    ready = bool(room.worker and room.worker.ready and not room.worker.error)
    unavailable = room.worker.error if room.worker else room.error
    can_act = ready and not room.busy
    spinner = '|/-\\'[int(time.monotonic()*5)%4]
    widgets['status'].text = 'Library unavailable. See the details below.' if unavailable else room.status if ready or room.error else 'Preparing the native library...'
    widgets['error'].text = (room.error or (room.worker.error if room.worker else '') or '')[:350]
    if key == 'newton':
        for mode in ('gentle','fast'):
            widgets[mode].enabled = can_act
        widgets['run'].enabled = can_act
        widgets['run'].name = 'Busy' if room.busy else 'Primary'
        widgets['run'].text = f'{spinner} Computing motion...' if room.busy else f'Run {room.mode} transfer'
        widgets['play'].enabled = can_act and room.result is not None
        widgets['play'].text = 'Pause' if room.playing else 'Replay' if room.elapsed>=8. else 'Resume'
        if room.result:
            if not room.playing and room.elapsed<8.:
                widgets['status'].text = 'Motion paused. Resume or run another profile.'
            index = min(round(room.elapsed*room.result['fps']),len(room.result['angles'])-1)
            widgets['live'].text = f"Showing {room.result['mode']} / {room.elapsed:.1f} s of 8.0 s\nLoad sway {abs(room.result['angles'][index]):.1f}°"
        else:
            widgets['live'].text = '3.2 m transfer / 35 kg rigid load / 120 Hz solver'
        widgets['comparison'].text = '\n'.join(
            f"{mode.title()}: {room.comparisons[mode]:.1f}° peak sway" if mode in room.comparisons else f'{mode.title()}: run to measure'
            for mode in ('gentle','fast'))
    else:
        flow = describe(room)
        action = flow['next']
        widgets['step'].text = {'publish': '1 / Configure and publish', 'install': '2 / Install at the receiving station',
                                'restore': 'Repair the installed asset', None: 'Complete / Asset delivered'}[action]
        widgets['primary'].text = (f'{spinner} Working...' if room.busy else
            {'publish': 'Publish this robot', 'install': 'Install at the right station', 'restore': 'Restore published asset', None: 'Installed and verified'}[action])
        widgets['primary'].enabled = can_act and action is not None
        widgets['primary'].name = 'Busy' if room.busy else 'Primary' if action else ''
        for value in ('logistics','inspection'):
            widgets[value].enabled = can_act
        selected = flow['selected']
        for name in ('hint','source','target','result'):
            widgets[name].text = flow[name]
        if unavailable:
            widgets['hint'].text = 'Library unavailable. See the details below.'
        elif not ready:
            widgets['hint'].text = 'Preparing the library. Controls will become available when ready.'
        elif room.busy:
            widgets['hint'].text = room.status
        widgets['previous'].enabled = can_act and room.selected > 0
        widgets['next'].enabled = can_act and room.selected < len(room.versions)-1
        widgets['repair_hint'].text = ('Install a revision first to try this exercise.' if not room.installed else
                                      'Changed USD detected. Restore the selected revision next.' if flow['modified'] else
                                      'Ready. Remove the payload to create a detectable file change.')
        if flow['modified']:
            widgets['repair'].collapsed = False
        titles = {'publish':'Publish source revision',
                  'install':'Installed & verified' if flow['installed'] else 'Install selected revision',
                  'verify':'Check installed files again', 'edit':'1  Remove installed payload',
                  'restore':'2  Restore selected revision'}
        allowed = {'publish': True, 'install': bool(selected) and not flow['installed'] and not flow['modified'],
                   'verify': bool(room.installed), 'edit': bool(room.installed) and not flow['modified'],
                   'restore': bool(selected) and flow['modified']}
        for name,title in titles.items():
            busy = room.busy and room.action==name
            widgets[name].text = f'{spinner} Working...' if busy else title
            widgets[name].name = 'Busy' if busy else 'Primary' if name==flow['next'] else ''
            widgets[name].enabled = can_act and allowed[name]
