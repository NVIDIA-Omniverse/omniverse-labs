# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A single next action teaches save, change and restore using real SDK results."""
import omni.ui as ui
from .style import MUTED

NAMES = ('Empty rack', 'Pick-ready', 'Replenished')


def names(lab):
    return NAMES if lab.gallery.factory.simready.assets else ('2 shelf levels', '3 shelf levels', '4 shelf levels')


def next_step(storage):
    e = storage.e
    if not storage.guided_saved or not e.revisions:
        return 'save', '1 / Save a baseline', 'Choose the configuration on the left, then save it. The right rack will show the saved snapshot.', 'Save this rack'
    if e.storage_variant != e.saved_variant:
        return 'restore', '3 / Restore the saved rack', 'The working rack differs from the saved snapshot. Restore it and watch the left rack match the right again.', 'Restore saved rack'
    if storage.guided_restored:
        return None, 'Complete / Snapshot restored', 'The saved USD was read back through OVStorage. Try another configuration, or save a new baseline.', 'Snapshot restored'
    return 'change', '2 / Change the working rack', 'Your baseline is safe on the right. Change the configuration on the left to see what restoring a snapshot will do.', 'Change working rack'


def advance(lab):
    action = next_step(lab.storage)[0]
    if action == 'change':
        e = lab.gallery.extra
        e.set_asset_variant((e.storage_variant+1)%3)
    elif action:
        lab.storage.submit(action)


def build(lab, accent):
    e = lab.gallery.extra
    lab._label('Keep a safe snapshot of your scene, change it, then restore it.', 17)
    lab._label('LEFT / working rack     RIGHT / saved snapshot', 14, MUTED)
    lab.storage_step = lab._label('', 19, accent)
    lab.storage_hint = lab._label('', 15)
    with ui.VStack(height=96, spacing=2):
        for value, name in enumerate(names(lab)):
            lab.storage_controls[f'levels{value+2}'] = lab.choice(name, lambda: e.storage_variant, value,
                lambda n=value: e.set_asset_variant(n))
    lab.storage_primary = ui.Button('', height=44, name='Primary', clicked_fn=lambda: advance(lab))
    lab.stats = lab._label('', 15, accent)
    lab.detail = lab._label('', 14, MUTED)
    with ui.CollapsableFrame('More / revision history and editing', collapsed=True):
        with ui.VStack(spacing=10):
            lab._label('Save another baseline, or inspect earlier snapshots. Browsing changes the right rack; restoring changes the left.', 14)
            lab.storage_save_button = ui.Button('Save a new baseline', height=36, clicked_fn=lambda: lab.storage.submit('save'))
            lab.revision_label = lab._label('', 14, accent)
            with ui.HStack(height=34, spacing=7):
                for key, title, step in [('previous', '< Older', -1), ('next', 'Newer >', 1)]:
                    lab.storage_controls[key] = ui.Button(title, clicked_fn=lambda s=step: lab.storage.submit('select', s))
            lab.storage_controls['restore'] = ui.Button('Restore this snapshot', height=36, clicked_fn=lambda: lab.storage.submit('restore'))
            with ui.HStack(height=34, spacing=7):
                lab.storage_controls['undo'] = ui.Button('Undo working edit', clicked_fn=e.undo_asset)
                lab.storage_controls['redo'] = ui.Button('Redo working edit', clicked_fn=lambda: e.undo_asset(True))


def refresh(lab):
    e, storage = lab.gallery.extra, lab.storage
    action, title, hint, button = next_step(storage)
    busy = bool(storage.task and not storage.task.done())
    lab.storage_step.text = title
    lab.storage_hint.text = hint
    lab.storage_primary.text = 'Working...' if busy else button
    lab.storage_primary.enabled = not busy and action is not None
    lab.storage_primary.name = 'Busy' if busy else 'Primary' if action else ''
    lab.storage_save_button.enabled = not busy
    choices = names(lab)
    lab.stats.text = f'Working: {choices[e.storage_variant]}\nSaved: {choices[e.saved_variant] if e.saved_variant is not None else "No snapshot yet"}'
    lab.detail.text = e.storage_status
    lab.revision_label.text = (f'{e.revisions[e.selected_revision].stem} / {len(e.revisions)} saved' if e.revisions else 'No saved snapshots yet')
    for control in lab.storage_controls.values():
        control.enabled = not busy
    lab.storage_controls['undo'].enabled = not busy and bool(e.undo_stack)
    lab.storage_controls['redo'].enabled = not busy and bool(e.redo_stack)
    lab.storage_controls['previous'].enabled = not busy and e.selected_revision > 0
    lab.storage_controls['next'].enabled = not busy and e.selected_revision < len(e.revisions)-1
    lab.storage_controls['restore'].enabled = not busy and bool(e.revisions)
