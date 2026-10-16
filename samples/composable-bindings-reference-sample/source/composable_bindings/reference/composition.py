# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Local configuration composition. The data service consumes ordinary JSON snapshots.

The service configuration watcher composes the base and reusable fragments. It publishes a
complete immutable configuration directory and atomically switches the selector.
This works with already-running data services; Kit and producer code need no changes.
"""
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from .config import read, load_configuration, validate_configuration, binding_path, load_event_definitions, split_event_definitions
from .simulation import compose_simulation
from .reload import atomic_json


def compose(selection):
    base = Path(selection['base'])
    if 'assets' not in read(base / 'events.json'):
        raise ValueError('A base needs asset definitions. Use --add for event/display fragments.')
    source, bindings, view = load_configuration(base, selection.get('view'), selection.get('bindings'))
    for directory in selection.get('additions', []):
        directory = Path(directory)
        extra_source, extra_view = load_event_definitions(directory), read(directory/'view.json')
        if set(extra_source) not in ({'version', 'events'}, {'version', 'events', 'assets'}) or set(extra_view) not in ({'version', 'displays'}, {'version', 'displays', 'scene'}):
            raise ValueError('Additions extend events/assets and displays/scene without overriding existing entries')
        if extra_source['version'] != 1 or extra_view['version'] != 1:
            raise ValueError('Unsupported addition version')
        source['assets'].extend(extra_source.get('assets', []))
        if (directory / 'bindings.json').exists():
            extra_bindings = read(directory / 'bindings.json')
            if extra_bindings.get('version') != 1:
                raise ValueError('Unsupported bindings addition version')
            bindings['bindings'].extend(extra_bindings['bindings'])
        view.setdefault('scene', []).extend(extra_view.get('scene', []))
        event_ids = {e['type'] for e in source['events']}
        display_ids = {d['id'] for d in view['displays']}
        for event in extra_source['events']:
            if event['type'] in event_ids:
                raise ValueError('Addition conflicts with event type: ' + event['type'])
            event_ids.add(event['type']);source['events'].append(event)
        for display in extra_view['displays']:
            if display['id'] in display_ids:
                raise ValueError('Addition conflicts with display ID: ' + display['id'])
            display_ids.add(display['id']);view['displays'].append(display)
    return validate_configuration(source, bindings, view)


def _publish(state, selection):
    config = compose(selection)
    simulation = compose_simulation(selection, config[0])
    digest = hashlib.sha256(json.dumps((config, simulation), sort_keys=True).encode()).hexdigest()
    # Always publish a private, self-contained runtime snapshot. The editable
    # source-ID catalog stays outside the simulator's configuration.
    compiled = state/'composed';compiled.mkdir(exist_ok=True,mode=0o700)
    directory = compiled/('revision-rules-'+digest)
    if not directory.exists():
        directory.mkdir(mode=0o700)
        definitions,rules=split_event_definitions(config[0])
        for filename,value in (('events.json',definitions),('rules.json',rules),
                               ('bindings.json',config[1]),('view.json',config[2])):
            atomic_json(directory/filename,value)
        if simulation is not None:
            atomic_json(directory / 'simulation.json', simulation)
    target = {'directory':str(directory),'view':None}
    current = read(state/'configuration.json')
    if current != target:
        previous = Path(current['directory'])
        if previous.parent == state/'composed' and previous.exists():
            os.utime(previous, None)
        atomic_json(state/'configuration.json',target)
    # Retain deselected snapshots for 60 seconds so in-flight readers finish.
    for old in (state/'composed').glob('revision-*'):
        if str(old) != target['directory'] and time.time()-old.stat().st_mtime>60:
            shutil.rmtree(old)
    status = {'request':selection['request'],'base':selection['base'],
              'additions':selection.get('additions',[]),
              'bindings':str(binding_path(selection['base'],selection.get('bindings')).resolve()),
              'digest':digest,'error':None}
    if not (state/'composition-status.json').exists() or read(state/'composition-status.json') != status:
        atomic_json(state/'composition-status.json',status)
    return status


def refresh(state):
    """Called only by the service-owned configuration watcher."""
    state = Path(state)
    if not (state/'composition.json').exists():
        return
    selection = {}
    try:
        selection = read(state/'composition.json')
        _publish(state, selection)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        message = 'Configuration rejected; keeping last valid state: ' + str(error)
        atomic_json(state/'composition-status.json', {'request':selection.get('request'), 'error':message})
        target = read(state/'configuration.json')
        if target.get('error') != message:
            atomic_json(state/'configuration.json', dict(target, error=message))


def configure(state, *, base=None, view=None, bindings=None, add=None, remove=None, status=False):
    state=Path(state)
    if not (state/'configuration.json').exists():
        raise ValueError('Start the service first with this --state directory')
    from .transport import request
    try:
        with request(read(state/'client.json'), '/catalog') as response:
            response.read()
    except OSError as error:
        raise ValueError('Start the service first with this --state directory') from error
    if status:
        if (state/'composition-status.json').exists():
            return read(state/'composition-status.json')
        return {'base':read(state/'configuration.json')['directory'],'additions':[]}
    if base is not None:
        previous=read(state/'composition.json') if (state/'composition.json').exists() else read(state/'configuration.json')
        selection={'base':str(Path(base).resolve()),'view':str(Path(view).resolve()) if view else None,
                   'bindings':previous.get('bindings'), 'additions':[]}
    elif (state/'composition.json').exists():
        selection=read(state/'composition.json')
    else:
        current=read(state/'configuration.json')
        selection={'base':current['directory'],'view':current.get('view'),'bindings':current.get('bindings'),'additions':[]}
    if add is not None:
        path=str(Path(add).resolve())
        if path not in selection['additions']:selection['additions'].append(path)
    if remove is not None:
        path=str(Path(remove).resolve())
        if path not in selection['additions']:raise ValueError('That addition is not active')
        selection['additions'].remove(path)
    selection['bindings']=str(binding_path(selection['base'], bindings or selection.get('bindings')).resolve())
    selection['request']=str(uuid.uuid4())
    compose(selection)  # Reject before changing either desired or active state.
    # For old selections whose base is a generated directory, keep it outside GC.
    if Path(selection['base']).parent.name == 'composed' and Path(selection['base']).name.startswith('revision-'):
        raise ValueError('Select a named base with --set before composing additions')
    atomic_json(state/'composition.json',selection)
    # The service is the sole snapshot writer. Wait for its acknowledgement.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        path = state/'composition-status.json'
        if path.exists():
            result = read(path)
            if result.get('request') == selection['request']:
                if result.get('error'):
                    raise ValueError(result['error'])
                return result
        time.sleep(.05)
    raise ValueError('Configuration saved, but the service has not acknowledged it yet')
