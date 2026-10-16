# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration owned by the synthetic producer, outside the ingestion contract."""
import copy
from pathlib import Path
from .config import number, read


def validate_simulation(config, definitions):
    if set(config) != {'version', 'source', 'assets', 'events'} or config['version'] != 1:
        raise ValueError('Expected simulation version 1 with source, assets and events')
    source = config['source']
    assets = [a['id'] for a in config['assets']]
    if not assets or len(set(assets)) != len(assets) or any(set(a) != {'id'} for a in config['assets']):
        raise ValueError('Simulation assets contain unique source-issued IDs')
    known = {a['id'] for a in definitions['assets'] if a['source'] == source}
    if not set(assets) <= known:
        raise ValueError('Simulation assets must be declared by this source')
    specs = {e['type']: e for e in definitions['events'] if e['source'] == source}
    if len({e['type'] for e in config['events']}) != len(config['events']):
        raise ValueError('Duplicate simulated event')
    prior = {}
    for event in config['events']:
        spec = specs.get(event['type'])
        if spec is None or not number(event['interval']) or not .1 <= event['interval'] <= 60:
            raise ValueError('Simulation needs a declared event and interval from 0.1 to 60 seconds')
        targets = event['assets']
        if not targets or len(set(targets)) != len(targets) or not set(targets) <= set(assets) & set(spec['assets']):
            raise ValueError('Simulation event targets must be declared source IDs')
        metrics = {m['name']: m for m in spec['metrics']}
        if len(event['metrics']) != len(metrics) or {m['name'] for m in event['metrics']} != set(metrics):
            raise ValueError('Simulation must provide the declared metrics for each simulated event')
        for metric in event['metrics']:
            if set(metric) not in ({'name', 'simulation'}, {'name', 'aggregate'}):
                raise ValueError('Simulation metric needs a wave or aggregate, not event metadata')
            if metrics[metric['name']].get('alarm') == 'source':
                raise ValueError('The wave simulator does not produce source-owned alarms')
            if 'simulation' in metric:
                wave = metric['simulation']
                if set(wave) != {'center', 'amplitude', 'period'} or any(not number(v) for v in wave.values()) or wave['period'] <= 0:
                    raise ValueError('Wave needs finite center/amplitude and a positive period')
            else:
                aggregate = metric['aggregate']
                if set(aggregate) != {'operation', 'metric', 'members'} or aggregate['operation'] != 'sum':
                    raise ValueError('Aggregation supports sum with a metric and members')
                members = aggregate['members']
                previous = prior.get(aggregate['metric'])
                if not members or len(set(members)) != len(members) or previous is None or not set(members) <= set(previous[0]['assets']):
                    raise ValueError('Aggregation needs an earlier simulated metric on all members')
                if previous[0]['interval'] != event['interval'] or previous[1] != metrics[metric['name']]['unit']:
                    raise ValueError('Aggregation inputs need matching intervals and units')
        prior.update({m['name']: (event, metrics[m['name']]['unit']) for m in event['metrics'] if 'simulation' in m})
    return config


def compose_simulation(selection, definitions):
    """Compose only optional simulator files; external-source fragments need none."""
    path = Path(selection['base']) / 'simulation.json'
    config = read(path) if path.exists() else None
    for directory in selection.get('additions', []):
        path = Path(directory) / 'simulation.json'
        if not path.exists():
            continue
        extra = read(path)
        if config is None or extra.get('source') != config['source'] or set(extra) != {'version', 'source', 'events'} or extra['version'] != 1:
            raise ValueError('Simulation additions extend the base simulator source only')
        config['events'].extend(copy.deepcopy(extra['events']))
    return validate_simulation(config, definitions) if config is not None else None
