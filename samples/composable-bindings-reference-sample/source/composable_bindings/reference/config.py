# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative configuration validation. No imports, expressions or eval from JSON."""
import copy
import json
import math
import re
from pathlib import Path

NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_.-]{0,95}$")


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


SOURCE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{0,95}$")


def source_key(entry):
    """Keep the issuing system and its verbatim ID together, including in caches."""
    return entry['source'], entry['id']


def validate_identity(entry):
    if not isinstance(entry.get('source'), str) or not SOURCE.fullmatch(entry['source']):
        raise ValueError('Source must be a named system (letters, digits and underscores)')
    if not isinstance(entry.get('id'), str) or not 1 <= len(entry['id']) <= 256:
        raise ValueError('A source-issued ID must be a nonempty string of at most 256 characters')


def validate_rules(rules):
    if rules.get('version') != 1 or not isinstance(rules.get('rules'), list):
        raise ValueError('Expected rules version 1 and a rules list')
    seen = set()
    for rule in rules['rules']:
        if set(rule) != {'metric', 'threshold'} or not isinstance(rule['metric'], str) or not NAME.fullmatch(rule['metric']) or not number(rule['threshold']):
            raise ValueError('Each alarm rule requires a metric and a finite numeric threshold')
        if rule['metric'] in seen:
            raise ValueError('Duplicate alarm rule for metric: ' + rule['metric'])
        seen.add(rule['metric'])
    return rules


def load_event_definitions(directory):
    """Load the ingestion contract and edge rules, independently of any producer."""
    directory = Path(directory)
    definitions = read(directory / 'events.json')
    metrics = {m['name']: m for e in definitions['events'] for m in e['metrics']}
    if any('condition' in m for m in metrics.values()):
        raise ValueError('Put edge alarm conditions in rules.json')
    for rule in validate_rules(read(directory / 'rules.json'))['rules']:
        if rule['metric'] not in metrics:
            raise ValueError('Rule references unknown metric: ' + rule['metric'])
        metric = metrics[rule['metric']]
        if metric.get('alarm') == 'source':
            raise ValueError('A source-owned alarm cannot also have an edge threshold')
        metric['condition'] = {'above': rule['threshold']}
    return definitions


def split_event_definitions(config):
    """Export the producer contract separately from edge rules."""
    definitions = copy.deepcopy(config)
    rules = []
    for event in definitions['events']:
        for metric in event['metrics']:
            condition = metric.pop('condition', None)
            if condition:
                rules.append(dict(metric=metric['name'], threshold=condition['above']))
    return definitions, {'version': 1, 'rules': rules}


def validate_events(c):
    if c.get('version') != 1 or set(c) != {'version', 'assets', 'events'}:
        raise ValueError('Expected event contract version 1 with assets and events')
    assets, events = c['assets'], c['events']
    if not 1 <= len(assets) <= 64 or not 1 <= len(events) <= 8:
        raise ValueError('Expected 1–64 assets and 1–8 event definitions')
    for asset in assets:
        validate_identity(asset)
        if set(asset) != {'source', 'id'}:
            raise ValueError('Contract assets contain only source and id')
    identities = {source_key(a) for a in assets}
    if len(identities) != len(assets):
        raise ValueError('Source/ID pairs must be unique')
    if len({e['type'] for e in events}) != len(events):
        raise ValueError('Event types must be unique')
    metric_names, emitted = set(), set()
    for event in events:
        if set(event) != {'type', 'source', 'assets', 'metrics'} or not NAME.fullmatch(event['type']):
            raise ValueError('Each event declares type, source, assets and metrics only')
        targets = event['assets']
        if not isinstance(targets, list) or not targets or any(not isinstance(v, str) for v in targets) or len(set(targets)) != len(targets):
            raise ValueError('Event assets must be nonempty, unique source IDs')
        if any((event['source'], target) not in identities for target in targets):
            raise ValueError('Event references an unknown source/ID pair')
        if not 1 <= len(event['metrics']) <= 16:
            raise ValueError('Expected 1–16 metrics per event')
        for m in event['metrics']:
            if set(m) - {'name', 'unit', 'alarm', 'condition'}:
                raise ValueError('Event metrics declare name, unit and optional alarm ownership only')
            if not NAME.fullmatch(m['name']) or m['name'] in metric_names:
                raise ValueError('Metric names must be valid and globally unique')
            metric_names.add(m['name'])
            if not isinstance(m['unit'], str) or len(m['unit']) > 32:
                raise ValueError('Metric unit must be a short string')
            if 'alarm' in m and m['alarm'] != 'source':
                raise ValueError('The only explicit alarm owner is source')
            condition = m.get('condition')
            if condition is not None and (set(condition) != {'above'} or not number(condition['above'])):
                raise ValueError('This example supports a numeric above condition')
            if condition and m.get('alarm') == 'source':
                raise ValueError('A metric cannot have both source and edge alarms')
            names = [m['name']]
            if condition or m.get('alarm') == 'source':
                names.append(m['name'] + '.alarm')
            if emitted.intersection(names):
                raise ValueError('Measurement and alarm names must not collide')
            emitted.update(names)
    return c


def load_view(path):
    return validate_view(read(path))


def validate_view(c):
    if c.get("version") != 1 or not 1 <= len(c.get("displays", [])) <= 32:
        raise ValueError("Expected view version 1 and 1–32 displays")
    ids = set()
    for d in c["displays"]:
        if not NAME.fullmatch(d["id"]) or not isinstance(d.get("title"), str) or not 1 <= len(d["title"]) <= 160:
            raise ValueError("Displays require an identifier and short title")
        if d["id"] in ids or not NAME.fullmatch(d["metric"]):
            raise ValueError("Display IDs must be unique; metric names must be valid")
        ids.add(d["id"])
        if d.get('placement', 'world') not in ('world', 'hud-top-left'):
            raise ValueError('Display placement must be world or hud-top-left')
        if d["kind"] not in ("label", "gauge"):
            raise ValueError("Supported display kinds: label, gauge")
        if type(d.get("decimals", 1)) is not int or not 0 <= d.get("decimals", 1) <= 6:
            raise ValueError("Display decimals must be 0–6")
        if d["kind"] == "gauge":
            limits = d.get("range", [])
            if len(limits) != 2 or not all(number(x) for x in limits) or limits[0] >= limits[1]:
                raise ValueError("Gauge requires an increasing finite range")
    for b in c.get("scene", []):
        validate_identity(b)
        if 'display_color' in b:
            color = b['display_color']
            if not isinstance(color, dict) or not isinstance(color.get('metric'), str) or not NAME.fullmatch(color['metric']):
                raise ValueError('Display color requires a metric')
            mode = color.get('mode', 'state')
            keys = {'metric', 'colors'} | ({'mode'} if 'mode' in color else set()) | ({'range'} if mode == 'gradient' else set())
            if mode not in ('state', 'gradient') or set(color) != keys:
                raise ValueError('Display color supports state or gradient mode with its palette and range')
            if mode == 'gradient':
                limits = color['range']
                if not isinstance(limits, list) or len(limits) != 2 or not all(number(v) for v in limits) or limits[0] >= limits[1]:
                    raise ValueError('Color gradient requires an increasing finite range')
            palette = color['colors']
            states = {'low', 'high', 'unknown'} if mode == 'gradient' else {'normal', 'alarm', 'unknown'}
            if not isinstance(palette, dict) or set(palette) != states:
                raise ValueError('Display color palette must match its mode and include unknown')
            if any(not isinstance(rgb, list) or len(rgb) != 3 or not all(number(v) and 0 <= v <= 1 for v in rgb) for rgb in palette.values()):
                raise ValueError('Display colors must be three linear RGB components from 0 to 1')
        if "position" in b and (len(b["position"]) != 3 or not all(number(x) for x in b["position"])):
            raise ValueError("Scene position must have three finite coordinates")
        if not number(b.get('gauge_scale',1.)) or not .25 <= b.get('gauge_scale',1.) <= 3.:
            raise ValueError('Gauge scale must be between 0.25 and 3')
        if b.get('geometry','tube') not in ('tube','platform'):
            raise ValueError('Scene geometry must be tube or platform')
        for field in ('geometry_position','gauge_offset','size'):
            if field in b and (len(b[field]) != 3 or not all(number(v) for v in b[field])):
                raise ValueError('Scene vectors must have three finite coordinates')
        if 'size' in b and any(v <= 0 for v in b['size']):
            raise ValueError('Scene size must be positive')
    if len({source_key(b) for b in c.get("scene", [])}) != len(c.get("scene", [])):
        raise ValueError("Scene presentation source/ID pairs must be unique")
    return c


def load_bindings(path):
    return validate_bindings(read(path))


def validate_bindings(c):
    if c.get('version') != 1 or not 1 <= len(c.get('bindings', [])) <= 64:
        raise ValueError('Expected binding version 1 and 1–64 mappings')
    entries = c['bindings']
    for entry in entries:
        validate_identity(entry)
        if set(entry) != {'source', 'id', 'label'}:
            raise ValueError('Bindings contain source-issued source/id and a label; no aliases')
        if not isinstance(entry['label'], str) or not 1 <= len(entry['label']) <= 160:
            raise ValueError('Bindings require a short display label')
    if len({source_key(entry) for entry in entries}) != len(entries):
        raise ValueError('Binding source/ID pairs must be unique')
    return c


DEFAULT_BINDINGS = Path(__file__).resolve().parents[3] / 'config/bindings.json'


def binding_path(directory, explicit=None):
    if explicit is not None: return Path(explicit)
    # Deployment configurations and runtime snapshots may be self-contained.
    legacy = Path(directory) / 'bindings.json'
    return legacy if legacy.exists() else DEFAULT_BINDINGS


def load_configuration(directory, view_path=None, bindings_path=None):
    directory = Path(directory)
    source = load_event_definitions(directory)
    bindings = load_bindings(binding_path(directory, bindings_path))
    view = load_view(view_path or directory / 'view.json')
    return validate_configuration(source, bindings, view)


def validate_configuration(source, bindings, view):
    validate_events(source)
    validate_bindings(bindings)
    if view is not None:
        validate_view(view)
    if {source_key(a) for a in source['assets']} != {source_key(b) for b in bindings['bindings']}:
        raise ValueError('Every source asset must have exactly one binding')
    known = {m['name'] for e in source['events'] for m in e['metrics']}
    ids = {source_key(b) for b in bindings['bindings']}
    if view is not None and any(d['metric'] not in known for d in view['displays']):
        raise ValueError('View references an unknown metric')
    if view is not None and any(source_key(s) not in ids for s in view.get('scene', [])):
        raise ValueError('Scene references an unknown binding')
    if view is not None:
        for entry in view.get('scene', []):
            color = entry.get('display_color')
            if color and not any(e['source'] == entry['source'] and entry['id'] in e['assets'] and
                                 any(m['name'] == color['metric'] for m in e['metrics']) for e in source['events']):
                raise ValueError('Display color metric must apply to its source/ID pair')
    return source, bindings, view
