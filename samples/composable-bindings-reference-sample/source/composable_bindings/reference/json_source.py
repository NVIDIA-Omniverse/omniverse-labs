# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small second-source adapter: JSON Lines to the common ingestion contract.

Field mappings belong to the source integration. No simulator, USD, alarm
evaluation or service implementation is imported here.
"""
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from .config import NAME, SOURCE, read
from .transport import request


def validate_mapping(mapping):
    required = {'version', 'source', 'event_type', 'id_field', 'time_field', 'event_id_field', 'values', 'alarms'}
    if set(mapping) != required or mapping['version'] != 1:
        raise ValueError('Expected source mapping version 1 with identity, time, values and alarms')
    if not SOURCE.fullmatch(mapping['source']) or not NAME.fullmatch(mapping['event_type']):
        raise ValueError('Invalid source system or event type')
    fields = [mapping[key] for key in ('id_field', 'time_field', 'event_id_field')]
    for kind in ('values', 'alarms'):
        if not isinstance(mapping[kind], dict) or any(not NAME.fullmatch(name) for name in mapping[kind]):
            raise ValueError('Metric mappings must be dictionaries of declared metric names')
        fields.extend(mapping[kind].values())
    if any(not isinstance(field, str) or not field or any(not part for part in field.split('.')) for field in fields):
        raise ValueError('Mapping fields must be nonempty dotted JSON object paths')
    return mapping


def field(record, path, optional=False):
    value = record
    for key in path.split('.'):
        if not isinstance(value, dict) or key not in value:
            if optional:
                return None
            raise ValueError('Missing source field: ' + path)
        value = value[key]
    return value


def adapt(record, mapping):
    """Preserve the source ID, occurrence ID, time and alarm decisions verbatim."""
    return {'source': mapping['source'], 'source_id': field(record, mapping['id_field']),
            'event_id': field(record, mapping['event_id_field']),
            'time': field(record, mapping['time_field']), 'event_type': mapping['event_type'],
            'values': {name: field(record, path, optional=True) for name, path in mapping['values'].items()},
            'alarms': {name: field(record, path, optional=True) for name, path in mapping['alarms'].items()}}


def submit(credentials, raw):
    """Bounded retry with the same source occurrence ID; refresh stale revisions."""
    for attempt in range(3):
        try:
            with request(credentials, '/producer-config') as response:
                revision = json.load(response)['revision']
            with request(credentials, '/ingest', dict(raw, config_revision=revision)) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code != 409 or attempt == 2:
                raise
        except OSError:
            if attempt == 2:
                raise
        time.sleep(.2)


def run(mapping_path, input_path, credentials_path):
    mapping = validate_mapping(read(mapping_path))
    credentials = read(credentials_path)
    stream = sys.stdin if str(input_path) == '-' else Path(input_path).open()
    try:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                result = submit(credentials, adapt(json.loads(line), mapping))
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ValueError(f'Source record {line_number} rejected: {error}') from error
            print(json.dumps(result), flush=True)
    finally:
        if stream is not sys.stdin:
            stream.close()
