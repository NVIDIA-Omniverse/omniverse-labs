# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic source adapter; source-issued IDs are preserved on the stream."""
import json
import sys
from urllib.error import HTTPError
import time
import uuid
from pathlib import Path
from .aggregation import measurements
from .config import read
from .model import utc
from .transport import request
from .simulation import validate_simulation


def run(state, credentials_path, stop):
    config = None
    revision = None
    refresh = 0.
    credentials = read(credentials_path)
    due = {}
    started = time.monotonic()
    last_error = None
    while not stop.is_set():
        now = time.monotonic()
        if now >= refresh:
            refresh = now + .25
            try:
                with request(credentials, '/producer-config') as response:
                    active = json.load(response)
                selection = read(Path(state) / 'configuration.json')
                simulation = validate_simulation(read(Path(selection['directory']) / 'simulation.json'), active['definitions'])
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                # The selector can advance before the service accepts its contract.
                # Retry that boundary and partial file edits without killing the producer.
                message = 'Simulation configuration not ready; retrying: ' + str(error)
                if message != last_error:
                    print(message, file=sys.stderr, flush=True)
                    last_error = message
                stop.wait(.05)
                continue
            last_error = None
            if active['revision'] != revision or simulation != config:
                config, revision = simulation, active['revision']
                due = {e['type']: 0. for e in config['events']}
        stale = False
        if not any(deadline <= now for deadline in due.values()):
            stop.wait(.02)
            continue
        batch = measurements(config, now - started)
        observed = utc(time.time())
        for event in config['events']:
            if due[event['type']] > now:
                continue
            due[event['type']] = now + event['interval']
            for native in event['assets']:
                values = {m['name']: batch[native, m['name']] for m in event['metrics']}
                raw = {'source': config['source'], 'source_id': native, 'event_type': event['type'],
                       'event_id': str(uuid.uuid4()), 'time': observed, 'values': values,
                       'config_revision': revision}
                # Retry the same occurrence ID, so the broker can deduplicate it.
                for attempt in range(3):
                    try:
                        with request(credentials, '/ingest', raw) as response:
                            response.read()
                        break
                    except HTTPError as error:
                        if error.code != 409: raise
                        refresh = 0.
                        stale = True
                        break
                    except OSError:
                        if attempt == 2:
                            raise
                        if stop.wait(.2):
                            return
                if stale: break
            if stale: break
        stop.wait(.02)
