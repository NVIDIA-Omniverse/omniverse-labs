# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkout-friendly stdlib entrypoints; separate producer/service/client processes."""
import argparse
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from .config import load_configuration, binding_path
from .reload import ConfigurationWatcher, atomic_json
from .client import Client
from .service import Service
from .transport import Server
from . import simulator

ROOT = Path(__file__).resolve().parents[3]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['demo', 'serve', 'produce', 'adapt', 'console', 'validate', 'configure'])
    choice = p.add_mutually_exclusive_group()
    choice.add_argument('--config', '--set', dest='config', type=Path, help='Replace the complete configuration and clear additions')
    choice.add_argument('--add', type=Path, help='Merge an event/display fragment into the current configuration')
    choice.add_argument('--remove', type=Path, help='Remove one previously added fragment')
    choice.add_argument('--status', action='store_true', help='Show active base, additions and configuration errors')
    p.add_argument('--state', type=Path, default=ROOT / '.runtime/default')
    p.add_argument('--view', type=Path)
    p.add_argument('--bindings', type=Path, help='Source ID catalog and labels (default: config/bindings.json)')
    p.add_argument('--port', type=int, default=int(os.environ.get('CB_REFERENCE_PORT', '9861')))
    p.add_argument('--duration', type=float, default=0)
    p.add_argument('--no-console', action='store_true')
    p.add_argument('--mapping', type=Path, help='JSON source adapter field mapping')
    p.add_argument('--input', default='-', help='JSON Lines source records (default: stdin)')
    a = p.parse_args(argv)
    a.state = a.state.resolve()
    if a.command == 'adapt':
        if a.mapping is None:
            p.error('adapt requires --mapping')
        from .json_source import run
        try:
            return run(a.mapping, a.input, a.state / 'producer.json')
        except (OSError, ValueError, KeyError, TypeError) as error:
            p.error(str(error))
    if a.mapping is not None or a.input != '-':
        p.error('--mapping and --input apply to adapt only')
    if a.command == 'configure':
        from .composition import configure
        if not (a.config or a.add or a.remove or a.status):
            p.error('Choose --set/--config, --add, --remove or --status')
        if a.view and not a.config:p.error('--view is only valid with --set/--config')
        try:
            result = configure(a.state, base=a.config, view=a.view, bindings=a.bindings, add=a.add, remove=a.remove, status=a.status)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            p.error(str(error))
        print(json.dumps(result, indent=2))
        return
    if a.add or a.remove or a.status:p.error('--add, --remove and --status apply to configure')
    a.config = (a.config or ROOT/'examples/default').resolve()
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    if a.command == 'console':
        from clients.console_client.app import run
        return run(Client(a.state / 'client.json', a.view), stop, a.duration)
    if a.command == 'produce':
        return simulator.run(a.state, a.state / 'producer.json', stop)
    try:
        source, bindings, view = load_configuration(a.config, a.view, a.bindings)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        p.error(f'Cannot load configuration {a.config}: {error}. '
                'Expected events.json, rules.json and view.json, plus config/bindings.json or --bindings PATH. '
                'Use examples/default as a complete configuration; add examples/voltage with configure --add.')
    if a.command == 'validate':
        print('Valid event contract, rules, source IDs and view configurations')
        return
    a.state.mkdir(parents=True, exist_ok=True, mode=0o700)
    read_token, ingest_token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    service = Service(source, bindings, view=view, recordings=a.state / 'recordings')
    server = Server(a.port, service, read_token, ingest_token)
    atomic_json(a.state / 'configuration.json', {'directory': str(a.config),
                'view': str(a.view.resolve()) if a.view else None,
                'bindings': str(binding_path(a.config,a.bindings).resolve())})
    for stale in ('composition.json', 'composition-status.json'):
        (a.state / stale).unlink(missing_ok=True)
    watcher = ConfigurationWatcher(service, a.state / 'configuration.json').start()
    url = 'http://127.0.0.1:' + str(server.server_port)
    atomic_json(a.state / 'client.json', {'url': url, 'token': read_token})
    atomic_json(a.state / 'producer.json', {'url': url, 'token': ingest_token})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    producer = None
    print(f'Service at {url}; client credentials: {a.state / "client.json"}', flush=True)
    try:
        if a.command == 'demo':
            producer = subprocess.Popen([sys.executable, str(ROOT / 'reference.py'), 'produce',
                      '--config', str(a.config), '--state', str(a.state)])
        if a.command == 'demo' and not a.no_console:
            from clients.console_client.app import run
            run(Client(a.state / 'client.json', a.view), stop, a.duration)
        else:
            deadline = time.monotonic() + a.duration if a.duration else float('inf')
            while not stop.wait(.1) and time.monotonic() < deadline:
                # Ctrl-C reaches the whole console group, so the producer can exit
                # before our own handler runs (on Windows only after the wait returns).
                # Allow that pending stop request to land before reporting a failure.
                if producer and producer.poll() is not None and not stop.wait(1):
                    raise RuntimeError(f'Producer exited with code {producer.returncode}')
    finally:
        watcher.stop()
        service.recorder.close()
        if producer:
            producer.terminate()
            try: producer.wait(timeout=7)
            except subprocess.TimeoutExpired:
                producer.kill(); producer.wait()
        server.stopping.set()
        server.shutdown(); server.server_close(); thread.join(timeout=2)
