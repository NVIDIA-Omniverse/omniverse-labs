# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live terminal consumer; no service ownership, Kit or UI dependencies."""
import argparse
from pathlib import Path
import signal
import threading
import time
from composable_bindings.reference.client import Client


def run(client, stop, duration=0):
    client.start()
    deadline = time.monotonic() + duration if duration else float('inf')
    last_status = None
    try:
        while not stop.is_set() and time.monotonic() < deadline:
            for line in client.drain_logs():
                print(line, flush=True)
            status = client.snapshot()['status']
            if status != last_status:
                print('# ' + status, flush=True)
                last_status = status
            stop.wait(.1)
    finally:
        client.stop()



def main(argv=None):
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=root / '.runtime/default')
    parser.add_argument('--client-config', type=Path, help='Reader credentials; overrides --state')
    parser.add_argument('--view', type=Path, help='Optional local display configuration')
    parser.add_argument('--duration', type=float, default=0, help='Stop after this many seconds; zero runs until Ctrl-C')
    args = parser.parse_args(argv)
    credentials = args.client_config or args.state / 'client.json'
    if not credentials.is_file():
        parser.error('Start the service first or provide --state or --client-config.')
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    run(Client(credentials, args.view), stop, args.duration)


if __name__ == '__main__':
    main()
