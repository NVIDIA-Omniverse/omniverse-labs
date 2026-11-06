#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Private, inherited-socket worker for the opt-in public SDK experiment.

The control socket is separate from stdout: native SDK logging cannot corrupt
replies. One client owns this process. A new render cannot overwrite a held
frame until that client has copied and acknowledged it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import struct
import threading
import faulthandler
import sys
import time
import traceback

import runpy
from functools import lru_cache


@lru_cache(maxsize=1)
def _frame_protocol():
    root = Path(__file__).resolve().parents[1]
    path = root / 'addon/ovrtx_blender_example/frame_protocol.py'
    if not path.is_file():
        path = root / 'ovrtx_blender_example/frame_protocol.py'
    return runpy.run_path(str(path))


MAX_MESSAGE = 16 * 1024 * 1024
BOOTSTRAP_MAX_BYTES = 4096
BOOTSTRAP_IDENTITY = {}


def receive_control_socket(fd=None, *, shared_stdin=False, timeout_s=15.0):
    """Authenticate this actual worker before Windows socket duplication.

    The fresh secret arrives only through inherited stdin. Native imports and
    control work start only after bounded identity/capability negotiation.
    """
    if (fd is None) == (not shared_stdin):
        raise ValueError('Select exactly one worker control bootstrap')
    if not 0 < timeout_s <= 15:
        raise ValueError('Invalid bootstrap timeout')
    watchdog = threading.Timer(timeout_s, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    connection = None
    bootstrap = None
    try:
        line = sys.stdin.buffer.readline(BOOTSTRAP_MAX_BYTES + 1)
        if len(line) > BOOTSTRAP_MAX_BYTES or not line.endswith(b'\n'):
            raise ValueError('Invalid worker identity bootstrap size')
        try:
            challenge = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            raise ValueError('Invalid worker identity bootstrap') from exc
        if (not isinstance(challenge, dict) or type(challenge.get('protocol')) is not int
                or challenge.get('protocol') != 2):
            raise ValueError('PROTOCOL_MISMATCH: required bootstrap version is 2')
        if (type(challenge.get('epoch')) is not int or challenge['epoch'] <= 0
                or not isinstance(challenge.get('session'), str)
                or len(challenge['session']) != 64
                or any(c not in '0123456789abcdef' for c in challenge['session'])):
            raise ValueError('Invalid worker identity challenge')
        negotiated = _frame_protocol()['negotiate'](challenge.get('application'))
        sys.stdin.close()
        identity = {'protocol': 2, 'session': challenge['session'],
                    'epoch': challenge['epoch'], 'pid': os.getpid(),
                    'capabilities': dict(_frame_protocol()['CAPABILITIES']), 'application': negotiated}
        if shared_stdin:
            port = challenge.get('port')
            if type(port) is not int or not 0 < port < 65536:
                raise ValueError('Invalid private bootstrap port')
            bootstrap = socket.create_connection(('127.0.0.1', port), timeout_s)
            bootstrap.sendall(json.dumps(identity).encode() + b'\n')
            def read_exact(size):
                data = bytearray()
                while len(data) < size:
                    chunk = bootstrap.recv(size - len(data))
                    if not chunk:
                        raise ValueError('Truncated worker socket bootstrap')
                    data.extend(chunk)
                return bytes(data)
            length = struct.unpack('!I', read_exact(4))[0]
            if not 0 < length <= BOOTSTRAP_MAX_BYTES:
                raise ValueError('Invalid worker socket bootstrap size')
            connection = socket.fromshare(read_exact(length))
            connection.settimeout(timeout_s)
            connection.sendall(json.dumps({'ready': True, 'session': challenge['session'],
                'epoch': challenge['epoch'], 'pid': os.getpid()}).encode() + b'\n')
        else:
            connection = socket.socket(fileno=fd)
            connection.settimeout(timeout_s)
            connection.sendall(json.dumps(identity).encode() + b'\n')
            ack = bytearray()
            while not ack.endswith(b'\n') and len(ack) <= BOOTSTRAP_MAX_BYTES:
                part = connection.recv(1)
                if not part:
                    raise ValueError('Worker bootstrap closed')
                ack.extend(part)
            if ack != b'{"ready":true}\n':
                raise ValueError('Invalid worker bootstrap acknowledgment')
        BOOTSTRAP_IDENTITY.clear()
        BOOTSTRAP_IDENTITY.update(identity)
        connection.settimeout(None)
        return connection
    except BaseException:
        if connection is not None:
            connection.close()
        raise
    finally:
        if bootstrap is not None:
            bootstrap.close()
        watchdog.cancel()


def send_response(connection, response):
    """Send one complete bounded reply, including under socket backpressure."""
    payload = json.dumps(response, allow_nan=False).encode() + b'\n'
    if len(payload) > MAX_MESSAGE:
        payload = json.dumps({
            'id': response.get('id'), 'epoch': response.get('epoch'), 'session': response.get('session'),
            'pid': os.getpid(), 'ok': False,
            'error': 'Runtime response exceeds 16 MiB limit',
        }).encode() + b'\n'
    connection.sendall(payload)


def unload_stages(backend):
    """``stage=<s>s`` for each unload stage the backend timed (empty if none ran)."""
    timings = getattr(backend, 'unload_timings', None) or {}
    return ' '.join(f'{name}={seconds:.3f}s' for name, seconds in timings.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('renderer', 'physics'), required=True)
    control = parser.add_mutually_exclusive_group(required=True)
    control.add_argument('--fd', type=int)
    control.add_argument('--socket-share-stdin', action='store_true')
    parser.add_argument('--bootstrap-timeout', type=float, default=15.0)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    # Parent launches with -I: never import through PYTHONPATH, user site or
    # cwd. Only this selected worker's adjacent runtime modules are admitted.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # Local sidecars do not need Hub discovery or its startup/network waits.
    # An explicitly configured deployment can still select another mode.
    os.environ.setdefault('OMNICLIENT_HUB_MODE', 'disabled')
    faulthandler.enable()
    args.output_root.mkdir(parents=True, exist_ok=True)
    connection = receive_control_socket(args.fd, shared_stdin=args.socket_share_stdin,
                                        timeout_s=args.bootstrap_timeout)
    stream = connection.makefile('rb')
    backend = None
    try:
        from runtime_identity import verified_identity
        runtime_identity = verified_identity()
        if args.kind == 'renderer':
            from public_sdk_backend import RendererBackend as backend_type
        else:
            from incremental_physics_backend import IncrementalPhysicsBackend as backend_type
        backend = backend_type(output_root=args.output_root)
        if args.kind == 'renderer':
            # Frame descriptors and the owned-resource registry use exactly
            # the authenticated bootstrap session, including after restart.
            backend.epoch = BOOTSTRAP_IDENTITY['session']
        while True:
            line = stream.readline(MAX_MESSAGE + 1)
            if not line:
                break
            if len(line) > MAX_MESSAGE or not line.endswith(b'\n'):
                raise ValueError('control message exceeds limit')
            request = json.loads(line)
            if (request.get('epoch') != BOOTSTRAP_IDENTITY['epoch']
                    or request.get('session') != BOOTSTRAP_IDENTITY['session']):
                raise ValueError('Control request belongs to wrong worker session')
            request_id = request.get('id')
            try:
                op = request['op']
                started = time.monotonic()
                if op in ('load', 'unload'):
                    print(f'public_sdk begin {request_id} {op}', file=sys.stderr, flush=True)
                params = request.get('params', {})
                if not isinstance(params, dict):
                    raise ValueError('params must be an object')
                if os.environ.get('OV_BLENDER_PUBLIC_SDK_TRACE_UPDATES') == '1' and op in (
                    'update', 'update_attributes', 'update_transforms', 'update_scene', 'resize'
                ):
                    print('public_sdk update_trace ' + json.dumps(
                        {'id': request_id, 'op': op, 'params': params}), file=sys.stderr, flush=True)
                # A hung SDK call leaves evidence without tracing idle workers.
                faulthandler.dump_traceback_later(60, repeat=False)
                try:
                    result = backend.dispatch(op, params)
                    if op == 'health':
                        if args.kind == 'physics':
                            if result.get('runtime_identity', {}).get('runtime_lock') != runtime_identity:
                                raise ValueError('Physics admission identity does not match the verified runtime lock')
                            result = {**result, 'runtime_lock': runtime_identity}
                        else:
                            result = {**result, 'runtime_identity': runtime_identity}
                finally:
                    faulthandler.cancel_dump_traceback_later()
                if op in ('load', 'unload'):
                    print(f'public_sdk end {request_id} {op} {time.monotonic() - started:.3f}s', file=sys.stderr, flush=True)
                if op == 'unload' and unload_stages(backend):
                    print(f'public_sdk unload-stages {request_id} {unload_stages(backend)}', file=sys.stderr, flush=True)
                response = {'id': request_id, 'ok': True, 'result': result,
                            'epoch': BOOTSTRAP_IDENTITY['epoch'], 'session': BOOTSTRAP_IDENTITY['session'], 'pid': os.getpid()}
            except Exception as exc:
                traceback.print_exc()
                response = {'id': request_id, 'ok': False,
                            'error': f'{type(exc).__name__}: {exc}',
                            'epoch': BOOTSTRAP_IDENTITY['epoch'], 'session': BOOTSTRAP_IDENTITY['session'], 'pid': os.getpid()}
            send_response(connection, response)
    finally:
        try:
            if backend is not None:
                # The add-on closes the socket and waits (bounded) for this
                # process to exit, so this unload is part of its shutdown time.
                started, ok = time.monotonic(), False
                try:
                    backend.dispatch('unload', {})
                    ok = True
                finally:
                    print(f'public_sdk exit-unload {time.monotonic() - started:.3f}s kind={args.kind}'
                          f'{"" if ok else " failed"} {unload_stages(backend)}'.rstrip(),
                          file=sys.stderr, flush=True)
        finally:
            stream.close()
            connection.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
