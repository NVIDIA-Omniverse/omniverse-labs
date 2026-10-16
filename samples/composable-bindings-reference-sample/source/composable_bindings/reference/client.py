# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The complete shared client. Network, decoding and formatting run off the UI thread."""
import json
import threading
import time
from collections import deque
from .config import read, load_view
from .model import ViewModel, presentation_rows
from .transport import request


class Client:
    def __init__(self, credentials_path, view_path=None):
        self.credentials_path = credentials_path
        credentials = read(credentials_path)
        self.local_override = view_path is not None
        self.view_path = view_path
        self.config_revision = 0
        self.connections = 0
        self.epoch = None
        self.config_error = None
        self.local_error = None
        self.next_view_check = 0.
        self.view = load_view(view_path) if view_path else {'version': 1, 'displays': [], 'scene': []}
        self.lock = threading.Lock()
        self.model = ViewModel()
        self.catalog = {'bindings': [], 'metrics': []}
        self.status = 'Connecting'
        self.revision = 0
        self.rows = []
        self.lines = deque(maxlen=2048)
        self.dropped_log_lines = 0
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, name='binding-reader', daemon=True)
            self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=6)

    def reload_view(self):
        if not self.local_override:
            return  # The server watches and broadcasts the shared display configuration.
        view = load_view(self.view_path)
        with self.lock:
            self.local_error = None
            if view == self.view: return
            self.view = view
            self._present()
            self.revision += 1

    def _consume(self, packet):
        with self.lock:
            if packet['kind'] == 'snapshot':
                self.epoch = packet.get('epoch')
                self.catalog = packet.get('catalog', self.catalog)
                self.config_revision = packet.get('config_revision', self.config_revision)
                self.config_error = packet.get('config_error')
                if packet.get('view') is not None and not self.local_override:
                    self.view = packet['view']
                self.model.clear()
                events = packet['events']
                self.status = 'Live; snapshot: ' + packet['reason']
                if packet['reason'] == 'slow_consumer_gap':
                    self.lines.append('GAP: slow consumer; restored latest state, transitions were lost')
            elif packet['kind'] == 'event':
                events = [packet['event']]
                self.status = 'Live'
            else:
                raise ValueError('Unknown stream packet')
            for event in events:
                changed = self.model.apply(event)
                lines = self.model.log_lines(changed, self.view)
                self.dropped_log_lines += max(0, len(self.lines) + len(lines) - self.lines.maxlen)
                self.lines.extend(lines)
            self._present()
            self.revision += 1

    def _present(self):
        self.rows = presentation_rows(self.model, self.catalog, self.view)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                credentials = read(self.credentials_path)
                with request(credentials, '/catalog') as response:
                    catalog = json.load(response)
                with self.lock:
                    self.catalog = catalog
                with request(credentials, '/events') as stream:
                    self.connections += 1
                    while not self.stop_event.is_set():
                        line = stream.readline(2 * 1024 * 1024)
                        if not line:
                            raise ConnectionError('Stream closed')
                        if self.local_override and time.monotonic() >= self.next_view_check:
                            self.next_view_check = time.monotonic() + .25
                            try: self.reload_view()
                            except (OSError, ValueError, KeyError, TypeError) as error:
                                with self.lock:
                                    self.local_error = 'Display configuration rejected: ' + type(error).__name__
                                    self.revision += 1
                        if line.startswith(b'data: '):
                            self._consume(json.loads(line[6:]))
            except (OSError, ValueError, KeyError, TypeError) as error:
                with self.lock:
                    self.status = 'Disconnected; values stale: ' + type(error).__name__
                    self.revision += 1
                self.stop_event.wait(.5)

    def snapshot(self):
        """Small immutable-by-convention presentation snapshot; no network or decoding."""
        with self.lock:
            return {'revision': self.revision, 'status': self.local_error or self.config_error or self.status, 'rows': self.rows,
                    'stale': self.status.startswith('Disconnected'),
                    'config_revision': self.config_revision,
                    'scene': list(self.view.get('scene', []))}

    def drain_logs(self):
        with self.lock:
            result = list(self.lines)
            self.lines.clear()
            if self.dropped_log_lines:
                result.insert(0, f'GAP: {self.dropped_log_lines} shell log lines dropped')
                self.dropped_log_lines = 0
            return result
