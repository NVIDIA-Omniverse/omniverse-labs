# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded asynchronous JSONL recording of public stream packets; no database."""
import json
import os
from pathlib import Path
import queue
import re
import threading
import time
import uuid

MAX_BYTES = 64 * 1024 * 1024
FORMAT = 'composable-bindings-recording'


class Session:
    def __init__(self, directory, initial):
        self.id = uuid.uuid4().hex
        self.created = time.time()
        # Use a high-resolution elapsed clock for closely spaced Windows events.
        self.started = time.perf_counter()
        self.duration = 0.
        self.count = 0
        self.error = None
        self.finished = False
        self.directory = directory
        self.queue = queue.Queue(4096)
        self.stopping = threading.Event()
        self.accepting_lock = threading.Lock()
        self.queue.put({'at': 0., 'packet': initial})
        self.thread = threading.Thread(target=self._write, name='recording-writer', daemon=True)
        self.thread.start()

    def metadata(self):
        return dict(id=self.id, file=self.id+'.jsonl', created=self.created,
                    duration=self.duration if self.finished else time.perf_counter()-self.started,
                    packets=self.count, recording=not self.finished, error=self.error)

    def offer(self, packet):
        with self.accepting_lock:
            if self.stopping.is_set() or self.finished:
                return
            try:
                self.queue.put_nowait({'at': time.perf_counter()-self.started, 'packet': packet})
            except queue.Full:
                self.error = 'Recording stopped: writer queue overflow; capture is incomplete'
                self.stopping.set()

    def stop(self):
        # Close admission atomically, then drain accepted packets off the bus lock.
        with self.accepting_lock:
            self.stopping.set()
        self.thread.join(timeout=4)
        if self.thread.is_alive():
            raise ValueError('Recording is still flushing; try Stop again')
        return self.metadata()

    def _write(self):
        partial = self.directory/(self.id+'.jsonl.part')
        try:
            with partial.open('x', encoding='utf-8') as stream:
                header = dict(format=FORMAT, version=1, id=self.id, created=self.created)
                stream.write(json.dumps(header)+'\n')
                size = 0
                while not self.stopping.is_set() or not self.queue.empty():
                    try:
                        item = self.queue.get(timeout=.1)
                    except queue.Empty:
                        continue
                    data = json.dumps(item, separators=(',', ':'), allow_nan=False)+'\n'
                    size += len(data.encode('utf-8'))
                    if size > MAX_BYTES:
                        self.error = 'Recording stopped: 64 MiB size limit; capture is incomplete'
                        break
                    stream.write(data)
                    self.duration = item['at']
                    self.count += 1
                if not self.error:
                    self.duration = max(self.duration, time.perf_counter()-self.started)
                footer = dict(kind='end', duration=self.duration, packets=self.count, error=self.error)
                stream.write(json.dumps(footer)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(partial, self.directory/(self.id+'.jsonl'))
        except (OSError, ValueError, TypeError) as error:
            self.error = 'Recording failed: '+type(error).__name__
        finally:
            self.finished = True


class FileRecorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.session = None
        self.lock = threading.Lock()

    def start(self, initial):
        with self.lock:
            if self.session and not self.session.finished:
                raise ValueError('A recording is already active')
            self.session = Session(self.directory, initial)
            return self.session.metadata()

    def offer(self, packet):
        with self.lock:
            if self.session:
                self.session.offer(packet)

    def stop(self):
        with self.lock:
            session = self.session
        if not session:
            raise ValueError('No recording has been started')
        return session.stop()

    def close(self):
        if self.session and not self.session.finished:
            self.session.stop()

    def path(self, identity):
        if not re.fullmatch(r'[0-9a-f]{32}', identity):
            raise ValueError('Invalid recording ID')
        path = self.directory/(identity+'.jsonl')
        if not path.is_file():
            raise ValueError('Recording is not finalized or does not exist')
        return path

    def catalog(self):
        recordings = []
        for path in self.directory.glob('*.jsonl'):
            try:
                with path.open('rb') as stream:
                    header = json.loads(stream.readline(65536))
                    stream.seek(max(0, path.stat().st_size-4096))
                    tail = stream.read().splitlines()
                    footer = json.loads(tail[-1])
                if header.get('format') != FORMAT or footer.get('kind') != 'end':
                    continue
                recordings.append(dict(id=path.stem, file=path.name, created=header['created'],
                                       duration=footer['duration'], packets=footer['packets'],
                                       error=footer.get('error'), recording=False))
            except (OSError, ValueError, KeyError, IndexError):
                continue
        recordings.sort(key=lambda r:r['created'], reverse=True)
        with self.lock:
            active = self.session.metadata() if self.session and not self.session.finished else None
            error = self.session.error if self.session else None
        return dict(recordings=recordings, active=active, error=error)
