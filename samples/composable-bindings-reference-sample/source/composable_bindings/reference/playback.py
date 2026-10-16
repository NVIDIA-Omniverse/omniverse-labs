# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared, UI-independent file replay and asynchronous recording controls."""
from bisect import bisect_right
import json
import math
import queue
import threading
import time
from urllib.error import HTTPError
from .config import read
from .model import ViewModel, presentation_rows
from .recording import FORMAT, MAX_BYTES
from .transport import request


class RecordingIndex:
    """Seek by indexed samples, preserving recorded config boundaries and alarms."""
    def __init__(self, lines):
        self.series = {}
        self.contexts = []
        self.context_times = []
        model = ViewModel()
        self.header = json.loads(next(lines))
        if self.header.get('format') != FORMAT or self.header.get('version') != 1:
            raise ValueError('Unsupported recording format')
        self.duration = None
        previous = -1.
        for index, line in enumerate(lines):
            item = json.loads(line)
            if item.get('kind') == 'end':
                self.duration = float(item['duration'])
                if item.get('error'):
                    raise ValueError(item['error'])
                break
            at = float(item['at'])
            if not math.isfinite(at) or at < previous:
                raise ValueError('Recording times must be finite and ordered')
            previous = at
            packet = item['packet']
            if packet['kind'] == 'snapshot':
                model.clear()
                self.contexts.append((index, packet))
                self.context_times.append(at)
                events = packet['events']
            elif packet['kind'] == 'event':
                events = [packet['event']]
            else:
                raise ValueError('Unknown recording packet')
            for event in events:
                for sample in model.apply(event):
                    times, samples = self.series.setdefault((sample.identity,sample.metric), ([],[]))
                    times.append(at)
                    samples.append((index,sample))
        if not self.contexts or self.duration is None or not math.isfinite(self.duration) or self.duration < max(0.,previous):
            raise ValueError('Incomplete recording')

    def snapshot(self, position):
        position = max(0., min(self.duration, position))
        context = bisect_right(self.context_times, position)-1
        if context < 0:
            raise ValueError('No initial recording snapshot')
        boundary, packet = self.contexts[context]
        model = ViewModel()
        for key,(times,samples) in self.series.items():
            i = bisect_right(times,position)-1
            if i >= 0 and samples[i][0] >= boundary:
                model.latest[key] = samples[i][1]
        view = packet['view']
        return dict(rows=presentation_rows(model,packet['catalog'],view),
                    scene=list(view.get('scene',[])),config_revision=packet.get('config_revision',0),
                    status='Local capture replay; not source history; live collection continues')


class PlaybackController:
    """Kit callbacks enqueue work; network, indexing and seeks stay on this worker."""
    def __init__(self, live_client, credentials_path):
        self.client = live_client
        self.credentials_path = credentials_path
        self.lock = threading.RLock()
        self.commands = queue.Queue()
        self.stopping = threading.Event()
        self.thread = self.monitor = None
        self.index = None
        self.frame = None
        self.requested_position = None
        self.revision = 0
        self.polled_error = None  # Last catalog/connection error, cleared once polling recovers.
        self.state = dict(mode='live', playing=False, position=0., duration=0., created=0.,
                          recordings=[], active=None, file=None, busy=False, error=None)

    def start(self):
        self.thread = threading.Thread(target=self._run,name='recording-playback',daemon=True)
        self.thread.start()
        self.monitor = threading.Thread(target=self._monitor,name="recording-catalog",daemon=True)
        self.monitor.start()
        return self

    def stop(self):
        self.stopping.set()
        for thread in (self.thread,self.monitor):
            if thread: thread.join(timeout=6)

    def record(self): self.commands.put(('record',None))
    def stop_recording(self): self.commands.put(('stop',None))
    def load(self, identity=None): self.commands.put(('load',identity))
    def live(self): self.commands.put(('live',None))
    def play(self): self.commands.put(('play',None))
    def pause(self): self.commands.put(('pause',None))
    def refresh(self): self.commands.put(('refresh',None))

    def seek(self, seconds):
        if not math.isfinite(seconds):
            return
        with self.lock:
            self.requested_position = seconds  # Coalesce rapid slider changes.

    def _set(self, **values):
        with self.lock:
            self.state.update(values)
            self.revision += 1

    def _json(self, path, body=None):
        try:
            with request(read(self.credentials_path),path,body) as response:
                return json.load(response)
        except HTTPError as error:
            try:
                message = json.load(error).get('error',str(error))
            except (ValueError,TypeError):
                message = str(error)
            raise ValueError(message) from error

    def _refresh(self):
        info = self._json('/recordings')
        self._set(recordings=info['recordings'],active=info['active'])
        self._polled(info.get('error'))

    def _polled(self, error):
        # Replace only a polling error; command errors stay until the next command.
        with self.lock:
            if self.state['error'] in (None,self.polled_error):
                self._set(error=error)
            self.polled_error = error

    def _seek(self, position):
        if not self.index:
            return
        position = max(0., min(self.index.duration,position))
        frame = self.index.snapshot(position)
        with self.lock:
            self.frame = frame
            self.state['position'] = position
            self.revision += 1

    def _command(self, name, value):
        self._set(error=None)
        if name == 'record':
            self._set(mode='live',playing=False)
            self._json('/recordings/start',{})
            self._refresh()
        elif name == 'stop':
            self._json('/recordings/stop',{})
            self._refresh()
        elif name == 'refresh':
            self._refresh()
        elif name == 'live':
            with self.lock:
                self.requested_position = None
            self._set(mode='live',playing=False)
        elif name == 'load':
            self._refresh()
            recordings = self.state['recordings']
            identity = value or (recordings[0]['id'] if recordings else None)
            if identity is None:
                raise ValueError('Record and stop a capture first')
            self._set(busy=True,playing=False)
            try:
                with request(read(self.credentials_path),'/recordings/'+identity) as response:
                    data = response.read(MAX_BYTES+65537)
                if len(data)>MAX_BYTES+65536:
                    raise ValueError('Recording exceeds playback size limit')
                index = RecordingIndex(iter(data.splitlines()))
                self.index = index
                self._seek(0.)
                self._set(mode='recorded',duration=index.duration,created=index.header['created'],file=identity+'.jsonl')
            finally:
                self._set(busy=False)
        elif name == 'play':
            if self.index and self.state['mode']=='recorded':
                if self.state['position'] >= self.index.duration:
                    self._seek(0.)
                self._set(playing=True)
        elif name == 'pause':
            self._set(playing=False)

    def _monitor(self):
        # A slow/disconnected service must not delay scrubbing a loaded file.
        while not self.stopping.is_set():
            self._poll()
            self.stopping.wait(1.)

    def _poll(self):
        try:
            self._refresh()
        except (OSError,ValueError,KeyError,TypeError) as error:
            with self.lock:
                self.polled_error = str(error)
                self._set(error=self.polled_error)

    def _run(self):
        tick = time.monotonic()
        while not self.stopping.is_set():
            now = time.monotonic()
            elapsed = min(.1,now-tick)
            tick = now
            try:
                try:
                    command = self.commands.get_nowait()
                except queue.Empty:
                    command = None
                if command:
                    self._command(*command)
                with self.lock:
                    position = self.requested_position
                    self.requested_position = None
                if position is not None and self.state['mode']=='recorded':
                    self._set(playing=False)
                    self._seek(position)
                elif self.state['playing']:
                    self._seek(self.state['position']+elapsed)
                    if self.state['position']>=self.state['duration']:
                        self._set(playing=False)
            except (OSError,ValueError,KeyError,TypeError,StopIteration) as error:
                self._set(error=str(error),playing=False,busy=False)
            self.stopping.wait(.03)

    def snapshot(self):
        live = self.client.snapshot()
        with self.lock:
            frame = dict(self.frame) if self.state['mode']=='recorded' and self.frame else live
            frame['revision'] = (self.state['mode'],frame.get('revision',0),self.revision)
            frame['timeline'] = dict(self.state)
            return frame
