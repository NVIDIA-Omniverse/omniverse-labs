# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned, asynchronous SDK workers. No imports or waits on the UI thread."""
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from .owned_process import OwnedProcess


class LibraryWorker:
    def __init__(self, runtime_root, name, environment=None):
        self.root, self.name = Path(runtime_root), name
        self.python = self.root/f'{environment or name}-env/Scripts/python.exe'
        if not self.python.is_file():
            raise RuntimeError('Install the new rooms with tools/setup_extended_libraries.ps1.')
        self.requests, self.results = queue.Queue(), queue.Queue()
        self.stop = threading.Event()
        self.process = None
        self.ready = False
        self.error = None
        self.serial = 0
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._run, name='OV Lab '+name, daemon=True)
        self.thread.start()

    def submit(self, **payload):
        self.serial += 1
        self.started = time.monotonic()
        self.requests.put({'id': self.serial, **payload})
        return self.serial

    def invalidate(self):
        self.serial += 1

    def _run(self):
        env = os.environ.copy()
        for key in ('PYTHONPATH', 'PYTHONHOME', 'CARB_APP_PATH', 'OV_PXR_PLUGINPATH_2511',
                    'PXR_PLUGINPATH_NAME', 'OMNI_PLUGINS_BASE_PATH', 'OMNI_USD_PLUGINS_BASE_PATH'):
            env.pop(key, None)
        env['PYTHONNOUSERSITE'] = '1'
        try:
            with (self.root/f'{self.name}-worker.log').open('w', encoding='utf-8') as log:
                self.process = OwnedProcess(
                    [str(self.python), '-u', str(Path(__file__).with_name(self.name+'_worker.py')),
                     '--runtime-root', str(self.root)], env=env, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=log, text=True, encoding='utf-8', bufsize=1,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                child = self.process.process
                hello = json.loads(child.stdout.readline())
                if not hello.get('ready') or not self.process.owns(hello['pid']):
                    raise RuntimeError('The library worker could not initialize.')
                self.ready = True
                while not self.stop.is_set():
                    try:
                        request = self.requests.get(timeout=.1)
                    except queue.Empty:
                        if self.process.poll() is not None:
                            raise RuntimeError('The library worker stopped. Restart the lab to retry.')
                        continue
                    child.stdin.write(json.dumps(request)+'\n')
                    child.stdin.flush()
                    line = child.stdout.readline()
                    if not line:
                        raise RuntimeError('The library worker stopped. Restart the lab to retry.')
                    self.results.put(json.loads(line))
        except Exception as exc:
            if not self.stop.is_set():
                self.error = str(exc) or type(exc).__name__
                self.results.put({'id': None, 'error': self.error})
        finally:
            if self.process:
                child = self.process.process
                if self.process.poll() is None:
                    try:
                        child.stdin.write('{"close":true}\n')
                        child.stdin.flush()
                        self.process.wait(timeout=1)
                    except (OSError, subprocess.TimeoutExpired):
                        self.process.terminate()
                for stream in (child.stdin, child.stdout):
                    if stream:
                        try:
                            stream.close()
                        except OSError:
                            # A worker terminated during startup can leave a broken pipe.
                            # Still release every owned process handle.
                            pass
                self.process.release()

    def poll(self):
        latest = None
        while True:
            try:
                result = self.results.get_nowait()
            except queue.Empty:
                break
            if result['id'] in (None, self.serial):
                latest = result
        return latest

    def close(self):
        self.stop.set()
        self.thread.join(timeout=.2)
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.thread.join(timeout=2)
