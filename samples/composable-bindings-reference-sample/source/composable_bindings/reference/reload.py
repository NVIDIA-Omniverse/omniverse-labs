# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A single service owns validated configuration revisions and keeps the last good one."""
import json
import os
import tempfile
import threading
from pathlib import Path
from .config import read, load_configuration


def atomic_json(path, value):
    path = Path(path)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


class ConfigurationWatcher:
    def __init__(self, service, selector):
        self.service, self.selector = service, Path(selector)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name='configuration-watcher', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)

    def run(self):
        while not self.stop_event.is_set():
            try:
                from .composition import refresh
                refresh(self.selector.parent)
                selection = read(self.selector)
                if selection.get('error'):
                    self.service.report_error(selection['error'])
                    self.stop_event.wait(.25)
                    continue
                source, bindings, view = load_configuration(selection['directory'], selection.get('view'), selection.get('bindings'))
                self.service.configure(source, bindings, view)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                self.service.report_error('Configuration rejected (' + type(error).__name__ + '); using last valid revision')
            self.stop_event.wait(.25)
