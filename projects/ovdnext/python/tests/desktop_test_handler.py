# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disposable real child process for desktop request argv, polling and reaping."""
import json
import subprocess
import sys

from ovd_ui import desktop


def missing_handler(target):
    raise FileNotFoundError('No disposable test handler')


def install_handler(tmp_path, monkeypatch, code='pass'):
    binary = tmp_path / 'desktop_handler.py'
    log = tmp_path / 'desktop_arguments.json'
    binary.write_text('import json,sys,time\nfrom pathlib import Path\n'
        + f'Path({str(log)!r}).write_text(json.dumps(sys.argv[1:]))\n' + code + '\n', encoding='utf-8')
    def launch(target):
        return subprocess.Popen([sys.executable, '-B', str(binary), target],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    monkeypatch.setattr(desktop, '_launch', launch)
    return log
