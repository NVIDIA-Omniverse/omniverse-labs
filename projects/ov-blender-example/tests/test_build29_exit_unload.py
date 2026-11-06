# SPDX-License-Identifier: Apache-2.0
"""The exit-time unload and the worker's exit after socket close are timed (build 29 review)."""
import importlib.util
import json
from pathlib import Path
import re
import socket
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'addon'))
from ovrtx_blender_example.direct_runtime_session import RuntimeSession

spec = importlib.util.spec_from_file_location('public_sdk_worker_exit_test', ROOT / 'runtime/public_sdk_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class FakeRenderer:
    """Times its unload stages like RendererBackend; detach and destroy take measurable time."""

    def __init__(self, output_root):
        self.loaded = True
        self.unload_timings = {}

    def dispatch(self, op, params):
        if op == 'unload':
            timings = self.unload_timings = {}
            if self.loaded:
                for stage, seconds in (('detach', .15), ('renderer_destroy', .1)):
                    started = time.monotonic()
                    time.sleep(seconds)
                    timings[stage] = time.monotonic() - started
                self.loaded = False
            return {'unloaded': True}
        return {}


def run_worker(monkeypatch, tmp_path, requests=()):
    identity = {'protocol': 2, 'session': 'a' * 64, 'epoch': 1}
    ours, theirs = socket.socketpair()
    for number, op in enumerate(requests):
        ours.sendall(json.dumps({'id': number, 'op': op, 'params': {}, **identity}).encode() + b'\n')
    ours.shutdown(socket.SHUT_WR)

    def receive(fd, shared_stdin, timeout_s):
        worker.BOOTSTRAP_IDENTITY.clear()
        worker.BOOTSTRAP_IDENTITY.update(identity)
        return theirs

    backend = ModuleType('public_sdk_backend')
    backend.RendererBackend = FakeRenderer
    monkeypatch.setitem(sys.modules, 'public_sdk_backend', backend)
    monkeypatch.setitem(sys.modules, 'runtime_identity', SimpleNamespace(verified_identity=lambda: 'lock'))
    monkeypatch.setattr(worker, 'receive_control_socket', receive)
    monkeypatch.setattr(worker, 'faulthandler', SimpleNamespace(
        enable=lambda: None, dump_traceback_later=lambda *a, **k: None, cancel_dump_traceback_later=lambda: None))
    monkeypatch.setenv('OMNICLIENT_HUB_MODE', 'disabled')
    monkeypatch.setattr(sys, 'argv', ['worker', '--kind', 'renderer', '--fd', '0', '--output-root', str(tmp_path)])
    try:
        assert worker.main() == 0
        replies = ours.makefile('rb').read().splitlines()
    finally:
        ours.close()
    return [json.loads(reply) for reply in replies]


def test_worker_logs_exit_unload_duration(monkeypatch, tmp_path, capsys):
    # The add-on closes the socket without an explicit unload: the finally path unloads.
    run_worker(monkeypatch, tmp_path)
    lines = [line for line in capsys.readouterr().err.splitlines() if 'exit-unload' in line]
    assert len(lines) == 1
    match = re.fullmatch(r'public_sdk exit-unload ([0-9.]+)s kind=renderer detach=([0-9.]+)s renderer_destroy=([0-9.]+)s',
                         lines[0])
    assert match, lines[0]
    total, detach, destroy = map(float, match.groups())
    assert detach >= .14 and destroy >= .09
    # Each value is logged rounded to 1 ms: allow that rounding in the sum.
    assert total >= detach + destroy - .002


def test_exit_unload_after_explicit_unload_has_no_stages(monkeypatch, tmp_path, capsys):
    replies = run_worker(monkeypatch, tmp_path, ['unload'])
    assert replies[0]['ok'] and replies[0]['result'] == {'unloaded': True}
    err = capsys.readouterr().err.splitlines()
    # The explicit unload keeps its begin/end lines and adds its stage breakdown.
    assert re.fullmatch(r'public_sdk end 0 unload [0-9.]+s', err[1])
    assert re.fullmatch(r'public_sdk unload-stages 0 detach=[0-9.]+s renderer_destroy=[0-9.]+s', err[2])
    # Nothing was left to tear down at socket close.
    assert re.fullmatch(r'public_sdk exit-unload [0-9.]+s kind=renderer', err[3])


FAKE_WORKER = "import sys\nsys.path.insert(0, " + repr(str(ROOT / 'runtime')) + ")\n" + r'''
import argparse, json, os, time
from public_sdk_worker import receive_control_socket
p = argparse.ArgumentParser()
p.add_argument('--kind'); p.add_argument('--fd', type=int); p.add_argument('--output-root')
p.add_argument('--socket-share-stdin', action='store_true')
p.add_argument('--bootstrap-timeout', type=float, default=15)
a = p.parse_args()
connection = receive_control_socket(a.fd, shared_stdin=a.socket_share_stdin, timeout_s=a.bootstrap_timeout)
for line in connection.makefile('rb'):
    req = json.loads(line)
    connection.sendall(json.dumps({'id': req['id'], 'epoch': req['epoch'], 'session': req['session'],
                                   'pid': os.getpid(), 'ok': True, 'result': {}}).encode() + b'\n')
time.sleep(float(os.environ['FAKE_TEARDOWN_S']))
'''


def fake_session(tmp_path, teardown_s):
    script = tmp_path / 'worker.py'
    script.write_text(FAKE_WORKER)
    return RuntimeSession(sys.executable, script, 'renderer', tmp_path / 'renderer', timeout_s=5,
                          environment={'FAKE_TEARDOWN_S': str(teardown_s)})


def test_stop_process_records_close_to_exit_time(tmp_path):
    session = fake_session(tmp_path, .6)
    session.request('load', {'scene_path': '/accepted.usda'})
    pid = session.pid
    session.close()
    stop = session.diagnostics()['last_stop']
    assert stop['epoch'] == 1 and stop['pid'] == pid
    assert stop['socket_closed'] and stop['waited']
    assert not stop['terminated'] and not stop['killed']
    assert stop['returncode'] == 0
    assert .5 <= stop['close_to_exit_s'] < 10
    # The record survives the Blender process, beside worker.log.
    written = json.loads((tmp_path / 'renderer/epoch-1/process-exit.json').read_text())
    assert written == stop
    assert session.diagnostics()['stops'] == [stop]


def test_stop_process_records_termination(tmp_path):
    session = fake_session(tmp_path, 30)
    session.exit_wait_s = .3
    session.request('load', {'scene_path': '/accepted.usda'})
    session.close()
    stop = session.diagnostics()['last_stop']
    assert stop['waited'] and stop['terminated']
    assert stop['returncode'] not in (None, 0)
    assert .3 <= stop['close_to_exit_s'] < 5


def test_controller_reports_worker_stop():
    from ovrtx_blender_example.direct_sdk_client import PublicSdkRenderClient
    from ovrtx_blender_example.ovrtx_session_controller import OvrtxSessionController
    stop = {'epoch': 1, 'pid': 7, 'close_to_exit_s': 1.25, 'terminated': False}
    client = PublicSdkRenderClient.__new__(PublicSdkRenderClient)
    client.runtime = SimpleNamespace(diagnostics=lambda: {'last_stop': stop})
    assert client.worker_stop == stop
    controller = OvrtxSessionController()
    controller._client = SimpleNamespace(delete_simulation=lambda sid: 'stopped', shutdown=lambda: 'stopped',
                                         shutdown_after_delete_failure=True, worker_stop=stop)
    controller._simulation_id = 'sim'
    assert controller._deactivate_active() == 'stopped'
    diagnostics = controller.diagnostics()
    assert diagnostics['worker_stop'] == stop
    assert diagnostics['lifecycle_events'][-1]['worker_stop'] == stop
