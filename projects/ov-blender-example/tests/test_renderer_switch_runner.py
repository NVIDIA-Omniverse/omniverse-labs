"""The process observer must identify an actual owned subprocess without bpy."""
import importlib.util
from pathlib import Path
import subprocess
import sys

spec = importlib.util.spec_from_file_location('switch_runner', Path(__file__).parents[1]/'scripts/run_blender_renderer_switch.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_actual_child_process_identity_and_parentage():
    child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE)
    try:
        table = runner.process_table()
        assert table[child.pid][0] == __import__('os').getpid()
        first = runner.process_identity(child.pid)
        second = runner.process_identity(child.pid)
        assert first == second
        assert first['pid'] == child.pid
        assert first['created']
        assert Path(first['executable']).samefile(sys.executable)
    finally:
        child.communicate(timeout=10)


def test_recycled_descendant_cannot_be_adopted_or_parent_new_children():
    root = {'pid': 1, 'created': 10, 'executable': 'blender'}
    original = {'pid': 2, 'created': 20, 'executable': 'worker'}
    identities = {1: root, 2: {**original, 'created': 30},
                  3: {'pid': 3, 'created': 40, 'executable': 'unrelated'}}
    owned = {2: original}
    assert runner.discover_owned({2: (1, 'other'), 3: (2, 'other')}, root, owned, identities.__getitem__) == {}
    assert owned == {2: original}
    assert runner.live_owned(owned, identities.__getitem__) == {}
    changes = runner.changed_worker_identities({2}, owned, identities.__getitem__)
    assert changes == [{'expected': original, 'observed': identities[2]}]


def test_child_older_than_current_parent_is_not_owned():
    root = {'pid': 1, 'created': 30, 'executable': 'blender'}
    identities = {1: root, 2: {'pid': 2, 'created': 20, 'executable': 'unrelated'}}
    assert runner.discover_owned({2: (1, 'other')}, root, {}, identities.__getitem__) == {}


def test_termination_handle_rejects_wrong_creation_then_stops_actual_owned_child():
    child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE)
    try:
        identity = runner.process_identity(child.pid)
        runner.terminate_owned({**identity, 'created': int(identity['created']) + 1})
        assert child.poll() is None
        runner.terminate_owned(identity)
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.communicate(timeout=10)
