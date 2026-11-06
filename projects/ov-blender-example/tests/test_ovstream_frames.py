# SPDX-License-Identifier: Apache-2.0
"""Resource budgets at the external publisher's allocation boundary."""
import importlib.util
from pathlib import Path
import sys
import threading
import types

import pytest


@pytest.fixture
def publisher_module(monkeypatch):
    pytest.importorskip("ovstream")
    pytest.importorskip("warp")
    path = Path(__file__).resolve().parents[1] / "runtime" / "ovstream_frames.py"
    spec = importlib.util.spec_from_file_location("ovstream_frames_budget_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_publication_counts_successes_and_keeps_submitted_timestamp(publisher_module, monkeypatch):
    mod = publisher_module
    publisher = mod.ShmPublisher.__new__(mod.ShmPublisher)
    publisher._publication_sequence = 0
    publisher._stream_name, publisher.epoch = 'test-stream', 'test-epoch'
    publisher._dimensions = (7, 5)
    publisher._buffer = object()
    publisher._event = types.SimpleNamespace(cuda_event=1)
    publisher._held_token = None
    frames = []
    publisher._server = types.SimpleNamespace(stream_video=frames.append)
    monkeypatch.setattr(mod.ovstream.VideoFrame, 'from_cuda_array',
        lambda buffer, **kwargs: kwargs)
    monkeypatch.setattr(mod.time, 'monotonic_ns', lambda: 123456)
    monkeypatch.setattr(mod, '_windows_shm', lambda: pytest.fail('publisher must not map headers'))
    monkeypatch.setattr(mod.os, 'open', lambda *args, **kwargs: pytest.fail('publisher must not open SHM'))
    stream = types.SimpleNamespace(record_event=lambda event: None, cuda_stream=2)
    first = publisher._publish_buffer(7, 5, stream)
    second = publisher._publish_buffer(7, 5, stream)
    assert (first['sequence'], second['sequence']) == (1, 2)
    assert first['capture_timestamp_ns'] == frames[0]['start_timestamp_ns'] == 123456
    assert first['producer_pid'] == mod.os.getpid()
    assert first['copy_token'] != second['copy_token']
    def fail(frame):
        raise RuntimeError('publication failed after possible native write')
    publisher._server.stream_video = fail
    with pytest.raises(RuntimeError, match='publication failed'):
        publisher._publish_buffer(7, 5, stream)
    assert publisher._publication_sequence == 2
    assert publisher._dimensions is None


def test_new_stream_resets_publication_sequence(publisher_module, monkeypatch):
    mod = publisher_module
    publisher = mod.ShmPublisher.__new__(mod.ShmPublisher)
    publisher._publication_sequence = 19
    publisher._max_region_bytes, publisher._min_free_bytes = 10000, 0
    publisher._registry_streams = set()
    publisher.device = types.SimpleNamespace(ordinal=0, context=1)
    publisher._stop_stream = lambda: None
    publisher._write_registry = lambda: None
    calls = []
    server = types.SimpleNamespace(start=lambda config: calls.append(config), close=lambda: None)
    monkeypatch.setattr(mod, '_check_allocation', lambda *args, **kwargs: None)
    monkeypatch.setattr(mod.ovstream, 'Server', lambda kind: server)
    monkeypatch.setattr(mod.wp, 'empty', lambda *args, **kwargs: object())
    monkeypatch.setattr(mod.wp, 'Event', lambda **kwargs: object())
    publisher._start(7, 5)
    assert publisher._publication_sequence == 0
    assert publisher._dimensions == (7, 5) and publisher._server is server
    assert len(calls) == 1


def test_budget_includes_pitch_padding_slot_headers_and_reserve(publisher_module, monkeypatch):
    mod = publisher_module
    monkeypatch.setattr(mod.sys, 'platform', 'linux')
    required = 128 + 2 * (64 + 64 * 5)
    monkeypatch.setattr(mod, '_posix_available_memory', lambda: required+16)
    assert mod._check_allocation(7, 5, max_region_bytes=required, min_free_bytes=16) == required
    with pytest.raises(RuntimeError, match="Insufficient"):
        mod._check_allocation(7, 5, max_region_bytes=required, min_free_bytes=17)


@pytest.mark.parametrize("width,height,budget", [(0,5,10000),(5,8193,10000),(8192,8192,256*1024*1024)])
def test_reject_oversized_allocation_before_touching_shared_memory(publisher_module, monkeypatch, width, height, budget):
    def forbidden():
        raise AssertionError("resource probe should not run after bad dimension/budget")
    monkeypatch.setattr(publisher_module, '_posix_available_memory', forbidden)
    with pytest.raises(ValueError):
        publisher_module._check_allocation(width, height, max_region_bytes=budget, min_free_bytes=0)


def test_partial_transfer_requires_cancel_and_preserves_publisher_identity(publisher_module):
    publisher = publisher_module.ShmPublisher.__new__(publisher_module.ShmPublisher)
    publisher._lock = threading.Lock()
    publisher._held_token = "current"
    publisher._raw_transfer = types.SimpleNamespace(index=0, count=3, payload=object(), failed=False, stream=None)
    publisher.epoch = "persistent-session"
    publisher._server = object()
    server = publisher._server
    with pytest.raises(RuntimeError, match="Incomplete"):
        publisher.release("current")
    with pytest.raises(RuntimeError, match="does not match"):
        publisher.cancel_raw_transfer("old-token")
    assert publisher._raw_transfer is not None
    publisher.cancel_raw_transfer("current")
    assert publisher._raw_transfer is None and publisher._held_token is None
    assert publisher.epoch == "persistent-session" and publisher._server is server


def test_windows_allocation_uses_commit_memory_not_posix_filesystem(publisher_module, monkeypatch):
    mod = publisher_module
    monkeypatch.setattr(mod.sys, 'platform', 'win32')
    monkeypatch.setattr(mod, '_windows_shm', lambda: types.SimpleNamespace(available_memory=lambda:1024))
    required=128+2*(64+64*5)
    assert mod._check_allocation(7,5,max_region_bytes=1024,min_free_bytes=0)==required
    with pytest.raises(RuntimeError,match='Windows'):
        mod._check_allocation(7,5,max_region_bytes=1024,min_free_bytes=1024)


def test_last_chunk_releases_owned_payload_and_rejects_next(publisher_module):
    publisher = publisher_module.ShmPublisher.__new__(publisher_module.ShmPublisher)
    publisher._lock = threading.Lock()
    publisher._held_token = "last"
    publisher._raw_transfer = types.SimpleNamespace(index=2, count=3, payload=object(), failed=False)
    with pytest.raises(RuntimeError, match="Last raw chunk"):
        publisher.next_raw_chunk("last")
    publisher.release("last")
    assert publisher._raw_transfer is None and publisher._held_token is None
    with pytest.raises(RuntimeError, match="does not match"):
        publisher.release("last")


def test_failed_chunk_cannot_be_retried_or_acknowledged(publisher_module, monkeypatch):
    publisher = publisher_module.ShmPublisher.__new__(publisher_module.ShmPublisher)
    publisher._lock = threading.Lock()
    publisher._held_token = "current"
    stream = object()
    publisher._raw_transfer = types.SimpleNamespace(index=0, count=3, payload=object(), failed=False, stream=stream)
    publisher.device = object()
    monkeypatch.setattr(publisher_module.wp, "get_stream", lambda _: None)
    completions = []
    monkeypatch.setattr(publisher_module.wp, 'synchronize_stream', completions.append)
    def failed_publish(_):
        raise RuntimeError("injected publication failure")
    monkeypatch.setattr(publisher, "_publish_raw_chunk", failed_publish)
    with pytest.raises(RuntimeError, match="injected"):
        publisher.next_raw_chunk("current")
    with pytest.raises(RuntimeError, match="requires cancellation"):
        publisher.next_raw_chunk("current")
    with pytest.raises(RuntimeError, match="Incomplete"):
        publisher.release("current")
    publisher.cancel_raw_transfer("current")
    assert publisher._raw_transfer is None and publisher._held_token is None
    assert completions == [stream]


def test_completed_payload_reuse_requires_ack_and_gpu_completion(publisher_module, monkeypatch):
    mod = publisher_module
    publisher = mod.ShmPublisher.__new__(mod.ShmPublisher)
    publisher._lock = threading.Lock()
    publisher._held_token = "last"
    payload = types.SimpleNamespace(size=64)
    publisher._raw_transfer = types.SimpleNamespace(index=0, count=1, payload=payload, failed=False)
    publisher._event = object()
    publisher.device = object()
    completions = []
    monkeypatch.setattr(mod.wp, "synchronize_event", lambda event: completions.append(event))
    with pytest.raises(RuntimeError, match="still in use"):
        publisher._take_raw_payload(64)
    publisher.release("last")
    assert completions == [publisher._event]
    assert publisher._take_raw_payload(64) is payload
    assert publisher._raw_payload_cache is None


def test_payload_reuse_rejects_wrong_size_and_failed_completion(publisher_module, monkeypatch):
    mod = publisher_module
    publisher = mod.ShmPublisher.__new__(mod.ShmPublisher)
    publisher._lock = threading.Lock()
    publisher._held_token = None
    publisher._raw_transfer = None
    publisher.device = object()
    publisher._raw_payload_cache = types.SimpleNamespace(size=64)
    fresh = object()
    monkeypatch.setattr(mod.wp, "empty", lambda *args, **kwargs: fresh)
    assert publisher._take_raw_payload(32) is fresh
    publisher._held_token = "last"
    publisher._raw_transfer = types.SimpleNamespace(index=0, count=1, payload=object(), failed=False, stream=object())
    publisher._event = object()
    def fail(_):
        raise RuntimeError("CUDA completion failed")
    monkeypatch.setattr(mod.wp, "synchronize_event", fail)
    monkeypatch.setattr(mod.wp, "synchronize_stream", fail)
    with pytest.raises(RuntimeError, match="completion failed"):
        publisher.release("last")
    assert publisher._held_token == "last" and publisher._raw_payload_cache is None
    with pytest.raises(RuntimeError, match="completion failed"):
        publisher.cancel_raw_transfer("last")
    assert publisher._held_token == "last" and publisher._raw_transfer is not None
    monkeypatch.setattr(mod.wp, "synchronize_stream", lambda stream: None)
    publisher.cancel_raw_transfer("last")
    assert publisher._raw_payload_cache is None


@pytest.mark.parametrize("budget", [0, True, 4095, 64*1024*1024+1])
def test_invalid_carrier_budget_rejected_before_sdk_initialization(publisher_module, monkeypatch, budget):
    monkeypatch.setattr(publisher_module.ovstream, "initialize", lambda: pytest.fail("invalid budget must not initialize SDK"))
    with pytest.raises(ValueError, match="carrier budget"):
        publisher_module.ShmPublisher(raw_chunk_bytes=budget)


def test_large_windows_carrier_still_counts_both_ring_slots(publisher_module):
    with pytest.raises(ValueError, match="budget"):
        publisher_module._check_allocation(8192,2048,max_region_bytes=64*1024*1024,min_free_bytes=0)


@pytest.mark.parametrize('width,height,expected', [
    (620, 531, True), (1280, 720, True), (1920, 1080, True),
    (2530, 1305, True), (3840, 2160, True), (8192, 8192, False),
])
def test_explicit_hdr_upload_uses_whole_message_budget(
        publisher_module, width, height, expected):
    assert publisher_module._hdr_upload_fits(width, height, width*height*8) is expected


def test_oversized_explicit_staging_rejects_before_gpu_or_shm_allocation(publisher_module, monkeypatch):
    mod = publisher_module
    publisher = mod.ShmPublisher.__new__(mod.ShmPublisher)
    publisher._lock = threading.Lock()
    publisher._closed, publisher._held_token = False, None
    publisher.device = object()
    source = types.SimpleNamespace(shape=(2048, 8192, 4), strides=(8192*8, 8, 2),
        ndim=3, ptr=1, dtype=mod.wp.float16, device=publisher.device)
    monkeypatch.setattr(mod.wp, 'array', lambda **kw: object())
    publisher._start = lambda *a: pytest.fail('Oversized staging must not allocate a ring')
    publisher._take_raw_payload = lambda *a: pytest.fail('Oversized staging must not allocate GPU storage')
    with pytest.raises(ValueError, match='Requested HDR staging exceeds'):
        publisher.publish_raw_planes_cuda([('linear_rgba16f', 'rgba16f', source)], presentation_rgba32f=True)


@pytest.mark.parametrize('code', [5, 32, 33])
def test_registry_retries_windows_contention_without_losing_previous_registry(
        publisher_module, monkeypatch, tmp_path, code):
    mod = publisher_module
    publisher = mod.ShmPublisher.__new__(mod.ShmPublisher)
    publisher._registry_path = tmp_path / 'registry.json'
    publisher._registry_path.write_text('previous-complete-registry')
    publisher._registry_streams = {'blender-' + 'a' * 32}
    publisher.epoch = 'epoch'
    original = mod.os.replace
    calls, sleeps = [], []
    def replace(source, target):
        calls.append(source)
        if len(calls) < 3:
            assert target.read_text() == 'previous-complete-registry'
            error = PermissionError('sharing contention')
            error.winerror = code
            raise error
        original(source, target)
    monkeypatch.setattr(mod.sys, 'platform', 'win32')
    monkeypatch.setattr(mod.os, 'replace', replace)
    monkeypatch.setattr(mod.time, 'sleep', sleeps.append)
    publisher._write_registry()
    assert sleeps == [.01, .02] and len(set(calls)) == 1
    assert mod.json.loads(publisher._registry_path.read_text())['streams'] == sorted(publisher._registry_streams)
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('platform,code,attempts', [('win32', 5, 6), ('win32', 87, 1), ('linux', 5, 1)])
def test_registry_permanent_failure_is_bounded_and_propagates(
        publisher_module, monkeypatch, tmp_path, platform, code, attempts):
    mod = publisher_module
    publisher = mod.ShmPublisher.__new__(mod.ShmPublisher)
    publisher._registry_path = tmp_path / 'registry.json'
    publisher._registry_path.write_text('previous')
    publisher._registry_streams, publisher.epoch = set(), 'epoch'
    calls, sleeps = [], []
    def replace(*args):
        calls.append(args)
        error = PermissionError('persistent denial')
        error.winerror = code
        raise error
    monkeypatch.setattr(mod.sys, 'platform', platform)
    monkeypatch.setattr(mod.os, 'replace', replace)
    monkeypatch.setattr(mod.time, 'sleep', sleeps.append)
    with pytest.raises(PermissionError, match='persistent denial'):
        publisher._write_registry()
    assert len(calls) == attempts and sum(sleeps) <= .311
    assert publisher._registry_path.read_text() == 'previous'
    assert not list(tmp_path.glob('*.tmp'))


def test_registry_teardown_error_releases_cuda_and_shutdown_but_remains_visible(publisher_module, monkeypatch):
    mod = publisher_module
    publisher = mod.ShmPublisher.__new__(mod.ShmPublisher)
    publisher._lock, publisher._closed = threading.Lock(), False
    publisher._held_token, publisher._raw_transfer, publisher._raw_payload_cache = 'held', object(), object()
    publisher._buffer = publisher._event = object()
    publisher._dimensions, publisher._stream_name = (7, 5), 'stream'
    publisher._registry_streams = {'stream'}
    calls = []
    publisher._server = types.SimpleNamespace(stop=lambda: calls.append('stop'), close=lambda: calls.append('close'))
    def fail():
        raise PermissionError('persistent registry denial')
    publisher._write_registry = fail
    monkeypatch.setattr(mod.ovstream, 'shutdown', lambda: calls.append('shutdown'))
    with pytest.raises(PermissionError, match='registry denial'):
        publisher.close()
    assert calls == ['stop', 'close', 'shutdown']
    assert all(getattr(publisher, field) is None for field in
               ('_server', '_buffer', '_event', '_dimensions', '_held_token', '_raw_transfer', '_raw_payload_cache'))


@pytest.mark.skipif(sys.platform != 'win32', reason='Win32 file sharing contract')
@pytest.mark.parametrize('release_reader', [False, True])
def test_registry_real_windows_reader_contention(publisher_module, tmp_path, release_reader):
    import ctypes
    import time
    mod = publisher_module
    publisher = mod.ShmPublisher.__new__(mod.ShmPublisher)
    publisher._registry_path = tmp_path / 'registry.json'
    publisher._registry_path.write_text('old')
    publisher._registry_streams, publisher.epoch = set(), 'contention'
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                               ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    api.CreateFileW.restype = ctypes.c_void_p
    api.CloseHandle.argtypes, api.CloseHandle.restype = [ctypes.c_void_p], ctypes.c_int
    handle = api.CreateFileW(str(publisher._registry_path), 0x80000000, 1, None, 3, 0, None)
    assert handle not in (None, ctypes.c_void_p(-1).value)
    closed = threading.Event()
    def close_reader():
        assert api.CloseHandle(handle)
        closed.set()
    timer = threading.Timer(.05, close_reader) if release_reader else None
    if timer:
        timer.start()
    started = time.monotonic()
    try:
        if release_reader:
            publisher._write_registry()
            assert mod.json.loads(publisher._registry_path.read_text())['streams'] == []
            assert closed.is_set()
        else:
            with pytest.raises(PermissionError):
                publisher._write_registry()
            assert publisher._registry_path.read_text() == 'old'
        assert time.monotonic()-started < 2
        assert not list(tmp_path.glob('*.tmp'))
    finally:
        if timer:
            timer.join()
        if not closed.is_set():
            close_reader()


def test_diagnostics_separates_active_cache_carrier_ring_and_lease(publisher_module):
    publisher = publisher_module.ShmPublisher.__new__(publisher_module.ShmPublisher)
    publisher._lock = threading.Lock()
    publisher._closed, publisher._held_token = False, 'lease'
    publisher._dimensions, publisher._server = (257, 3), object()
    publisher._registry_streams = {'stream'}
    publisher._raw_transfer = types.SimpleNamespace(layout={'byte_count': 131584}, count=43, index=0)
    publisher._raw_payload_cache = None
    active = publisher.diagnostics()
    assert active['held_leases'] == active['active_transfers'] == 1
    assert active['active_payload_bytes'] == 131584 and active['cached_payload_bytes'] == 0
    assert active['carrier_gpu_bytes'] == 3084
    assert active['shm_region_bytes'] == 128 + 2*(64+1088*3)
    assert active['pending_chunks'] == 42 and active['registered_streams'] == 1
    publisher._raw_transfer, publisher._held_token = None, None
    publisher._raw_payload_cache = types.SimpleNamespace(size=131584)
    cached = publisher.diagnostics()
    assert cached['held_leases'] == cached['active_transfers'] == cached['active_payload_bytes'] == cached['pending_chunks'] == 0
    assert cached['cached_payload_bytes'] == 131584
    # stream_video failure invalidates reuse dimensions, but does not make
    # the existing GPU/ring allocation disappear from ownership accounting.
    publisher._dimensions = None
    publisher._buffer = types.SimpleNamespace(shape=(3, 257, 4))
    failed = publisher.diagnostics()
    assert failed['carrier_gpu_bytes'] == active['carrier_gpu_bytes']
    assert failed['shm_region_bytes'] == active['shm_region_bytes']


def test_close_completes_packing_stream_before_retiring_held_storage(publisher_module, monkeypatch):
    mod = publisher_module
    publisher = mod.ShmPublisher.__new__(mod.ShmPublisher)
    publisher._lock, publisher._closed = threading.Lock(), False
    stream = object()
    publisher._raw_transfer = types.SimpleNamespace(stream=stream)
    publisher._held_token, publisher._raw_payload_cache = 'lease', None
    calls = []
    def sync(value):
        assert value is stream and publisher._raw_transfer is not None
        calls.append('sync')
    monkeypatch.setattr(mod.wp, 'synchronize_stream', sync)
    publisher._stop_stream = lambda: calls.append('stop')
    monkeypatch.setattr(mod.ovstream, 'shutdown', lambda: calls.append('shutdown'))
    publisher.close()
    assert calls == ['sync', 'stop', 'shutdown']
    assert publisher._raw_transfer is None and publisher._held_token is None
