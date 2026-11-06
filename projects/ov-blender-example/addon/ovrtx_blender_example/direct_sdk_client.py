# SPDX-License-Identifier: Apache-2.0
"""External OVStream clients; no NVIDIA SDK is imported in Blender.

The controller owns scene generation, editing and presentation policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import json
import os
import re
import shutil
import socket
import stat
import sys
from pathlib import Path
import tempfile
import threading
import time
from typing import Any
import uuid

from . import color_presentation, rtpt_worker_config, worker_failure_logs
from .ovrtx_runtime_client import RenderClientError, RenderResult, ValueUpdateRecoveryRequired, _flip_payload_rows
from .ovrtx_value_updates import OvrtxValueUpdateResult, _attribute_values, _transform_values

PYTHON_ENV = 'OV_BLENDER_PUBLIC_SDK_PYTHON'
OUTPUT_ENV = 'OV_BLENDER_PUBLIC_SDK_OUTPUT'
OVRTX_CACHE_ENV = 'OV_BLENDER_OVRTX_CACHE_ROOT'


def _unaffected_population_values(values, before, after, *, effective_changed_paths=None):
    # Scoped overlays project the compiled effective changes, not every authored
    # contributor difference. A harmless ancestor over/def history change must
    # not invalidate all accepted descendant poses/material values. Keep subtree
    # invalidation for actual parent, material, creation and deletion edits.
    touched = (set(effective_changed_paths) if effective_changed_paths is not None else
               {key[0] for key in set(before) | set(after) if before.get(key) != after.get(key)})
    return {key: value for key, value in values.items()
            if not any(key[0] == path or key[0].startswith(path.rstrip('/') + '/') for path in touched)}


def _prepare_hdr_upload(render_var):
    return (render_var == 'HdrColor' and sys.platform == 'win32'
            and os.environ.get('OVRTX_HDR_GPU_FLOAT32', '0') != '0')


def enabled() -> bool:
    return True


def worker_script() -> Path:
    from .bundled_runtime import public_sdk_worker_script
    return public_sdk_worker_script()


# Warp's kernel cache and the CUDA JIT cache default to the user profile. Native
# code cannot use a non-ASCII profile folder (Warp's startup message then fails to
# encode in the worker log; the CUDA cache lands in a mis-encoded folder), so for
# such a profile they move beside the runtime's other caches, which are ASCII.
_PROFILE_CACHES = (('WARP_CACHE_PATH', 'LOCALAPPDATA', 'warp'), ('CUDA_CACHE_PATH', 'APPDATA', 'cuda'))


def _worker_environment() -> dict[str, str]:
    """OVRTX 0.6 shader caches must outlive a worker epoch: a cold cache costs
    minutes per renderer start. Keep them with the profile's SDK store, outside
    the locked sidecar; an explicit environment value takes precedence."""
    from . import public_sdk_install
    root = public_sdk_install.ovrtx_cache_root()
    if root is None:
        return {}
    environment = {} if os.environ.get(OVRTX_CACHE_ENV, '').strip() else {OVRTX_CACHE_ENV: str(root)}
    if sys.platform == 'win32':
        for variable, profile, folder in _PROFILE_CACHES:
            if not os.environ.get(variable, '').strip() and not os.environ.get(profile, '').isascii():
                environment[variable] = str(root.parent / folder)
    return environment


# Every worker gets its own output root under %TEMP% (logs, frame resources, and
# without a cache root its DerivedDataCache: 0.3-1.6 GB). A root outlives its Blender
# process, so each new worker removes the roots whose Blender process has exited.
# Only roots this add-on made are touched: the exact name, and the owner marker
# written at creation. The process check is local, so a marked root is removed only
# if its marker names this host: OV_BLENDER_PUBLIC_SDK_OUTPUT may be a folder that
# other machines share. Roots without a host (from versions before the marker, or
# with its first schema) are recognised by the layout those versions wrote: only
# accepted-scene and epoch-N folders, every epoch holding the worker.log the add-on
# opened. Their host is unknown, so they are removed only from the default per-user
# %TEMP%. A root still in use (a file open in it) fails to delete and is retried at
# the next worker start.
OUTPUT_ROOT_MARKER = 'ov-blender-owner.json'
_OUTPUT_ROOT_NAME = re.compile(r'ov-public-(renderer|physics)-([1-9][0-9]{0,9})-[0-9a-f]{10}')
_LEGACY_ROOT_ENTRY = re.compile(r'epoch-[1-9][0-9]{0,5}|accepted-scene')
_MARKER_MAX_BYTES = 4096
_cleanup_running = threading.Lock()


def write_output_root_marker(root, kind, pid=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    record = dict(schema=2, application='ovrtx_blender_example', kind=kind,
                  pid=os.getpid() if pid is None else pid, host=socket.gethostname())
    (root / OUTPUT_ROOT_MARKER).write_text(json.dumps(record), encoding='utf-8')


def _is_link(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def _owned_output_root(root, pid, legacy_roots):
    """Whether ``root`` (named for ``pid``) is an output root this add-on made on
    this host; with ``legacy_roots``, also one without a host."""
    if _is_link(root) or not root.is_dir():
        return False
    marker = root / OUTPUT_ROOT_MARKER
    if marker.exists():
        if _is_link(marker) or not marker.is_file() or marker.stat().st_size > _MARKER_MAX_BYTES:
            return False
        try:
            record = json.loads(marker.read_text(encoding='utf-8'))
        except ValueError:
            return False
        if not (isinstance(record, dict) and record.get('application') == 'ovrtx_blender_example'
                and record.get('pid') == pid):
            return False
        if record.get('schema') == 2:
            return record.get('host') == socket.gethostname()
        return record.get('schema') == 1 and legacy_roots
    if not legacy_roots:
        return False
    entries = list(root.iterdir())
    epochs = [entry for entry in entries if entry.name.startswith('epoch-')]
    return bool(epochs) and all(
        _LEGACY_ROOT_ENTRY.fullmatch(entry.name) and not _is_link(entry) and entry.is_dir()
        for entry in entries) and all((epoch / 'worker.log').is_file() for epoch in epochs)


def _remove_output_root(root, kind, pid):
    """Delete one owned root; the marker goes last so a partial removal is retried."""
    size = 0
    def failed(error):
        raise error
    for parent, directories, files in os.walk(root, onerror=failed, followlinks=False):
        for name in (*directories, *files):
            path = Path(parent) / name
            if _is_link(path):
                raise OSError(f'Link inside a worker output root: {path}')
            if name in files:
                size += path.lstat().st_size
    marker = root / OUTPUT_ROOT_MARKER
    if not marker.exists():
        write_output_root_marker(root, kind, pid)
    for entry in root.iterdir():
        if entry.name != OUTPUT_ROOT_MARKER:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    marker.unlink()
    root.rmdir()
    return size


def remove_stale_output_roots(base, *, running=None, legacy_roots=False):
    """Remove this add-on's output roots under ``base`` whose Blender process is gone.

    Returns ``(removed root names, bytes freed)``. Roots of a live process, roots
    whose marker names another host, roots without a host unless ``legacy_roots``
    (``base`` is the default per-user %TEMP%), roots without the marker or legacy
    layout, and roots that fail to delete are kept.
    """
    if running is None:
        from .session_lifecycle import pid_running as running
    removed, freed = [], 0
    for root in sorted(Path(base).glob('ov-public-*')):
        match = _OUTPUT_ROOT_NAME.fullmatch(root.name)
        if not match:
            continue
        pid = int(match[2])
        try:
            if pid == os.getpid() or running(pid) or not _owned_output_root(root, pid, legacy_roots):
                continue
            freed += _remove_output_root(root, match[1], pid)
        except OSError:
            continue
        removed.append(root.name)
    return removed, freed


def _start_stale_root_cleanup(base, legacy_roots):
    if not _cleanup_running.acquire(blocking=False):
        return
    def run():
        try:
            removed, freed = remove_stale_output_roots(base, legacy_roots=legacy_roots)
            if removed:
                print(f'OVRTX: removed {len(removed)} stale worker output folders '
                      f'({freed / 2**20:.0f} MB) from {base}')
        except Exception as error:  # noqa: BLE001 - cleanup must never fail a render
            print(f'OVRTX: stale worker output cleanup failed: {error}')
        finally:
            _cleanup_running.release()
    try:
        threading.Thread(target=run, name='ov-public-output-cleanup', daemon=True).start()
    except BaseException:
        _cleanup_running.release()
        raise


def _session(kind: str):
    from .direct_runtime_session import RuntimeSession
    # Preserve the venv entrypoint: resolving its symlink loses pyvenv.cfg.
    from .bundled_runtime import public_sdk_python
    configured = public_sdk_python()
    if not configured:
        from . import public_sdk_install
        status = public_sdk_install.status()
        raise RenderClientError(str(status.get('error') or status.get('message') or
            'Rendering setup is not ready. Open the add-on preferences to check progress.'))
    python = Path(configured).expanduser().absolute()
    if not python.is_file():
        raise RenderClientError(f'Missing public SDK interpreter: {python}')
    base = Path(os.environ.get(OUTPUT_ENV, tempfile.gettempdir()))
    root = base / f'ov-public-{kind}-{os.getpid()}-{uuid.uuid4().hex[:10]}'
    write_output_root_marker(root, kind)
    _start_stale_root_cleanup(base, legacy_roots=OUTPUT_ENV not in os.environ)
    return RuntimeSession(python=str(python), worker_script=str(worker_script()),
                          kind=kind, output_root=root, environment=_worker_environment(),
                          timeout_s=float(os.environ.get('OV_BLENDER_PUBLIC_SDK_TIMEOUT_S', '600')))


def _plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return [_plain(item) for item in value]


def _flatten_settings(tree, prefix=''):
    result = {}
    for key, value in tree.items():
        path = prefix + '/' + key
        if isinstance(value, dict):
            result.update(_flatten_settings(value, path))
        else:
            result[path] = value
    return result


from .value_change_tracking import AcknowledgedValueLedger


@dataclass
class _State:
    spec: Any
    output_size: tuple[int, int] | None = None
    simulation_id: str = ""
    samples: int = 0
    session_samples: int = 0
    time_ns: int = 0
    epoch: int = 0
    desired: dict = field(default_factory=dict)
    accepted_values: AcknowledgedValueLedger = field(default_factory=AcknowledgedValueLedger)
    failure: Exception | None = None
    recovery_pending: bool = False
    collecting: bool = False
    recovery_baseline: dict | None = None
    stage_metadata: dict[str, Any] | None = None
    population_snapshot: Any = None
    population_revision: int = 0
    native_scene_path: str = ''
    native_population_path: str = ''
    native_source_paths: set = field(default_factory=set)
    inline_source_paths: dict = field(default_factory=dict)
    pending_source_paths: dict = field(default_factory=dict)


@dataclass
class _PendingPublicRender:
    client: Any
    simulation_id: str
    token: str
    sensors: tuple[str, ...]
    render_var: str
    epoch: int
    simulation_time_ns: int
    session_samples: int
    native_started_ns: int
    cancelled: bool = False
    resolved: bool = False

    @property
    def tickets(self):
        # Existing viewport pipeline expects cancellation/wait capabilities on
        # per-output tickets. Both operations stay on its serialized owner.
        return {self.render_var: self}

    def cancel(self):
        if self.cancelled or self.resolved:
            return
        self.cancelled = True
        # Logical cancellation is immediate; the renderer may still be reading
        # this stage. Keep the ticket registered until native completion is
        # acknowledged, including when this first cancellation RPC fails.
        self.client._cancel_native_render(self, timeout_ns=0)

    def wait(self, timeout_ms):
        if self.cancelled or self.resolved or self.client.runtime.epoch != self.epoch or self.client.runtime.returncode is not None:
            return
        self.client._call('wait_render', {'ticket': self.token,
                                         'timeout_ns': min(50_000_000, max(0, int(timeout_ms * 1e6)))})


class PublicSdkRenderClient:
    supports_direct_render_parameters = True
    camera_successor_deferral = True
    worker_owned = True
    # Explicit teardown can discard this owner's uncertain process without
    # sending another RPC or replaying the ambiguous operation.
    shutdown_after_delete_failure = True

    def __init__(self, request=None):
        from .render_capabilities import validate_request
        validate_request(request)
        self.runtime = _session('renderer')
        self.request = request
        self.states: dict[str, _State] = {}
        self._pending_native_renders = {}
        self.startup_diagnostics = {}
        self.last_render_timings = {}
        self.last_value_update_timings = {}
        self._reader = None
        self._source_store = None
        # Worker failures already surfaced: recoveries by failed epoch, and the
        # worker logs whose error lines were mirrored to the console.
        self._noticed_epoch = 0
        self._noticed_failed_epoch = 0
        self._mirrored_logs = set()

    def _scene_store(self):
        if self._source_store is None:
            from .immutable_scene_store import ImmutableSceneStore
            self._source_store = ImmutableSceneStore(Path(self.runtime.output_root) / 'accepted-scene')
        return self._source_store

    def _snapshot_source(self, path, *, flatten=True):
        try:
            return self._scene_store().capture(path, flatten=flatten).path
        except Exception:
            self._retain_sources()
            raise

    def _snapshot_inline(self, text):
        try:
            snapshot, rewritten = self._scene_store().capture_inline(text)
        except Exception:
            self._retain_sources()
            raise
        return snapshot.path, rewritten

    def _retain_sources(self):
        if self._source_store is not None:
            keep = set()
            for state in self.states.values():
                state.inline_source_paths = {key: path for key, path in state.inline_source_paths.items()
                    if key[0] != 'value' or key[1:] in state.desired}
                keep.update(state.native_source_paths)
                keep.add(state.native_scene_path)
                # Camera retarget advances recovery identity without replacing
                # population. Its previous full source remains the base for a
                # later conservative parent-pose projection (one extra source).
                keep.add(state.native_population_path)
                keep.update(state.inline_source_paths.values())
                keep.update(state.pending_source_paths.values())
            self._source_store.retain(keep)
            self.startup_diagnostics['scene_storage'] = {
                'retained_sources': len(self._source_store.snapshots),
                'cleanup_errors': list(self._source_store.cleanup_errors)}

    def configure_session(self, request):
        from .render_capabilities import validate_request
        validate_request(request)
        self.request = request

    def accept_render_parameters(self, simulation_id, spec):
        """Commit current camera/output identity after the native packet ACK."""
        self._state(simulation_id).spec = spec

    def _call(self, op, params):
        try:
            result = self.runtime.request(op, params)
        except RuntimeError as exc:
            error = RenderClientError(f'Public SDK {op}: {exc}')
            failure = self._worker_failure()
            if failure is not None:
                # The user-visible text names the native log; every later
                # rewording of this error keeps it.
                error = RenderClientError(worker_failure_logs.with_details(str(error), failure['worker_log']))
                error.worker_failure = failure
            raise error from exc
        if getattr(self.runtime, 'epoch', 0) != self._noticed_epoch:
            self._notice_recoveries()
        return result

    def _runtime_logs(self):
        epoch = int(getattr(self.runtime, 'epoch', 0) or 0)
        root = getattr(self.runtime, 'output_root', None)
        if epoch < 1 or root is None:
            return None
        return worker_failure_logs.log_diagnostics(root, epoch)

    def _new_recoveries(self):
        diagnostics = getattr(self.runtime, 'diagnostics', None)
        records = (diagnostics() or {}).get('recoveries', ()) if callable(diagnostics) else ()
        fresh = [dict(record) for record in records or ()
                 if int(record.get('failed_epoch') or 0) > self._noticed_failed_epoch]
        if fresh:
            self._noticed_failed_epoch = max(int(record['failed_epoch']) for record in fresh)
        self._noticed_epoch = getattr(self.runtime, 'epoch', 0)
        logs = self._runtime_logs()
        if logs is not None and isinstance(self.startup_diagnostics.get('render_worker'), dict):
            self.startup_diagnostics['render_worker']['logs'] = logs
        return fresh

    def _surface_failure(self, worker_log, level, **context):
        failure = worker_failure_logs.remember(worker_failure_logs.collect(worker_log), **context)
        if failure['worker_log'] not in self._mirrored_logs:
            self._mirrored_logs.add(failure['worker_log'])
            worker_failure_logs.mirror_to_console(failure, level)
        return failure

    def _worker_failure(self):
        """The failed worker's logs: the recovery's failed epoch, else the current one."""
        try:
            fresh = self._new_recoveries()
            failed = [record for record in fresh if not record.get('recovered')]
            record = (failed or fresh or [None])[-1]
            if record is not None and record.get('worker_log'):
                return self._surface_failure(record['worker_log'], 'ERROR', cause=record.get('cause', ''),
                                             op=record.get('op', ''), recovered=False)
            logs = self._runtime_logs()
            if logs is None:
                return None
            return self._surface_failure(logs['worker_log'], 'ERROR', recovered=False)
        except Exception:
            # Diagnostics must never replace the original worker error.
            return None

    def _notice_recoveries(self, *, include_explicit=False):
        """Warn once for every worker this client replaced after a failure.

        A recovery that succeeded is otherwise silent; the warning names the
        failed worker's log. An explicit recover() after an uncertain edit is
        not a worker failure unless the caller saw the worker die.
        """
        try:
            records = self._new_recoveries()
        except Exception:
            return
        from . import user_messages
        for record in records:
            if not record.get('recovered') or not record.get('worker_log'):
                continue
            cause = record.get('cause') or 'explicit'
            if cause == 'explicit' and not include_explicit:
                continue
            if cause == 'explicit':
                cause = 'worker_exited'
            failure = self._surface_failure(record['worker_log'], 'WARNING', cause=cause,
                                            op=record.get('op', ''), recovered=True)
            during = f" during {record['op']}" if record.get('op') else ''
            user_messages.report_warning(worker_failure_logs.with_details(
                f"OVRTX renderer restarted after a worker failure ({cause}{during})",
                failure['worker_log']), context=f"worker-recovery:{failure['worker_log']}")

    def _load_params(self, spec):
        # Validate before USD reads, scene replacement or native allocation.
        self._validate_mvp_outputs(tuple(spec.sensor_paths), spec.render_var)
        from .render_capabilities import validate_candidate_settings
        validate_candidate_settings(self.request)
        quality = getattr(self.request, 'rtpt_quality', None)
        settings = rtpt_worker_config.rtpt_carb_overrides(quality)
        dlss_enabled = bool(getattr(self.request, 'dlss_enabled', True))
        dlss_quality = str(getattr(self.request, 'dlss_quality', 'INHERIT'))
        dlss = rtpt_worker_config.dlss_carb_overrides(dlss_enabled, dlss_quality)
        if dlss_quality != 'INHERIT' or not dlss_enabled:
            from .properties import resolve_dlss_quality
            # The matched public SDK uses Token presets on RenderProducts;
            # legacy worker Carb configuration uses the corresponding Int.
            dlss['rtx']['post']['dlss']['execMode'] = resolve_dlss_quality(dlss_quality, dlss_enabled)
        settings.setdefault('rtx', {}).update(dlss.get('rtx', {}))
        path = spec.ovrtx_scene_composition.composed_scene_path
        metadata = getattr(spec, 'stage_metadata', None)
        if metadata is None:
            # Blender already provides USD; NVIDIA SDKs remain outside it.
            from pxr import Usd
            stage = Usd.Stage.Open(str(path))
            if stage is None:
                raise RenderClientError(f'Cannot read accepted USD metadata: {path}')
            metadata = {key: _plain(stage.GetMetadata(key)) for key in (
                'upAxis', 'metersPerUnit', 'timeCodesPerSecond', 'framesPerSecond',
                'startTimeCode', 'endTimeCode', 'defaultPrim', 'renderSettingsPrimPath')
                if stage.HasAuthoredMetadata(key)}
        return {'scene_path': str(path), 'sensor_paths': list(spec.sensor_paths),
                'width': spec.width, 'height': spec.height, 'render_var': spec.render_var,
                'settings': _flatten_settings(settings), 'stage_metadata': metadata}

    def start_session(self, spec, simulation_id=None):
        identity = simulation_id or f'ovrtx-blender-public-{os.getpid()}'
        if self.states:
            raise RenderClientError('A public SDK worker owns one active rendering session')
        params = self._load_params(spec)
        from .scene_population_delta import capture_population_snapshot
        try:
            population_snapshot = capture_population_snapshot(params['scene_path'])
        except Exception:
            population_snapshot = None
        try:
            params['scene_path'] = self._snapshot_source(params['scene_path'])
            result = self._call('load', params)
        except Exception:
            # A failed initial load has no accepted predecessor. Confirm native
            # teardown before releasing its candidate source/asset versions.
            self.runtime.close()
            self._retain_sources()
            raise
        self.states[identity] = _State(spec, simulation_id=identity,
            epoch=int(result.get('_runtime_epoch', 0)),
            stage_metadata=deepcopy(params.get('stage_metadata')),population_snapshot=population_snapshot,
            native_scene_path=params['scene_path'],native_population_path=params['scene_path'],
            native_source_paths={params['scene_path']})
        # The worker's own log files, not the add-on's (stdout) log routing.
        self.startup_diagnostics = {'render_worker': {'status': 'ready', 'backend': 'public_sdk', **result,
                                                      'logs': self._runtime_logs() or {}}}
        return identity

    def can_update_scene(self, current, desired):
        # A color-format change must follow the controller's session re-key.
        # Population replacement alone cannot retain the old output binding.
        if current.render_var != desired.render_var:
            return False
        # The accepted load already captured the native stage metadata. Avoid
        # reopening its entire USD composition for each camera reconfiguration.
        state = next((item for item in self.states.values() if item.spec is current), None)
        current_metadata = None if state is None else state.stage_metadata
        if current_metadata is None:
            current_metadata = self._load_params(current)['stage_metadata']
        return current_metadata == self._load_params(desired)['stage_metadata']

    def update_scene(self, simulation_id, spec):
        state = self._state(simulation_id)
        if state.failure is not None:
            raise ValueUpdateRecoveryRequired(str(state.failure)) from state.failure
        if state.recovery_pending:
            raise RenderClientError('Complete recovered state replay before replacing the scene')
        started = time.perf_counter()
        phase_started = started
        params = self._load_params(spec)
        host_timings = {'load_parameters_ms': (time.perf_counter()-phase_started)*1000}
        from .scene_population_delta import compile_population_delta
        retirement = None
        previous_composition = state.spec.ovrtx_scene_composition
        if previous_composition is not None:
            # Direct camera switches advance current runtime identity without
            # changing bootstrap USD. Retire the override actually present in
            # that USD, which may belong to a different camera than state.spec.
            retired = next((layer for layer in getattr(previous_composition, 'presentation_layers', ())
                            if layer.get('source') == 'viewport_camera_projection'
                            and layer.get('target_path')
                            and layer.get('target_path') != getattr(spec, 'camera_prim_path', '')), None)
            if retired is not None:
                retirement = {'camera_path': retired['target_path'],
                              'layer_path': retired.get('path')}
        delta = compile_population_delta(state.spec.ovrtx_scene_composition.composed_scene_path,
                                         params['scene_path'],accepted_snapshot=state.population_snapshot,
                                         retired_camera_projection=retirement)
        params['scene_path'] = self._snapshot_source(params['scene_path'])
        if delta.path is not None and len(state.native_source_paths) < 33:
            params['population_delta'] = {'path': self._snapshot_source(delta.path, flatten=False),
                'predecessor_path': state.native_scene_path}
            effective_paths = getattr(delta, 'changed_prim_paths', None)
            if effective_paths is not None:
                params['population_changed_prim_paths'] = list(effective_paths)
        try:
            phase_started = time.perf_counter()
            result = self._call('update_scene', params)
            host_timings['native_update_rpc_ms'] = (time.perf_counter()-phase_started)*1000
            host_timings['total_ms'] = (time.perf_counter()-started)*1000
            result['host_update_timings'] = host_timings
            result['delta_preparation_ms'] = delta.preparation_ms
            result['delta_phase_timings'] = delta.phase_timings
            result['delta_changed_prims'] = delta.changed_prims
            result['delta_changed_properties'] = delta.changed_properties
            result['delta_created_prims'] = delta.created_prims
            result['delta_deleted_prims'] = delta.deleted_prims
            result['delta_fallback_reason'] = delta.fallback_reason
            if result.get('scene_updated') is not True:
                if result.get('rollback_succeeded') is not True:
                    state.failure = RenderClientError('Scene generation restoration was not confirmed')
                raise RenderClientError(result.get('error', 'Scene generation was rejected'))
        except RenderClientError as error:
            if getattr(self.runtime, '_uncertain_operation', None):
                state.failure = error
            if state.failure is None:
                self._retain_sources()
            raise
        state.population_revision += 1
        self.startup_diagnostics['scene_update'] = dict(result)
        retained = {}
        if result.get('population_update') == 'scoped_overlay' and state.population_snapshot is not None and delta.candidate_snapshot is not None:
            retained = _unaffected_population_values(state.desired, state.population_snapshot.fields,
                                                     delta.candidate_snapshot.fields,
                                                     effective_changed_paths=getattr(delta, 'changed_prim_paths', None))
            retained = {key: row for key, row in retained.items() if key[1] != 'visibility'}
        state.population_snapshot = delta.candidate_snapshot
        state.spec = spec
        state.native_scene_path = params['scene_path']
        state.native_population_path = params['scene_path']
        if result.get('population_update') == 'scoped_overlay':
            state.native_source_paths.add(params['population_delta']['path'])
        else:
            state.native_source_paths = {params['scene_path']}
        state.inline_source_paths = {key: path for key, path in state.inline_source_paths.items()
            if key[0] == 'value' and key[1:] in retained}
        state.pending_source_paths.clear()
        state.output_size = (spec.width, spec.height)
        state.stage_metadata = deepcopy(params.get('stage_metadata'))
        state.desired = retained
        state.accepted_values.clear()
        state.accepted_values.acknowledge(list(retained.values()), epoch=self.runtime.epoch)
        state.samples = 0
        self._retain_sources()
        return result

    def direct_projection_token(self, simulation_id):
        """Constant-size identity for acknowledged direct projection reuse.

        Camera/value writes retain it. Successful population updates change it;
        load/new client state and worker recovery change state identity/epoch.
        No scene traversal, USD access or diagnostics snapshot is required.
        """
        state = self._state(simulation_id)
        if state.failure is not None or state.recovery_pending:
            return None
        return (id(self), id(state), self.runtime.epoch, state.population_revision)

    def retarget_camera(self, simulation_id, spec, values):
        state = self._state(simulation_id)
        if state.failure is not None:
            raise ValueUpdateRecoveryRequired(str(state.failure)) from state.failure
        from .scene_population_delta import capture_population_snapshot
        try:
            desired_snapshot = capture_population_snapshot(spec.ovrtx_scene_composition.composed_scene_path,
                unchanged_source_snapshot=state.population_snapshot)
        except Exception:
            desired_snapshot = None
        try:
            native_source = self._snapshot_source(spec.ovrtx_scene_composition.composed_scene_path)
        except Exception:
            self._retain_sources()
            raise
        try:
            result = self._call('retarget_camera', {'values': values,
                'source_scene_path':native_source})
        except RenderClientError as error:
            if getattr(self.runtime, '_uncertain_operation', None):
                state.failure = error
            else:
                self._retain_sources()
            raise
        if result.get('camera_retargeted') is not True:
            state.failure = RenderClientError('Native camera retarget was not acknowledged')
            raise state.failure
        state.population_snapshot = desired_snapshot
        state.native_scene_path = native_source
        state.spec = spec
        state.samples = 0
        state.desired.update({(v['prim_path'], v['attribute_name']): dict(v) for v in values})
        state.accepted_values.acknowledge(values, epoch=self.runtime.epoch)
        self._retain_sources()
        return result

    def _state(self, identity):
        try:
            return self.states[identity]
        except KeyError as exc:
            raise RenderClientError('No active public SDK rendering session') from exc

    def render_result(self, simulation_id, *, selected_sensor_paths, render_var, additional_samples):
        state = self._state(simulation_id)
        if state.failure is not None:
            raise ValueUpdateRecoveryRequired(str(state.failure)) from state.failure
        if state.recovery_pending:
            raise RenderClientError('Complete recovered state replay before rendering')
        sensors = tuple(selected_sensor_paths) or state.spec.sensor_paths
        self._validate_mvp_outputs(sensors, render_var)
        if not sensors or any(sensor not in state.spec.sensor_paths for sensor in sensors):
            raise RenderClientError('Selected output is not declared by the current scene')
        started = time.perf_counter_ns()
        with self.runtime.transaction():
            result = self._call('render', {
                'sensor_paths': list(sensors), 'render_var': render_var,
                'samples': max(1, int(additional_samples)),
                'simulation_time_ns': state.time_ns + 16_666_667,
                'presentation_rgba32f': _prepare_hdr_upload(render_var),
                'semantic_outputs': bool(getattr(self.request, 'semantic_output_path', '')),
                'camera_prim_path': str(getattr(self.request, 'camera_prim_path', getattr(state.spec, 'camera_prim_path', ''))),
            })
            return self._consume_render_result(state, result, sensors, render_var,
                                               additional_samples, started)

    def _consume_render_result(self, state, result, sensors, render_var,
                               additional_samples, started, session_samples=None, before_copy=None):
        failed = False
        cancelled = False
        copy_ms = ack_ms = 0.0
        try:
            width, height = result['width'], result['height']
            if type(width) is not int or type(height) is not int or width < 1 or height < 1 or width * height > 67_108_864:
                raise RenderClientError('Invalid frame dimensions')
            if (width, height) != (state.output_size or (state.spec.width, state.spec.height)):
                raise RenderClientError('Frame dimensions differ from the outstanding request')
            if result.get('render_var') != render_var:
                raise RenderClientError('Renderer returned the wrong render variable')
            if result.get('sensor_path') != sensors[0]:
                raise RenderClientError('Renderer returned the wrong render product')
            if before_copy is not None:
                before_copy()
            copy_started = time.perf_counter_ns()
            frame = self._copy_frame(result, width, height, render_var)
            semantic_capture = None
            if bool(getattr(self.request, 'semantic_output_path', '')):
                from .semantic_output import capture_from_reply
                semantic_capture = capture_from_reply(result, result.pop('_semantic_ids', b''), width=width, height=height, sensor_path=sensors[0])
            elif 'semantic' in result:
                raise RenderClientError('Renderer returned unrequested semantic products')
            copy_ms = (time.perf_counter_ns() - copy_started) / 1e6
        except BaseException:
            failed = True
            if self._reader is not None:
                self._reader.close()
                self._reader = None
                self._reader_key = None
            token = result.get('shm', {}).get('copy_token')
            if token and 'raw_frame' in result:
                try:
                    acknowledgement = self._call('cancel_raw_transfer', {'copy_token': token})
                    if acknowledgement.get('cancelled') is not True:
                        raise RenderClientError('Raw transfer cancellation was not acknowledged')
                    cancelled = True
                except Exception as cleanup_error:
                    state.failure = cleanup_error
            raise
        finally:
            token = result.get('shm', {}).get('copy_token')
            if token and not cancelled:
                ack_started = time.perf_counter_ns()
                try:
                    acknowledgement = self._call('release_frame', {'copy_token': token})
                    if acknowledgement.get('released') is not True:
                        raise RenderClientError('Frame release was not acknowledged')
                except Exception as cleanup_error:
                    state.failure = cleanup_error
                    if not failed:
                        # Pixels are not publishable until ACK succeeds. An
                        # uncertain ACK may follow producer death; detach now
                        # so this reader cannot retain that epoch's mapping.
                        reader, self._reader = self._reader, None
                        self._reader_key = None
                        if reader is not None:
                            reader.close()
                        raise
                finally:
                    ack_ms = (time.perf_counter_ns() - ack_started) / 1e6
        epoch = int(result.get('_runtime_epoch', state.epoch))
        if epoch != state.epoch:
            state.samples = 0
            state.session_samples = 0
            state.epoch = epoch
        state.samples = int(result.get('completed_samples', state.samples + max(1, int(additional_samples))))
        if session_samples is None:
            state.session_samples += max(1, int(additional_samples))
        delivered_session_samples = state.session_samples if session_samples is None else session_samples
        delivered_stamp = int(result.get('simulation_time_ns', state.time_ns + 16_666_667))
        state.time_ns = max(state.time_ns, delivered_stamp)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        timings = {'backend': 'public_sdk', 'transport': result.get('transport'),
                   'runtime_epoch': epoch, 'runtime_pid': result.get('_runtime_pid'),
                   'simulation_id': state.simulation_id, 'native_stage_id': result.get('native_stage_id'),
                   'stage_ordinal': result.get('ordinal'),
                   'worker_load_requests': getattr(self.runtime, '_request_counts', {}).get('load', 0),
                   'scene_path': str(state.spec.ovrtx_scene_composition.composed_scene_path),
                   **result.get('timings', {}),
                   **result.get('_acquisition_timings', {}),
                   'acquire_copy_ms': copy_ms,
                   'frame_copy_and_chunk_rpc_ms': copy_ms,
                   'frame_ack_ms': ack_ms,
                   'render_to_owned_frame_ms': elapsed,
                   'native_render_ms_scope': 'render_submission_to_owned_frame_including_wait_copy_ack'}
        self.last_render_timings = {'native_render_ms': elapsed, 'native_timings': timings}
        return RenderResult(width=width, height=height, rgba8=frame[0],
                            completed_samples=state.samples,
                            session_completed_samples=delivered_session_samples,
                            simulation_time_ns=delivered_stamp,
                            render_output_simulation_time_ns=delivered_stamp,
                            render_var=render_var, product_id=sensors[0], linear_rgba16f=frame[1],
                            frame_format=(color_presentation.FRAME_FORMAT_RGBA16F if frame[1] else color_presentation.FRAME_FORMAT_RGBA8),
                            frame_color_mode=(color_presentation.FRAME_COLOR_MODE_SCENE_LINEAR if frame[1] else color_presentation.FRAME_COLOR_MODE_DISPLAY_LDR),
                            native_timings=timings, rgba8_origin=result.get('origin', 'top_left'),
                            presentation_rgba32f=result.get('_presentation_rgba32f', b''), semantic_capture=semantic_capture)

    def _copy_frame(self, result, width, height, render_var):
        from .ovstream_shm import HeldFrame, OvstreamShmReader
        if render_var not in ('HdrColor', 'LdrColor'):
            raise RenderClientError(f'Unsupported color render variable: {render_var}')
        descriptor = HeldFrame(**result['shm'])
        if descriptor.producer_pid != result.get('_runtime_pid'):
            raise RenderClientError('SHM producer differs from the owned worker')
        if '_runtime_session' in result and descriptor.epoch != result['_runtime_session']:
            raise RenderClientError('SHM epoch differs from the authenticated worker session')
        # Reject incompatible layouts and identities before opening SHM or allocating
        # owned frame storage. Runtime identity was authenticated by RuntimeSession.
        from .frame_protocol import validate_held, validate_product
        validate_held(descriptor)
        if 'raw_frame' in result:
            validate_product(result['raw_frame'], product_id=result.get('sensor_path'),
                             aov=render_var, width=width, height=height)
        reader_key = (descriptor.stream_name, descriptor.epoch, descriptor.producer_pid)
        if getattr(self, '_reader_key', None) != reader_key:
            if self._reader is not None:
                self._reader.close()
            self._reader = OvstreamShmReader(stream_name=descriptor.stream_name,
                epoch=descriptor.epoch, producer_pid=descriptor.producer_pid)
            self._reader_key = reader_key
        if 'raw_frame' in result:
            from .ovstream_raw_frame import RawFrameAssembler
            assembler = RawFrameAssembler(descriptor, result['raw_frame'], result['chunk'])
            reply = result
            host_copy_ns = handoff_ns = 0
            chunk_count = 0
            while True:
                copy_started = time.perf_counter_ns()
                complete = assembler.append(self._reader, descriptor, reply['raw_frame'], reply['chunk'])
                host_copy_ns += time.perf_counter_ns() - copy_started
                chunk_count += 1
                if complete:
                    break
                handoff_started = time.perf_counter_ns()
                reply = self._call('next_raw_chunk', {'copy_token': descriptor.copy_token})
                handoff_ns += time.perf_counter_ns() - handoff_started
                result['shm'] = reply['shm']
                descriptor = HeldFrame(**reply['shm'])
                if descriptor.producer_pid != result.get('_runtime_pid'):
                    raise RenderClientError('Chunk producer differs from the owned worker')
            result['_acquisition_timings'] = {
                'raw_chunk_count': chunk_count,
                'raw_payload_bytes': result['raw_frame'].get('byte_count', 0),
                'raw_carrier_bytes': descriptor.width * descriptor.height * 4,
                'host_frame_copy_ms': host_copy_ns / 1e6,
                'chunk_handoff_ms': handoff_ns / 1e6,
            }
            raw = assembler.finish()
            if 'semantic' in result:
                if result['semantic'].get('epoch') != raw.epoch:
                    raise RenderClientError('Semantic metadata belongs to a different SHM epoch')
                result['_semantic_ids'] = raw.payload('semantic_ids', pixel_format='r32u', width=width, height=height)
            if any(p.plane.name == 'presentation_rgba32f' for p in raw.planes):
                if render_var != 'HdrColor':
                    raise RenderClientError('HDR upload plane returned for a non-HDR frame')
                result['_presentation_rgba32f'] = raw.payload('presentation_rgba32f',
                    pixel_format='rgba32f', width=width, height=height)
            linear = (raw.payload('linear_rgba16f', pixel_format='rgba16f', width=width, height=height)
                      if render_var == 'HdrColor' else b'')
            rgba = (raw.payload('rgba8', pixel_format='rgba8', width=width, height=height)
                    if render_var != 'HdrColor' or any(p.plane.name == 'rgba8' for p in raw.planes) else b'')
        else:
            if render_var == 'HdrColor':
                raise RenderClientError('HDR output requires an exact raw SHM payload')
            copied = self._reader.copy_frame(descriptor)
            if (copied.width, copied.height) != (width, height):
                raise RenderClientError('SHM dimensions differ from the requested output')
            rgba, linear = copied.rgba8, b''
        return rgba, linear

    def update_transforms(self, simulation_id, values):
        batch = _transform_values(values)
        rows = []
        for value in batch:
            rows.extend(({'prim_path': value.prim_path, 'attribute_name': 'omni:xform',
                          'value_type': 'Matrix4d', 'value': _plain(value.matrix)},
                         {'prim_path': value.prim_path, 'attribute_name': 'omni:resetXformStack',
                          'value_type': 'Bool', 'value': True}))
        return self._update(simulation_id, rows, len(batch))

    def update_attribute_values(self, simulation_id, values):
        batch = _attribute_values(values)
        rows = [{'prim_path': v.prim_path, 'attribute_name': v.attribute,
                 'value_type': v.value_type, 'value': _plain(v.value)} for v in batch]
        return self._update(simulation_id, rows, len(batch))

    def validate_attribute_values(self, simulation_id, values):
        self._state(simulation_id)
        _attribute_values(values)

    def _update(self, simulation_id, values, count):
        state = self._state(simulation_id)
        if not count:
            return OvrtxValueUpdateResult(0)
        if state.failure is not None:
            raise ValueUpdateRecoveryRequired(str(state.failure)) from state.failure
        sources = {}
        immutable = []
        try:
            for value in values:
                if value['value_type'].lower() in {'asset', 'asset[]'}:
                    path, value = self._scene_store().capture_value(value)
                    sources[('value', value['prim_path'], value['attribute_name'])] = path
                immutable.append(value)
        except Exception:
            self._retain_sources()
            raise
        values = immutable
        state.pending_source_paths.update(sources)
        if state.collecting:
            state.desired.update({(v['prim_path'], v['attribute_name']): deepcopy(v) for v in values})
            self._retain_sources()
            return OvrtxValueUpdateResult(count, state.time_ns + 16_666_667)
        changed = state.accepted_values.changed(values, epoch=self.runtime.epoch)
        counts = {'requested_value_count': len(values), 'sent_value_count': len(changed),
                  'unchanged_value_count': len(values) - len(changed)}
        if not changed:
            state.desired.update({(v['prim_path'], v['attribute_name']): deepcopy(v) for v in values})
            state.inline_source_paths.update(sources)
            for key in sources:
                state.pending_source_paths.pop(key, None)
            self._retain_sources()
            self.last_value_update_timings = counts
            return OvrtxValueUpdateResult(count, state.time_ns + 16_666_667, counts)
        try:
            result = self._call('update', {'values': changed})
            if type(result.get('updated_count')) is not int or result['updated_count'] != len(changed):
                raise RenderClientError('Public SDK did not acknowledge every value update')
        except RenderClientError as exc:
            state.failure = exc
            raise ValueUpdateRecoveryRequired(str(exc)) from exc
        derived = result.get('accepted_derived_values', [])
        state.desired.update({(v['prim_path'], v['attribute_name']): deepcopy(v) for v in values})
        state.desired.update({(row['prim_path'], row['attribute_name']): deepcopy(row) for row in derived})
        state.accepted_values.acknowledge([*changed, *derived], epoch=self.runtime.epoch)
        state.inline_source_paths.update(sources)
        for key in sources:
            state.pending_source_paths.pop(key, None)
        self._retain_sources()
        state.samples = 0
        timings = {**{key: value for key, value in result.items() if key != 'accepted_derived_values'},
                   **counts, 'accepted_derived_value_count': len(derived)}
        self.last_value_update_timings = timings
        return OvrtxValueUpdateResult(count, state.time_ns + 16_666_667, timings)

    def update_material_graphs(self, simulation_id, updates):
        """Acknowledge bounded inline material graphs without a scene token bump."""
        state = self._state(simulation_id)
        if state.failure is not None:
            raise ValueUpdateRecoveryRequired(str(state.failure)) from state.failure
        if state.collecting:
            raise RenderClientError('Material graphs cannot be deferred inside a value collection')
        updates = _plain(updates)
        if not isinstance(updates, list) or not updates:
            raise ValueError('Material graph updates require a nonempty list')
        roots = tuple(update['material_path'] for update in updates)
        sources = {}
        for update in updates:
            path, update['layer_body'] = self._snapshot_inline(update['layer_body'])
            sources[('material', update['material_path'])] = path
        try:
            result = self._call('material_graph_update', {'updates': updates})
            if result.get('material_graph_updated') is not True or result.get('material_count') != len(updates):
                raise RenderClientError('Public SDK did not acknowledge every material graph')
        except RenderClientError as exc:
            state.failure = exc
            raise ValueUpdateRecoveryRequired(str(exc)) from exc
        state.desired = _unaffected_population_values(state.desired, {}, {}, effective_changed_paths=roots)
        state.accepted_values.invalidate_subtrees(roots, epoch=self.runtime.epoch)
        state.samples = 0
        state.inline_source_paths.update(sources)
        self._retain_sources()
        self.last_value_update_timings = dict(result)
        return result

    def update_instance_graphs(self, simulation_id, updates):
        """Apply only nominated current GN owners; preserve the native stage."""
        state = self._state(simulation_id)
        if state.failure is not None:
            raise ValueUpdateRecoveryRequired(str(state.failure)) from state.failure
        if state.collecting:
            raise RenderClientError('Instance graphs cannot be deferred inside a value collection')
        updates = _plain(updates)
        if not isinstance(updates, list) or not updates:
            raise ValueError('Instance graphs require a nonempty owner list')
        roots = tuple(update['projection_path'] for update in updates)
        hidden = {(update['hidden_source_path'], 'visibility') for update in updates}
        sources = {}
        for update in updates:
            path, update['layer_body'] = self._snapshot_inline(update['layer_body'])
            sources[('instance', update['owner_id'])] = path
        try:
            result = self._call('instance_graph_update', {'updates': updates})
            if result.get('instance_graph_updated') is not True or result.get('owner_count') != len(updates):
                raise RenderClientError('Public SDK did not acknowledge every instance graph')
        except RenderClientError as exc:
            state.failure = exc
            raise ValueUpdateRecoveryRequired(str(exc)) from exc
        state.desired = _unaffected_population_values(state.desired, {}, {}, effective_changed_paths=roots)
        for key in hidden: state.desired.pop(key, None)
        state.accepted_values.invalidate_subtrees(roots, epoch=self.runtime.epoch)
        # The old visibility opinion is now owned by the inline source. A
        # generation change invalidates remaining accepted values normally.
        state.samples = 0
        state.inline_source_paths.update(sources)
        self._retain_sources()
        self.last_value_update_timings = dict(result)
        return result

    def resize_output(self, simulation_id, width, height):
        state = self._state(simulation_id)
        self._call('resize', {'width': int(width), 'height': int(height)})
        state.output_size = (int(width), int(height))

    def supports_async_render_sample(self):
        return True

    def _validate_mvp_outputs(self, sensors, render_var):
        if len(sensors) != 1:
            raise RenderClientError('Windows MVP requires exactly one RenderProduct; multiple products are future work')
        if render_var not in ('LdrColor', 'HdrColor'):
            raise RenderClientError('Additional AOVs are future work; select LdrColor or HdrColor')
        if getattr(self.request, 'semantic_output_path', ''):
            raise RenderClientError('Semantic/additional AOV output is future work for the Windows MVP')

    def begin_render_sample(self, simulation_id, *, selected_sensor_paths, render_var):
        state = self._state(simulation_id)
        if state.failure is not None:
            raise ValueUpdateRecoveryRequired(str(state.failure)) from state.failure
        if state.recovery_pending:
            raise RenderClientError('Complete recovered state replay before rendering')
        sensors = tuple(selected_sensor_paths) or state.spec.sensor_paths
        self._validate_mvp_outputs(sensors, render_var)
        if not sensors or any(sensor not in state.spec.sensor_paths for sensor in sensors):
            raise RenderClientError('Selected output is not declared by the current scene')
        started = time.perf_counter_ns()
        stamp = state.time_ns + 16_666_667
        result = self._call('begin_render', {'sensor_paths': list(sensors),
                            'presentation_rgba32f': _prepare_hdr_upload(render_var),
                            'semantic_outputs': bool(getattr(self.request, 'semantic_output_path', '')),
                            'camera_prim_path': str(getattr(self.request, 'camera_prim_path', getattr(state.spec, 'camera_prim_path', ''))),
                            'render_var': render_var, 'simulation_time_ns': stamp})
        epoch = int(result['_runtime_epoch'])
        if epoch != state.epoch:
            state.samples = state.session_samples = 0
            state.epoch = epoch
        state.session_samples += 1
        state.time_ns = stamp
        pending = _PendingPublicRender(self, simulation_id, result['ticket'], sensors,
                                       render_var, epoch, stamp, state.session_samples, started)
        self._pending_native_renders[id(pending)] = pending
        return pending

    def _cancel_native_render(self, pending, *, timeout_ns):
        if self.runtime.epoch != pending.epoch or self.runtime.returncode is not None:
            # RuntimeSession changes epoch only after the predecessor process
            # has been retired. Never replay an old ticket in the new worker.
            self._pending_native_renders.pop(id(pending), None)
            return
        result = self._call('cancel_render', {'ticket': pending.token, 'timeout_ns': timeout_ns})
        if result.get('native_complete') is True:
            self._pending_native_renders.pop(id(pending), None)

    def cancel_pending_renders(self):
        for pending in tuple(self._pending_native_renders.values()):
            pending.cancel()

    @property
    def has_pending_render_cancellations(self):
        return any(pending.cancelled for pending in tuple(self._pending_native_renders.values()))

    def worker_exited(self):
        # Polled by an idle viewport; the next request recovers the worker.
        return self.runtime.worker_exited()

    @property
    def pending_native_render_count(self):
        return len(self._pending_native_renders)

    def drain_render_cancellations(self):
        # One bounded native wait per owner-thread pass, never a busy wait or
        # pixel transfer. All viewports borrowing this client share the fence.
        pending = next((p for p in tuple(self._pending_native_renders.values()) if p.cancelled), None)
        if pending is not None:
            self._cancel_native_render(pending, timeout_ns=8_000_000)
        return not self.has_pending_render_cancellations

    def poll_render_sample(self, pending, *, before_copy=None):
        if pending.client is not self or pending.cancelled or pending.resolved:
            raise RenderClientError('Cancelled, consumed, or foreign renderer ticket')
        state = self._state(pending.simulation_id)
        if state.failure is not None or state.recovery_pending:
            raise RenderClientError('Current scene values require recovery before publication')
        if self.runtime.epoch != pending.epoch or self.runtime.returncode is not None:
            pending.resolved = True
            self._pending_native_renders.pop(id(pending), None)
            return self.render_result(pending.simulation_id, selected_sensor_paths=pending.sensors,
                                      render_var=pending.render_var, additional_samples=1)
        with self.runtime.transaction():
            try:
                result = self._call('poll_render', {'ticket': pending.token})
            except RenderClientError:
                if self.runtime.returncode is None:
                    raise
                # The old ticket is never replayed. Restore the acknowledged
                # scene only after confirming that its producer is dead.
                self.runtime.recover()
                self._notice_recoveries(include_explicit=True)
                pending.resolved = True
                self._pending_native_renders.pop(id(pending), None)
                return self.render_result(pending.simulation_id, selected_sensor_paths=pending.sensors,
                                          render_var=pending.render_var, additional_samples=1)
            if not result.get('ready'):
                return None
            pending.resolved = True
            self._pending_native_renders.pop(id(pending), None)
            return self._consume_render_result(state, result, pending.sensors, pending.render_var,
                                               1, pending.native_started_ns,
                                               session_samples=pending.session_samples,
                                               before_copy=before_copy)

    def needs_initial_value_replay(self, simulation_id):
        state = self._state(simulation_id)
        return state.session_samples == 0 and bool(state.desired)

    def replay_desired_values(self, simulation_id):
        state = self._state(simulation_id)
        return self._update(simulation_id, list(state.desired.values()), len(state.desired))

    def recover_value_updates(self, simulation_id, *, defer_replay=False):
        state = self._state(simulation_id)
        if state.failure is None:
            return False
        self.runtime.recover()
        if state.recovery_baseline is not None:
            state.desired = state.recovery_baseline
            state.recovery_baseline = None
        state.pending_source_paths.clear()
        self._retain_sources()
        state.accepted_values.clear()
        state.failure = None
        state.samples = state.session_samples = 0
        state.recovery_pending = bool(defer_replay)
        if not defer_replay:
            self.replay_desired_values(simulation_id)
        return True

    def begin_recovery_update(self, simulation_id):
        state = self._state(simulation_id)
        state.collecting = state.recovery_pending
        if state.collecting and state.recovery_baseline is None:
            state.recovery_baseline = deepcopy(state.desired)
        return state.collecting

    def end_recovery_update(self, simulation_id, *, commit):
        state = self._state(simulation_id)
        state.collecting = False
        if commit:
            self.replay_desired_values(simulation_id)
            state.recovery_pending = False
            state.recovery_baseline = None
        elif state.recovery_baseline is not None:
            state.desired = state.recovery_baseline
            state.recovery_baseline = None
            state.pending_source_paths.clear()
            self._retain_sources()

    def delete_simulation(self, simulation_id):
        if simulation_id not in self.states:
            return 'not_found'
        self._call('unload', {})
        self.states.pop(simulation_id)
        self._retain_sources()
        return 'stopped'

    @property
    def worker_stop(self):
        """The last worker stop: socket close to process exit, terminate/kill."""
        diagnostics = getattr(self.runtime, 'diagnostics', None)
        return diagnostics().get('last_stop') if diagnostics is not None else None

    def shutdown(self, **kwargs):
        if self._reader is not None:
            self._reader.close()
        self.runtime.close()
        self.states.clear()
        self._retain_sources()
        return 'stopped'


class PublicSdkPhysicsClient:
    supports_unchanged_generation_reuse = True
    supports_verified_simulation_reload = True
    pose_write_space = 'world'
    def __init__(self, config, simulation_id):
        self.config, self.simulation_id = config, simulation_id
        self.runtime = _session('physics')
        self.last_delete_diagnostics = {}
        self.startup_diagnostics = {}

    def start(self):
        self.startup_diagnostics = self.runtime.request('health', {})

    def public_sdk_identity(self):
        return self.runtime.request('health', {})['runtime_identity']

    def create_simulation(self):
        return self.runtime.request('load', {'scene_path': self.config.input_usd_path,
                                            'device': self.config.device,
                                            'body_prims': list(self.config.body_prims)})

    def read_body_states(self, simulation_time_ns):
        result = self.runtime.request('read', {'time_ns': simulation_time_ns,
                                               'body_prims': list(self.config.body_prims)})
        return result['states'], result

    def advance_and_read_body_states(self, start_step_count, steps, timestep_ns):
        from .ovphysx_runtime_client import _validate_advance_inputs
        start_step_count, steps, timestep_ns, count, stamp = _validate_advance_inputs(start_step_count, steps, timestep_ns)
        started = time.perf_counter()
        step_ms = 0.0
        reads_before = None
        native_reads_before = None
        with self.runtime.transaction():
            for step in range(steps):
                result = self.runtime.request('step', {'time_ns': (start_step_count + step + 1) * timestep_ns,
                    'dt_ns': timestep_ns, 'read_states': False})
                step_ms += float(result.get('step_ms', 0.0))
                if step == 0:
                    reads_before = result.get('state_read_calls')
                    native_reads_before = result.get('native_state_read_sessions')
            # Retain the public read boundary used by production qualification
            # and body roster validation; no substep gathers unused states.
            states, result = self.read_body_states(stamp)
        return states, {**result, 'step_count': count, 'simulation_time_ns': stamp,
                        'body_count': len(states), 'step_ms': step_ms,
                        'publication_substeps': steps,
                        'publication_native_state_reads': (result['native_state_read_sessions'] - native_reads_before
                            if native_reads_before is not None and 'native_state_read_sessions' in result else None),
                        'publication_state_reads': (result['state_read_calls'] - reads_before
                            if reads_before is not None and 'state_read_calls' in result else None),
                        'read_ms': result.get('state_read_ms', 0.0),
                        'total_ms': (time.perf_counter() - started) * 1000}

    def write_body_poses(self, poses, *, simulation_time_ns, reset=False):
        return self.runtime.request('write_poses', {'time_ns': simulation_time_ns, 'reset': reset,
            'poses': [{'prim_path': p.prim_path, 'translate': list(p.translate), 'orient': list(p.orient)} for p in poses]})

    def write_body_velocities(self, velocities, *, simulation_time_ns, reset=False):
        return self.runtime.request('write_velocities', {'time_ns': simulation_time_ns, 'reset': reset,
            'velocities': [{'prim_path': v.prim_path, 'linear': list(v.linear), 'angular': list(v.angular)} for v in velocities]})

    def write_body_masses(self, values):
        return self.runtime.request('write_masses', {'values': list(values)})

    @property
    def supports_static_geometry_updates(self):
        if self.config.device != 'cpu':
            return False
        from .physics_capabilities import validate_fixed_runtime
        try:
            validate_fixed_runtime(self.config, self._import_native_client,
                                   self.public_sdk_identity, family='live_geometry')
        except ValueError:
            return False
        return True

    def write_physics_geometry(self, values):
        if not self.supports_static_geometry_updates:
            raise ValueError('Live collider geometry has no measured production admission')
        return self.runtime.request('write_geometry', {'values': list(values)})

    def delete_simulation(self):
        self.runtime.request('unload', {})
        self.last_delete_diagnostics = {'status': 'stopped'}
        return 'stopped'

    def shutdown(self):
        self.runtime.close()
        return 'stopped'

    def _import_native_client(self):
        raise RuntimeError('Public SDK admission requires its own measured capability evidence; SIL binary identity is not applicable')
