# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned public-SDK subprocess, serialized RPC, and bounded renderer recovery.

Only acknowledged absolute scene state is retained. Physics advances and writes
whose acknowledgement was lost are never automatically retried. The caller must
hold ``transaction()`` across rendering and copying its shared-memory frame.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import secrets
import os
from pathlib import Path
import socket
import struct
import subprocess
import threading
import time
from typing import Any, Iterator, Mapping

MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_WINDOWS = os.name == "nt"
_BOOTSTRAP_MAGIC = b"OVBC1"
_BOOTSTRAP_MAX_BYTES = 4096
_BOOTSTRAP_ACK = b"OVBC1\n"
SAFE_RECOVERY_OPERATIONS = frozenset({"render", "read", "read_frame", "begin_render"})
# These mutations may follow recovery only when the old worker was already
# confirmed dead before sending this request. They are never retried after send.
UNSENT_RENDERER_EDITS = frozenset({"update", "resize", "update_scene", "retarget_camera", "material_graph_update", "instance_graph_update"})
PHYSICS_MUTATIONS = frozenset({"load", "step", "write_poses", "write_velocities", "write_masses", "write_geometry", "unload"})
# Error text in an ``ok:false`` reply that means the worker's device context is
# gone. Such a worker is poisoned: it is stopped and replaced, never reused.
FATAL_DEVICE_ERROR_PATTERNS = ("cudaError", "CUDA error 700", "illegal address", "DEVICE_LOST",
                               "invalid CUDA interop stream", "VK_ERROR_DEVICE_LOST")
# Seconds to wait before the first and second automatic respawn of a budget.
RESTART_BACKOFF_S = (1.0, 3.0)
# The restart budget is restored after this many successful renders in a row,
# or after this long without a worker failure.
BUDGET_RESET_RENDERS = 20
BUDGET_RESET_S = 600.0


def worker_environment(base: Mapping[str, str], overrides: Mapping[str, str]) -> dict[str, str]:
    """The worker's environment: ``base`` without Python/OCIO redirects, plus ``overrides``.

    CUDA's default device order (fastest first) can differ from nvidia-smi's.
    PCI bus order, unless the user chose an order, makes the CUDA index in
    ``OVRTX_ACTIVE_CUDA_GPUS`` name the GPU nvidia-smi shows at that index.
    """
    environment = {key: value for key, value in base.items()
                   if key.upper() not in {'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'OCIO'}}
    environment.update(overrides)
    if not environment.get('CUDA_DEVICE_ORDER'):
        environment['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    return environment


def is_fatal_device_error(text: object) -> bool:
    """Whether worker error text names a fatal device failure (case-insensitive)."""
    folded = str(text).casefold()
    return any(pattern.casefold() in folded for pattern in FATAL_DEVICE_ERROR_PATTERNS)


class RuntimeSessionError(RuntimeError):
    """The external runtime did not acknowledge a request."""


class RuntimeWorkerError(RuntimeSessionError):
    """The worker explicitly rejected a request (the transport is still usable)."""


class RuntimeFatalDeviceError(RuntimeWorkerError):
    """The worker reported a fatal device error; that worker is poisoned."""


class RuntimeTransportError(RuntimeSessionError):
    """The request outcome may be unknown because its transport failed."""


class RuntimeWorkerLost(RuntimeTransportError):
    """The worker closed or reset its control connection; that worker is gone."""


def is_worker_failure(error: BaseException | None) -> bool:
    """Whether ``error`` or an error it wraps is a failed transport or a fatal device error.

    Clients wrap session errors (``RenderClientError`` raised from
    ``RuntimeWorkerLost``), so the causes are searched. A request the worker
    rejected, or a recovery budget that was already exhausted, is not one.
    """
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, (RuntimeTransportError, RuntimeFatalDeviceError)):
            return True
        error = error.__cause__ or error.__context__
    return False


@dataclass(frozen=True)
class WorkerIdentity:
    """Authenticated producer identity; launcher is ownership metadata only."""
    pid: int
    launcher_pid: int
    epoch: int
    session: str
    capabilities: tuple[tuple[str, int], ...]


class RuntimeSession:
    """One independently owned renderer or physics process; no native imports.

    ``epoch`` increases for every spawn. A worker receives a fresh output folder
    per epoch; frame descriptors from a preceding process must not be reused.
    Renderer recovery replays load, coalesced acknowledged updates, then resize.
    The independent physics owner is never touched by renderer recovery.
    """

    # Seconds a worker gets to exit after its socket closes before terminate.
    exit_wait_s = 30.0
    restart_backoff_s = RESTART_BACKOFF_S
    budget_reset_renders = BUDGET_RESET_RENDERS
    budget_reset_s = BUDGET_RESET_S

    def __init__(
        self,
        python: str | Path,
        worker_script: str | Path,
        kind: str,
        output_root: str | Path,
        timeout_s: float = 60,
        *,
        max_restarts: int = 2,
        environment: dict[str, str] | None = None,
    ) -> None:
        if kind not in {"renderer", "physics"}:
            raise ValueError("kind must be renderer or physics")
        if timeout_s <= 0 or max_restarts < 0:
            raise ValueError("timeout must be positive and restart budget nonnegative")
        # The installed runtime lives in an ASCII folder (native_paths); an explicit
        # external interpreter outside one cannot be remapped.
        from .native_paths import require_ascii_native
        require_ascii_native(python, 'runtime interpreter')
        self.python = str(python)
        self.worker_script = str(Path(worker_script).resolve())
        self.kind = kind
        self.output_root = Path(output_root).resolve()
        self._environment = dict(environment or {})
        self._cleanup_errors: list[str] = []
        self._request_counts: dict[str, int] = {}
        self.timeout_s = float(timeout_s)
        self.max_restarts = int(max_restarts)
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._owner = None
        self._worker_pid = None
        self._worker_returncode = None
        self._session_nonce = ""
        self._capabilities = {}
        self._socket: socket.socket | None = None
        self._buffer = bytearray()
        self._epoch = 0
        self._next_id = 0
        self._restarts = 0
        self._clean_renders = 0
        self._last_failure_at: float | None = None
        self._recoveries: list[dict[str, Any]] = []
        self._first_load_pending = True
        self._poisoned = False
        self._stopped = True
        self._budget_resets = 0
        self._closed = False
        self._transport_failed = False
        self._uncertain_operation = ""
        self._load: dict[str, Any] | None = None
        self._values: dict[tuple[str, str], dict[str, Any]] = {}
        self._resize: dict[str, Any] | None = None
        self._material_graphs: dict[str, dict[str, Any]] = {}
        self._instance_graphs: dict[int, dict[str, Any]] = {}
        self._stops: list[dict[str, Any]] = []

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def pid(self) -> int | None:
        return self._worker_pid

    @property
    def launcher_pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def identity(self) -> WorkerIdentity | None:
        if self.pid is None:
            return None
        return WorkerIdentity(self.pid, self.launcher_pid, self.epoch,
                              self._session_nonce, tuple(sorted(self._capabilities.items())))

    @property
    def returncode(self) -> int | None:
        if self._owner is not None:
            return self._owner.poll_worker()
        if self.pid is not None:
            return self._worker_returncode
        return self._process.poll() if self._process is not None else None

    def worker_exited(self) -> bool:
        """Whether this session's renderer worker has exited (no request is sent).

        An idle viewport polls this to notice a worker that died between
        requests. Never blocks: a session busy with a request reports False.
        """
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if self._closed or self._process is None:
                return False
            return self.returncode is not None
        except OSError:
            return False
        finally:
            self._lock.release()

    @contextmanager
    def transaction(self) -> Iterator["RuntimeSession"]:
        """Serialize request and shared-frame copy against restart or overwrite."""
        with self._lock:
            yield self

    def request(self, op: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        # Snapshot before the operation so mutable Blender-side values cannot
        # change an acknowledged replay record after the fact.
        payload = self._snapshot({} if params is None else params)
        self._validate_retained_operation(op, payload)
        with self._lock:
            self._require_open()
            if self._uncertain_operation:
                raise RuntimeSessionError(
                    f"Outcome of {self._uncertain_operation!r} is unknown; "
                    "explicit renderer recover() or close is required"
                )
            if self._process is None:
                self._spawn()
            elif self.returncode is not None:
                if self.kind == "renderer" and (
                    op in SAFE_RECOVERY_OPERATIONS
                    or (self._load is not None and op in UNSENT_RENDERER_EDITS)
                ):
                    self._recover_locked(cause="worker_exited", op=op)
                else:
                    raise RuntimeTransportError(f"{self.kind} worker exited: {self._process.returncode}")
            elif self._transport_failed:
                raise RuntimeTransportError("Worker transport failed while process is alive; explicit recovery required")
            # The session's first load has no predecessor state, so a worker
            # death during it is recovered by a clean respawn and the same load.
            first_load = self.kind == "renderer" and op == "load" and self._first_load_pending
            if op == "load":
                self._first_load_pending = False
            try:
                result = self._exchange(op, payload)
            except RuntimeWorkerError as error:
                fatal = self.kind == "renderer" and isinstance(error, RuntimeFatalDeviceError)
                if fatal:
                    self._poison()
                if not fatal or not (op in SAFE_RECOVERY_OPERATIONS or first_load):
                    # A native exception can follow a partial body mutation or
                    # simulation step. A structured error does not prove rollback.
                    if (self.kind == "physics" and op in PHYSICS_MUTATIONS) or op in {'retarget_camera', 'material_graph_update', 'instance_graph_update'}:
                        self._uncertain_operation = op
                    raise
                result = self._recover_and_retry(op, payload, "fatal_device_error", error)
            except RuntimeTransportError as error:
                self._transport_failed = True
                self._note_failure()
                lost = self.kind == "renderer" and isinstance(error, RuntimeWorkerLost)
                if lost:
                    # EOF or reset is terminal for this worker even while its
                    # process is still exiting; stop it instead of waiting.
                    self._stop_process()
                if self.kind != "renderer" or not (op in SAFE_RECOVERY_OPERATIONS or first_load):
                    if op not in SAFE_RECOVERY_OPERATIONS:
                        self._uncertain_operation = op
                    raise
                if not lost:
                    # A timeout or invalid reply does not prove the worker is
                    # gone; a hung worker is never restarted automatically.
                    self._confirm_exit()
                    if self._process is None or self.returncode is None:
                        if first_load:
                            self._uncertain_operation = op
                        raise
                result = self._recover_and_retry(op, payload, "worker_lost" if lost else "worker_exited", error)
            self._retain(op, payload, result=result)
            if op in ('update', 'retarget_camera'):
                self._retain_derived(result)
            if op == "render" or (op == "poll_render" and result.get("ready")):
                self._note_render()
            return {**result, "_runtime_epoch": self._epoch, "_runtime_pid": self.pid,
                    "_runtime_session": self._session_nonce}

    def recover(self) -> None:
        """Explicitly discard an uncertain renderer outcome and restore acked state.

        This may terminate this owner's hung renderer. It cannot recover physics:
        its dynamic state cannot be reconstructed from initial conditions alone.
        """
        with self._lock:
            self._require_open()
            self._recover_locked()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._stop_process()
            # A failed terminate/reap must remain retryable. Marking closed
            # first would make a later shutdown falsely confirm completion.
            self._closed = True
            self._load = None
            self._values.clear()
            self._material_graphs.clear()
            self._instance_graphs.clear()
            self._resize = None

    def __enter__(self) -> "RuntimeSession":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "kind": self.kind,
                "epoch": self._epoch,
                "pid": self.pid,
                "launcher_pid": self.launcher_pid,
                "capabilities": dict(self._capabilities),
                "returncode": self.returncode,
                "restarts": self._restarts,
                "closed": self._closed,
                "transport_failed": self._transport_failed,
                "uncertain_operation": self._uncertain_operation,
                "retained_values": len(self._values),
                "retained_material_graphs": len(self._material_graphs),
                "retained_instance_graphs": len(self._instance_graphs),
                "has_scene": self._load is not None,
                "cleanup_errors": list(self._cleanup_errors),
                "request_counts": dict(self._request_counts),
                # Socket close to process exit for each stopped worker process.
                "last_stop": dict(self._stops[-1]) if self._stops else None,
                "stops": [dict(stop) for stop in self._stops],
                "poisoned": self._poisoned,
                "clean_renders": self._clean_renders,
                "budget_resets": self._budget_resets,
                # Every recovery, newest last: cause, failed worker and its log.
                "recoveries": [dict(record) for record in self._recoveries],
                "last_recovery": dict(self._recoveries[-1]) if self._recoveries else None,
            }

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeSessionError("Runtime session is closed")

    @staticmethod
    def _snapshot(params: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(params, Mapping):
            raise ValueError("Runtime parameters must be a mapping")
        raw = json.dumps(dict(params), allow_nan=False, separators=(",", ":")).encode()
        if len(raw) > MAX_MESSAGE_BYTES - 1024:
            raise ValueError("Runtime request exceeds 16 MiB limit")
        return json.loads(raw)

    @staticmethod
    def _validate_retained_operation(op: str, params: dict[str, Any]) -> None:
        if not isinstance(op, str) or not op or len(op) > 128:
            raise ValueError("Runtime operation must be a short nonempty string")
        if op in ("update", "retarget_camera"):
            values = params.get("values")
            if not isinstance(values, list):
                raise ValueError("update requires values list")
            for value in values:
                if not isinstance(value, dict) or not all(
                    isinstance(value.get(key), str) and value[key]
                    for key in ("prim_path", "attribute_name")
                ):
                    raise ValueError("update values need prim_path and attribute_name")
        if op == 'material_graph_update':
            updates = params.get('updates')
            if not isinstance(updates, list) or not updates or len(updates) > 256:
                raise ValueError('Material graph operation requires bounded updates')
            roots = set()
            for update in updates:
                if not isinstance(update, dict) or set(update) != {'material_path', 'layer_body'}:
                    raise ValueError('Material graph requires root and layer body')
                root, text = update['material_path'], update['layer_body']
                if (not isinstance(root, str) or not root.startswith('/') or root == '/'
                        or root in roots or any(p in {'', '.', '..'} for p in root[1:].split('/'))
                        or not isinstance(text, str) or not text.startswith('#usda 1.0\n')):
                    raise ValueError('Invalid material graph retained state')
                roots.add(root)
        if op == 'instance_graph_update':
            updates = params.get('updates')
            if not isinstance(updates, list) or not updates:
                raise ValueError('Instance graph operation requires updates')
            owners = set()
            for update in updates:
                if not isinstance(update, dict) or set(update) != {'owner_id', 'projection_path', 'hidden_source_path', 'layer_body'}:
                    raise ValueError('Instance graph requires owner identity, paths and layer body')
                owner = update['owner_id']
                if (type(owner) is not int or owner <= 0 or owner in owners
                        or update['projection_path'] != '/World/_DirectGeometry/Object_' + str(owner)
                        or not isinstance(update['hidden_source_path'], str)
                        or not update['hidden_source_path'].startswith('/')
                        or not isinstance(update['layer_body'], str)
                        or not update['layer_body'].startswith('#usda 1.0\n')):
                    raise ValueError('Invalid instance graph retained state')
                owners.add(owner)

    def _spawn(self) -> None:
        from .owned_process_tree import OwnedProcessTree
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._stopped = False
        self._poisoned = False
        self._epoch += 1
        self._session_nonce = secrets.token_hex(32)
        self._worker_pid = None
        self._worker_returncode = None
        self._capabilities = {}
        epoch_root = self.output_root / f"epoch-{self._epoch}"
        epoch_root.mkdir(parents=True, exist_ok=True)
        timeout = min(self.timeout_s, 15.0)
        deadline = time.monotonic() + timeout
        parent, child = socket.socketpair(socket.AF_INET) if _WINDOWS else socket.socketpair()
        listener = None
        try:
            self._owner = OwnedProcessTree()
            command = [self.python, '-I', '-B', self.worker_script, "--kind", self.kind,
                       "--output-root", str(epoch_root), "--bootstrap-timeout", str(timeout)]
            environment = worker_environment(os.environ, self._environment)
            # Blender selects its own display OCIO config at startup. The SDK
            # renderer consumes authored color settings and must not inherit
            # that display transform. SDK LDR pixels already include color
            # correction; Blender applies its color transform only to HDR.
            options: dict[str, Any] = {"close_fds": True, "env": environment,
                                       "stdin": subprocess.PIPE, "bufsize": 0}
            from .frame_protocol import offer
            challenge = {"protocol": 2, "session": self._session_nonce, "epoch": self._epoch, "application": offer()}
            if _WINDOWS:
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.bind(('127.0.0.1', 0)); listener.listen(1)
                challenge['port'] = listener.getsockname()[1]
                command.append("--socket-share-stdin")
                options['creationflags'] = subprocess.CREATE_NO_WINDOW
            else:
                command.extend(["--fd", str(child.fileno())])
                options.update(pass_fds=(child.fileno(),), start_new_session=True)
            with (epoch_root / "worker.log").open("ab", buffering=0) as log:
                self._process = self._owner.spawn(command, stdout=log, stderr=subprocess.STDOUT, **options)
            self._socket = parent
            self._write_bootstrap(challenge, deadline)
            if listener is not None:
                listener.settimeout(max(.001, deadline - time.monotonic()))
                bootstrap, address = listener.accept()
                try:
                    if address[0] != '127.0.0.1':
                        raise ValueError('Worker bootstrap peer is not local')
                    self._accept_hello(self._bootstrap_read(bootstrap, deadline))
                    # WSADuplicateSocket must target the authenticated Python
                    # worker, never the venv redirector stored in Popen.pid.
                    shared = child.share(self.pid)
                    if not 0 < len(shared) <= _BOOTSTRAP_MAX_BYTES:
                        raise ValueError('Invalid shared socket bootstrap size')
                    bootstrap.sendall(struct.pack('!I', len(shared)) + shared)
                finally:
                    bootstrap.close()
                ack = self._bootstrap_read(parent, deadline)
                if ack != {"ready": True, "session": self._session_nonce, "epoch": self._epoch, "pid": self.pid}:
                    raise ValueError('Invalid worker socket acknowledgment')
            else:
                self._accept_hello(self._bootstrap_read(parent, deadline))
                parent.sendall(b'{"ready":true}\n')
            parent.settimeout(None)
        except BaseException as error:
            parent.close()
            self._transport_failed = True
            self._stop_process()
            raise RuntimeTransportError(f"Worker identity bootstrap failed: {error}") from error
        finally:
            child.close()
            if listener is not None:
                listener.close()
        self._buffer.clear()
        self._transport_failed = False

    def _write_bootstrap(self, challenge, deadline):
        # The bounded secret is only on the owned stdin pipe, never argv, env,
        # disk or logs. A writer thread bounds even a non-reading child.
        done = threading.Event()
        errors = []
        process = self._process
        payload = json.dumps(challenge).encode() + b'\n'
        def write():
            try:
                remaining = memoryview(payload)
                while remaining:
                    count = process.stdin.write(remaining)
                    if not count:
                        raise ConnectionError('Worker bootstrap pipe closed')
                    remaining = remaining[count:]
            except BaseException as error:
                errors.append(error)
            finally:
                try:
                    process.stdin.close()
                except BaseException as error:
                    errors.append(error)
                finally:
                    done.set()
        threading.Thread(target=write, name='ov-public-bootstrap', daemon=True).start()
        if not done.wait(max(0, deadline - time.monotonic())):
            raise TimeoutError('Worker bootstrap pipe timed out')
        if errors:
            raise RuntimeTransportError('Worker bootstrap pipe failed') from errors[0]

    @staticmethod
    def _bootstrap_read(connection, deadline):
        data = bytearray()
        # No read-ahead: a duplicate hello must remain on the wire and fail
        # the first response/ACK validation rather than being discarded.
        while not data.endswith(b'\n'):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Worker identity bootstrap timed out')
            connection.settimeout(remaining)
            part = connection.recv(1)
            if not part:
                raise ConnectionError('Worker identity bootstrap closed')
            data.extend(part)
            if len(data) > _BOOTSTRAP_MAX_BYTES:
                raise ValueError('Worker identity bootstrap exceeds limit')
        return json.loads(data)

    def _accept_hello(self, hello):
        if (not isinstance(hello, dict) or type(hello.get('protocol')) is not int
                or hello.get('protocol') != 2):
            raise ValueError('PROTOCOL_MISMATCH: required bootstrap version is 2')
        if (not isinstance(hello, dict) or set(hello) != {'protocol', 'session', 'epoch', 'pid', 'capabilities', 'application'}
                or hello['protocol'] != 2 or type(hello['protocol']) is not int
                or hello['session'] != self._session_nonce
                or type(hello['epoch']) is not int or hello['epoch'] != self._epoch
                or not self._owner.contains(hello['pid'])):
            raise ValueError('Unauthenticated worker identity/session')
        from .frame_protocol import negotiate, CAPABILITIES
        negotiate(hello['application'])
        capabilities = hello['capabilities']
        if (not isinstance(capabilities, dict) or any(capabilities.get(key) != value for key, value in CAPABILITIES.items())
                or capabilities.get('owned_frames') != 1
                or any(type(value) is not int or value < 1 for value in capabilities.values())):
            raise ValueError('Incompatible worker control capabilities')
        if self._worker_pid is not None:
            raise ValueError('Duplicate worker hello')
        self._owner.authenticate(hello['pid'])
        self._worker_pid = hello['pid']
        self._capabilities = capabilities

    def _exchange(self, op: str, params: dict[str, Any]) -> dict[str, Any]:
        connection = self._socket
        if connection is None:
            raise RuntimeTransportError("No runtime control connection")
        self._next_id += 1
        request_id = self._next_id
        raw = json.dumps(
            {"id": request_id, "op": op, "params": params, "epoch": self._epoch, "session": self._session_nonce},
            allow_nan=False, separators=(",", ":"),
        ).encode() + b"\n"
        if len(raw) > MAX_MESSAGE_BYTES:
            raise ValueError("Runtime request exceeds 16 MiB limit")
        deadline = time.monotonic() + self.timeout_s
        try:
            connection.settimeout(self.timeout_s)
            connection.sendall(raw)
            self._request_counts[op] = self._request_counts.get(op, 0) + 1
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Runtime response deadline expired")
                connection.settimeout(remaining)
                chunk = connection.recv(min(65536, MAX_MESSAGE_BYTES + 1 - len(self._buffer)))
                if not chunk:
                    raise ConnectionError("Runtime control connection closed")
                self._buffer.extend(chunk)
                if len(self._buffer) > MAX_MESSAGE_BYTES:
                    raise ValueError("Runtime response exceeds 16 MiB limit")
            line, _, remainder = self._buffer.partition(b"\n")
            self._buffer = bytearray(remainder)
            response = json.loads(line)
            if not isinstance(response, dict) or response.get("id") != request_id:
                raise ValueError("Runtime response ID does not match request")
            if response.get("pid") != self.pid or response.get("session") != self._session_nonce:
                raise ValueError("Runtime response belongs to wrong worker/session")
            if response.get("epoch") != self._epoch:
                raise ValueError("Runtime response belongs to stale worker epoch")
            if not isinstance(response.get("ok"), bool):
                raise ValueError("Runtime response lacks boolean ok")
            if not response["ok"]:
                detail = response.get('error', 'worker rejected operation')
                if is_fatal_device_error(detail):
                    raise RuntimeFatalDeviceError(f"{self.kind} {op}: {detail}")
                raise RuntimeWorkerError(f"{self.kind} {op}: {detail}")
            result = response.get("result", {})
            if not isinstance(result, dict):
                raise ValueError("Runtime result must be an object")
            if op in ("update", "retarget_camera"):
                acknowledged = result.get("updated_count")
                if type(acknowledged) is not int or acknowledged != len(params["values"]):
                    # A valid response can still report a partially applied
                    # batch. Keep the prior acknowledged replay ledger intact
                    # and block publication until explicit recovery discards
                    # this uncertain worker state. Apply the same rule during
                    # replay so recovery cannot certify a partial restoration.
                    self._uncertain_operation = op
                    raise RuntimeSessionError(
                        "Runtime did not acknowledge every update; explicit renderer recover() required"
                    )
                if op == 'retarget_camera' and result.get('camera_retargeted') is not True:
                    self._uncertain_operation = op
                    raise RuntimeSessionError('Native camera retarget was not acknowledged; explicit renderer recover() required')
                self._validate_derived_receipt(op, params, result)
            if op == 'material_graph_update' and (
                    result.get('material_graph_updated') is not True
                    or result.get('material_count') != len(params['updates'])):
                self._uncertain_operation = op
                raise RuntimeSessionError('Material graph update was not acknowledged; explicit recover required')
            if op == 'instance_graph_update' and (
                    result.get('instance_graph_updated') is not True
                    or result.get('owner_count') != len(params['updates'])):
                self._uncertain_operation = op
                raise RuntimeSessionError('Instance graph update was not acknowledged; explicit recover required')
            if op == "update_scene" and result.get("scene_updated") is not True:
                if result.get("rollback_succeeded") is not True:
                    self._uncertain_operation = op
                    detail = result.get("error") or "Worker did not confirm predecessor restoration"
                    raise RuntimeSessionError(
                        f"Scene generation outcome is uncertain; explicit recovery required: {detail}"
                    )
                raise RuntimeWorkerError(result.get("error", "Scene generation rejected; predecessor restored"))
            return result
        except RuntimeWorkerError:
            raise
        except ConnectionError as exc:
            # EOF or reset: the worker closed its end, whether or not its
            # process has exited yet. A timeout is not a ConnectionError.
            raise RuntimeWorkerLost(f"{self.kind} {op}: {exc}") from exc
        except (OSError, ValueError) as exc:
            raise RuntimeTransportError(f"{self.kind} {op}: {exc}") from exc

    def _retain(self, op: str, params: dict[str, Any], *, result=None) -> None:
        if op in ("load", "update_scene"):
            self._load = params
            if op == 'update_scene' and (result or {}).get('population_update') == 'scoped_overlay':
                roots = params.get('population_changed_prim_paths')
                self._values = {key: row for key, row in self._values.items()
                    if roots is not None and key[1] != 'visibility'
                    and not any(key[0] == root or key[0].startswith(root.rstrip('/') + '/') for root in roots)}
            else:
                self._values.clear()
            self._material_graphs.clear()
            self._instance_graphs.clear()
            self._resize = None
        elif op in ("update", "retarget_camera"):
            if op == "retarget_camera" and self._load is not None and params.get("source_scene_path"):
                self._load = {**self._load,"scene_path":params["source_scene_path"]}
            for value in params["values"]:
                self._values[(value["prim_path"], value["attribute_name"])] = value
        elif op == 'material_graph_update':
            roots = tuple(update['material_path'] for update in params['updates'])
            for update in params['updates']:
                self._material_graphs[update['material_path']] = update
            self._values = {key: row for key, row in self._values.items()
                            if not any(key[0] == root or key[0].startswith(root + '/') for root in roots)}
        elif op == 'instance_graph_update':
            roots = tuple(update['projection_path'] for update in params['updates'])
            hidden = {(update['hidden_source_path'], 'visibility') for update in params['updates']}
            for update in params['updates']:
                self._instance_graphs[update['owner_id']] = update
            self._values = {key: row for key, row in self._values.items()
                if key not in hidden and not any(key[0] == root or key[0].startswith(root + '/') for root in roots)}
        elif op == "resize":
            self._resize = params
        elif op == "unload":
            self._load = None
            self._values.clear()
            self._material_graphs.clear()
            self._instance_graphs.clear()
            self._resize = None

    def _validate_derived_receipt(self, op, params, result):
        """Certify only inherited poses of previously explicitly retained keys."""
        rows = result.get('accepted_derived_values', [])
        parents = {row['prim_path'] for row in params['values']
                   if row['attribute_name'] == 'omni:xform'}
        requested = {(row['prim_path'], row['attribute_name']) for row in params['values']}
        seen = set()
        try:
            if not isinstance(rows, list):
                raise ValueError('derived values must be a list')
            for row in rows:
                if not isinstance(row, dict) or set(row) != {'prim_path', 'attribute_name', 'value_type', 'value'}:
                    raise ValueError('derived value has an invalid shape')
                path, name = row['prim_path'], row['attribute_name']
                if not isinstance(path, str) or not isinstance(name, str):
                    raise ValueError('derived value target must be strings')
                key = path, name
                previous = self._values.get(key)
                if key in seen or key in requested or previous is None or not any(path.startswith(parent + '/') for parent in parents):
                    raise ValueError('derived value is not a unique retained descendant')
                seen.add(key)
                if name == 'omni:xform' and row['value_type'] == 'Matrix4d':
                    matrix = row['value']
                    if (not isinstance(matrix, list) or len(matrix) != 4
                            or any(not isinstance(line, list) or len(line) != 4 for line in matrix)
                            or any(type(v) not in (float, int) or not math.isfinite(v) for line in matrix for v in line)):
                        raise ValueError('derived transform must be a finite 4x4 matrix')
                elif name == 'omni:resetXformStack' and row['value_type'] == 'Bool' and row['value'] is True:
                    pass
                else:
                    raise ValueError('unsupported derived value type')
                if previous.get('value_type') != row['value_type']:
                    raise ValueError('derived type differs from acknowledged target')
        except (ValueError, TypeError, KeyError) as exc:
            self._uncertain_operation = op
            raise RuntimeSessionError(f'Untrusted derived update receipt; explicit renderer recover() required: {exc}') from exc

    def _retain_derived(self, result):
        for row in result.get('accepted_derived_values', ()):
            self._values[(row['prim_path'], row['attribute_name'])] = self._snapshot(row)

    def _replay_values(self, batch):
        params = {'values': batch}
        result = self._exchange('update', params)
        self._retain('update', params)
        self._retain_derived(result)

    def _confirm_exit(self) -> None:
        deadline = time.monotonic() + .1
        while self._process is not None and self.returncode is None and time.monotonic() < deadline:
            time.sleep(.005)

    def _poison(self) -> None:
        # The worker still answers, but its device context is gone. Never reuse
        # it, and skip the graceful exit wait: native teardown may hang on it.
        self._poisoned = True
        self._note_failure()
        self._stop_process()

    def _note_failure(self) -> None:
        now = time.monotonic()
        if self._last_failure_at is not None and now - self._last_failure_at >= self.budget_reset_s:
            self._reset_budget()
        self._last_failure_at = now
        self._clean_renders = 0

    def _note_render(self) -> None:
        self._clean_renders += 1
        if self._restarts and (
                self._clean_renders >= self.budget_reset_renders
                or (self._last_failure_at is not None
                    and time.monotonic() - self._last_failure_at >= self.budget_reset_s)):
            self._reset_budget()

    def _reset_budget(self) -> None:
        if self._restarts:
            self._budget_resets += 1
        self._restarts = 0
        self._clean_renders = 0

    def _recover_and_retry(self, op: str, payload: dict[str, Any], cause: str,
                           error: RuntimeSessionError) -> dict[str, Any]:
        """Replace a dead or poisoned renderer, then send the request once more."""
        try:
            self._recover_locked(cause=cause, op=op, error=error)
        except RuntimeSessionError as failure:
            # Keep the original classification (transport or fatal device) so
            # callers can tell a worker failure from a rejected request.
            raise type(error)(f"{error}; recovery failed: {failure}") from failure
        try:
            return self._exchange(op, payload)
        except RuntimeFatalDeviceError:
            self._poison()
            raise
        except RuntimeTransportError as again:
            self._transport_failed = True
            self._note_failure()
            if isinstance(again, RuntimeWorkerLost):
                self._stop_process()
            if op not in SAFE_RECOVERY_OPERATIONS:
                self._uncertain_operation = op
            raise

    def _recover_locked(self, *, cause: str = "explicit", op: str = "",
                        error: BaseException | None = None) -> None:
        if self.kind != "renderer":
            raise RuntimeSessionError("Physics state cannot be automatically replayed; explicit reset required")
        self._note_failure()
        if self._restarts >= self.max_restarts:
            raise RuntimeSessionError(f"Renderer recovery budget exhausted ({self.max_restarts})")
        self._restarts += 1
        failed_epoch, failed_pid = self._epoch, self.pid
        self._stop_process()
        # Automatic respawns back off; an explicit recover() is the caller's choice.
        backoff = 0.0
        if cause != "explicit" and self.restart_backoff_s:
            backoff = float(self.restart_backoff_s[min(self._restarts, len(self.restart_backoff_s)) - 1])
        worker_log = self.output_root / f"epoch-{failed_epoch}" / "worker.log"
        record = {"cause": cause, "op": op, "error": "" if error is None else str(error),
                  "failed_epoch": failed_epoch, "failed_pid": failed_pid, "worker_log": str(worker_log),
                  "restart": self._restarts, "max_restarts": self.max_restarts,
                  "backoff_s": backoff, "recovered": False, "epoch": None}
        self._recoveries = [*self._recoveries[-7:], record]
        if cause != "explicit":
            print(f"OVRTX: {self.kind} worker recovery {self._restarts}/{self.max_restarts} "
                  f"after {cause} during {op!r} (epoch {failed_epoch}, pid {failed_pid})"
                  f"{': ' + record['error'] if record['error'] else ''}; "
                  f"respawning in {backoff:g} s. Worker log: {worker_log}", flush=True)
        if backoff > 0:
            time.sleep(backoff)
        self._spawn()
        record["epoch"] = self._epoch
        try:
            if self._load is not None:
                self._exchange("load", self._load)
                for graph in self._instance_graphs.values():
                    self._exchange('instance_graph_update', {'updates': [graph]})
                # Compile current graph resources before replaying typed shader
                # constants. One current graph per root, never operation history.
                for graph in self._material_graphs.values():
                    self._exchange('material_graph_update', {'updates': [graph]})
                if self._values:
                    # Split replay by request size: an editing session can
                    # accumulate more values than fit in one control message.
                    batch: list[dict[str, Any]] = []
                    size = 0
                    for value in tuple(self._values.values()):
                        item_size = len(json.dumps(value, separators=(",", ":")).encode()) + 1
                        if batch and size + item_size > MAX_MESSAGE_BYTES - 2048:
                            self._replay_values(batch)
                            batch, size = [], 0
                        batch.append(value)
                        size += item_size
                    if batch:
                        self._replay_values(batch)
                if self._resize is not None:
                    self._exchange("resize", self._resize)
        except BaseException:
            self._transport_failed = True
            self._stop_process()
            raise
        record["recovered"] = True
        self._uncertain_operation = ""

    def _stop_process(self) -> None:
        if self._stopped:
            # Already stopped, e.g. after a lost connection; keep its exit record.
            self._buffer.clear()
            return
        connection, self._socket = self._socket, None
        closed = time.monotonic()
        if connection is not None:
            connection.close()
        process = self._process
        stop = None
        if process is not None:
            stop = {"epoch": self._epoch, "pid": self.pid, "socket_closed": connection is not None,
                    "waited": False, "close_to_exit_s": None, "terminated": False, "killed": False}
        if process is not None and process.poll() is None and not self._transport_failed and not self._poisoned:
            stop["waited"] = True
            try:
                # Native renderer destruction and driver cache flush can take
                # more than five seconds on Linux. Give
                # EOF-driven teardown a bounded chance to release SDK resources.
                process.wait(timeout=self.exit_wait_s)
            except subprocess.TimeoutExpired:
                stop["terminated"] = True
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    stop["killed"] = True
                    process.kill()
                    process.wait(timeout=2)
            finally:
                stop["close_to_exit_s"] = round(time.monotonic() - closed, 3)
        if stop is not None:
            stop["returncode"] = process.poll()
            self._stops = [*self._stops[-7:], stop]
            try:
                # Beside worker.log, so the time survives the Blender process.
                (self.output_root / f"epoch-{self._epoch}" / "process-exit.json").write_text(
                    json.dumps(stop) + "\n", encoding="utf-8")
            except OSError:
                pass
        if self._owner is not None:
            self._owner.stop()
            self._worker_returncode = self._owner.poll_worker()
            self._owner = None
        if process is not None and process.poll() is not None and self.pid is not None:
            from .ovstream_shm_cleanup import cleanup_registry
            try:
                cleanup_registry(self.output_root / f"epoch-{self._epoch}" / "ovstream-resources.json",
                                 expected_pid=self.pid, expected_epoch=self._session_nonce)
            except (OSError, ValueError, RuntimeError) as exc:
                # Keep teardown reliable, but expose refused cleanup to callers.
                self._cleanup_errors.append(f"epoch-{self._epoch}: {exc}")
        self._buffer.clear()
        self._stopped = True
