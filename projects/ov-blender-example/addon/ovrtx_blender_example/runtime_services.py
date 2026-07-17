# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add-on-lifetime ownership of the installed OVRTX and OVPhysX services."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from . import bundled_runtime, native_client_support
from .ovphysx_runtime_client import DEFAULT_OVPHYSX_NATIVE_CLIENT_MODULE
from .ovrtx_gpu_lease import OvrtxGpuLease, acquire as acquire_gpu_lease


HEALTH_TIMEOUT_ENV = "OV_BLENDER_EXAMPLE_HEALTH_TIMEOUT_S"
HEALTH_INTERVAL_SECONDS = 1.0
_RETRYABLE_GRPC = {"UNAVAILABLE", "DEADLINE_EXCEEDED"}
_RETRYABLE_HEALTH = {"UNKNOWN", "NOT_SERVING", "SERVICE_UNKNOWN"}


class RuntimeServiceError(RuntimeError):
    """A required installed service could not become healthy."""


@dataclass(frozen=True)
class ServiceHealth:
    name: str
    endpoint: str
    status: str
    attempts: int


def health_timeout_seconds() -> int:
    return max(1, int(os.environ.get(HEALTH_TIMEOUT_ENV, "600")))


def wait_for_serving(
    name: str,
    endpoint: str,
    check: Callable[[float], str],
    *,
    process_alive: Callable[[], bool] | None = None,
    cancelled: Callable[[], bool] | None = None,
    deadline: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ServiceHealth:
    """Require aggregate gRPC health with the accepted bounded retry policy."""

    deadline = monotonic() + health_timeout_seconds() if deadline is None else deadline
    attempts = 0
    while True:
        if cancelled is not None and cancelled():
            raise RuntimeServiceError(f"{name} startup was cancelled")
        if process_alive is not None and not process_alive():
            raise RuntimeServiceError(f"{name} exited before reporting SERVING")
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            raise RuntimeServiceError(
                f"{name} did not report SERVING before the health deadline"
            )
        attempts += 1
        try:
            status = str(check(min(HEALTH_INTERVAL_SECONDS, remaining))).upper()
        except Exception as exc:
            if cancelled is not None and cancelled():
                raise RuntimeServiceError(f"{name} startup was cancelled") from exc
            diagnostics = native_client_support.exception_protocol_diagnostics(exc) or {}
            grpc_status = str(diagnostics.get("grpc_status", "")).upper()
            if grpc_status not in _RETRYABLE_GRPC:
                raise RuntimeServiceError(f"{name} health failed: {exc}") from exc
            status = grpc_status
        if cancelled is not None and cancelled():
            raise RuntimeServiceError(f"{name} startup was cancelled")
        if status == "SERVING":
            if deadline - monotonic() <= 0.0:
                raise RuntimeServiceError(
                    f"{name} did not report SERVING before the health deadline"
                )
            return ServiceHealth(name, endpoint, status, attempts)
        if status not in _RETRYABLE_HEALTH | _RETRYABLE_GRPC:
            raise RuntimeServiceError(
                f"{name} health reported unsupported status {status!r}"
            )
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            raise RuntimeServiceError(
                f"{name} did not report SERVING before the health deadline"
            )
        sleep(min(HEALTH_INTERVAL_SECONDS, remaining))


class RuntimeServiceOwner:
    """Keep required services healthy independently of renderer sessions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._root: Path | None = None
        self._clients: list[Any] = []
        self._modules: list[Any] = []
        self._managed_modules: set[int] = set()
        self._cancelled = threading.Event()
        self._gpu_lease: OvrtxGpuLease | None = None
        self._health: dict[str, ServiceHealth] = {}
        self._ovrtx_module: Any | None = None
        self._ovrtx_client: Any | None = None
        self._error = ""
        self._status = "stopped"

    def start(
        self,
        root: Path,
        *,
        force: bool = False,
        on_serving: Callable[[ServiceHealth], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, ServiceHealth]:
        root = Path(root).expanduser().resolve()
        with self._lock:
            if not force and self._root == root and len(self._health) == 2:
                return dict(self._health)
            self._close_locked()
            self._cancelled.clear()
            self._raise_if_cancelled(cancelled)
            self._status = "starting"
            try:
                self._start_locked(root, on_serving=on_serving, cancelled=cancelled)
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"
                self._close_locked(clear_error=False)
                self._status = "failed"
                if isinstance(exc, RuntimeServiceError):
                    raise
                raise RuntimeServiceError(str(exc)) from exc
            self._root = root
            self._error = ""
            self._status = "ready"
            return dict(self._health)

    def close(self) -> None:
        self.cancel()
        with self._lock:
            self._close_locked()

    def restart_ovrtx(self, root: Path) -> ServiceHealth:
        """Synchronously replace only the owner-managed OVRTX service."""

        root = Path(root).expanduser().resolve()
        if self._status == "starting":
            # The caller confirmed discarding aggregate startup. Always follow
            # that snapshot with an aggregate restart: startup may become ready
            # while cancel() is closing both services.
            self.cancel()
            return self.start(root, force=True)["ovrtx"]
        with self._lock:
            if "ovphysx" in self._health:
                self._close_ovrtx_locked()
                self._cancelled.clear()
                self._status = "starting"
                try:
                    self._start_ovrtx_locked(
                        bundled_runtime.defaults(root=root),
                        deadline=time.monotonic() + health_timeout_seconds(),
                        on_serving=None,
                        cancelled=None,
                    )
                except Exception as exc:
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._close_ovrtx_locked()
                    self._status = "failed"
                    raise RuntimeServiceError(str(exc)) from exc
                self._root = root
                self._error = ""
                self._status = "ready"
                return self._health["ovrtx"]
        return self.start(root, force=True)["ovrtx"]

    def cancel(self) -> None:
        self._cancelled.set()
        for client in tuple(self._clients):
            try:
                client.close()
            except Exception:
                pass
        for module in tuple(self._modules):
            if id(module) not in self._managed_modules:
                continue
            try:
                module.shutdown()
            except Exception:
                pass

    def owns_module(self, module: Any) -> bool:
        # "Owns" must mean "the owner will terminate this worker on close"
        # (i.e. it is in ``_managed_modules``), not merely "the owner retains a
        # reference to the module". ``_close_locked``/``cancel`` shut down only
        # managed modules, and the per-session clients skip their own native
        # shutdown when ``owns_module`` is True — so if this reported True for an
        # unmanaged module, neither path would terminate the worker (the
        # orphaned ovphysx-bridge-server). The two decisions must use one set.
        if not self._lock.acquire(blocking=False):
            return False
        try:
            return id(module) in self._managed_modules
        finally:
            self._lock.release()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "status": self._status,
            "root": str(self._root or ""),
            "health": {
                name: {
                    "endpoint": value.endpoint,
                    "status": value.status,
                    "attempts": value.attempts,
                }
                for name, value in tuple(self._health.items())
            },
            "error": self._error,
        }

    def _start_locked(
        self,
        root: Path,
        *,
        on_serving: Callable[[ServiceHealth], None] | None,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        deadline = time.monotonic() + health_timeout_seconds()
        self._raise_if_cancelled(cancelled)
        defaults = bundled_runtime.defaults(root=root)
        if not defaults.worker_command:
            raise RuntimeServiceError("installed OVRTX service command is unavailable")
        if not defaults.native_client_path:
            raise RuntimeServiceError("installed runtime native clients are unavailable")
        if defaults.native_client_path not in sys.path:
            sys.path.insert(0, defaults.native_client_path)

        self._gpu_lease = acquire_gpu_lease(
            metadata={"entrypoint": "RuntimeServiceOwner.start", "runtime_root": str(root)}
        )
        self._start_ovrtx_locked(
            defaults,
            deadline=deadline,
            on_serving=on_serving,
            cancelled=cancelled,
        )

        self._raise_if_cancelled(cancelled)
        ovphysx = importlib.import_module(DEFAULT_OVPHYSX_NATIVE_CLIENT_MODULE)
        ovphysx_address = os.environ.get(
            "OV_BLENDER_EXAMPLE_OVPHYSX_ADDRESS",
            bundled_runtime.DEFAULT_OVPHYSX_ADDRESS,
        )
        ovphysx_command = os.environ.get(
            "OV_BLENDER_EXAMPLE_OVPHYSX_WORKER_COMMAND",
            defaults.ovphysx_worker_command,
        ).strip()
        if ovphysx_command:
            from .ovphysx_runtime_client import _ovphysx_worker_environment
            from .ovrtx_runtime_client import _restore_environment

            worker_environment = _ovphysx_worker_environment(
                SimpleNamespace(ovphysx_address=ovphysx_address)
            )
            previous = _apply_environment(worker_environment)
            self._modules.append(ovphysx)
            self._managed_modules.add(id(ovphysx))
            try:
                ovphysx_result = ovphysx.start_worker(
                    {
                        "worker_command": ovphysx_command,
                        "address": ovphysx_address,
                        "log_path": "",
                        "ready_timeout_ms": _remaining_milliseconds(deadline),
                    }
                )
            finally:
                _restore_environment(previous)
            self._raise_if_cancelled(cancelled)
            if not isinstance(ovphysx_result, Mapping) or not bool(
                ovphysx_result.get("worker_process_alive")
            ):
                self._managed_modules.discard(id(ovphysx))
        else:
            ovphysx_result = ovphysx.connect({"address": ovphysx_address})
            self._raise_if_cancelled(cancelled)
        ovphysx_endpoint = _endpoint(
            ovphysx_result, ovphysx_address
        )
        ovphysx_client = ovphysx.Client(ovphysx_endpoint)
        self._retain(
            ovphysx,
            ovphysx_client,
            ovphysx_result,
            managed=bool(ovphysx_command),
        )
        self._health["ovphysx"] = wait_for_serving(
            "OVPhysX",
            ovphysx_endpoint,
            lambda timeout: ovphysx_client.health("", timeout),
            process_alive=_process_alive(ovphysx, ovphysx_result),
            cancelled=lambda: self._is_cancelled(cancelled),
            deadline=deadline,
        )
        self._adopt_serving_worker(ovphysx, bool(ovphysx_command))
        if on_serving is not None:
            on_serving(self._health["ovphysx"])

    def _start_ovrtx_locked(
        self,
        defaults: Any,
        *,
        deadline: float,
        on_serving: Callable[[ServiceHealth], None] | None,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        from .ovrtx_runtime_client import (
            _restore_environment,
            _sanitize_worker_environment,
            apply_worker_runtime_environment,
        )

        ovrtx = importlib.import_module(bundled_runtime.DEFAULT_OVRTX_NATIVE_CLIENT_MODULE)
        self._ovrtx_module = ovrtx
        self._modules.append(ovrtx)
        self._managed_modules.add(id(ovrtx))
        previous = _sanitize_worker_environment()
        previous.update(
            apply_worker_runtime_environment(os.environ, defaults.worker_command)
        )
        try:
            result = ovrtx.start_worker(
                {
                    "worker_command": defaults.worker_command,
                    "wait_seconds": _remaining_seconds(deadline),
                }
            )
        finally:
            _restore_environment(previous)
        self._raise_if_cancelled(cancelled)
        if not isinstance(result, Mapping) or not bool(result.get("worker_process_alive")):
            self._managed_modules.discard(id(ovrtx))
        endpoint = _endpoint(result, "127.0.0.1:50051")
        client = ovrtx.Client(endpoint)
        self._ovrtx_client = client
        self._retain(ovrtx, client, result)
        self._health["ovrtx"] = wait_for_serving(
            "OVRTX",
            endpoint,
            _module_serving_probe(ovrtx),
            process_alive=_process_alive(ovrtx, result),
            cancelled=lambda: self._is_cancelled(cancelled),
            deadline=deadline,
        )
        self._adopt_serving_worker(ovrtx, bool(defaults.worker_command))
        if on_serving is not None:
            on_serving(self._health["ovrtx"])

    def _is_cancelled(self, cancelled: Callable[[], bool] | None) -> bool:
        return self._cancelled.is_set() or (cancelled is not None and cancelled())

    def _raise_if_cancelled(self, cancelled: Callable[[], bool] | None) -> None:
        if self._is_cancelled(cancelled):
            raise RuntimeServiceError("runtime service startup was cancelled")

    def _retain(
        self,
        module: Any,
        client: Any,
        result: Any,
        *,
        managed: bool = True,
    ) -> None:
        if not any(module is owned for owned in self._modules):
            self._modules.append(module)
        self._clients.append(client)
        if managed and isinstance(result, Mapping) and bool(result.get("worker_process_alive")):
            self._managed_modules.add(id(module))

    def _adopt_serving_worker(self, module: Any, launched: bool) -> None:
        """Manage a worker the owner launched and confirmed serving.

        The native ``start_worker`` result's ``worker_process_alive`` flag is
        unreliable: ovphysx reports it False for a worker that is running and
        healthy, which drops the module from ``_managed_modules`` even though
        the owner spawned a killable process. A launched worker that has passed
        ``wait_for_serving`` is unambiguously ours to terminate, so adopt it for
        shutdown here rather than trusting that flag. ``launched`` is False on
        the connect path (an externally-owned server we must not kill).
        """

        if launched:
            self._managed_modules.add(id(module))

    def _close_locked(self, *, clear_error: bool = True) -> None:
        for client in reversed(self._clients):
            try:
                client.close()
            except Exception:
                pass
        for module in reversed(self._modules):
            if id(module) not in self._managed_modules:
                continue
            try:
                module.shutdown()
            except Exception:
                pass
        if self._gpu_lease is not None:
            self._gpu_lease.close()
        self._root = None
        self._clients.clear()
        self._modules.clear()
        self._managed_modules.clear()
        self._gpu_lease = None
        self._health.clear()
        self._ovrtx_module = None
        self._ovrtx_client = None
        if clear_error:
            self._error = ""
            self._status = "stopped"

    def _close_ovrtx_locked(self) -> None:
        module = self._ovrtx_module
        client = self._ovrtx_client
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
            self._clients = [owned for owned in self._clients if owned is not client]
        if module is not None:
            if id(module) in self._managed_modules:
                try:
                    module.shutdown()
                except Exception:
                    pass
            self._modules = [owned for owned in self._modules if owned is not module]
            self._managed_modules.discard(id(module))
        self._ovrtx_module = None
        self._ovrtx_client = None
        self._health.pop("ovrtx", None)


def _endpoint(result: Any, fallback: str) -> str:
    if not isinstance(result, Mapping):
        raise RuntimeServiceError("service startup returned a malformed response")
    return str(result.get("address") or result.get("endpoint") or fallback)


def _remaining_seconds(deadline: float) -> int:
    remaining = math.ceil(deadline - time.monotonic())
    if remaining <= 0:
        raise RuntimeServiceError("runtime services did not report SERVING before the health deadline")
    return remaining


def _remaining_milliseconds(deadline: float) -> int:
    remaining = int((deadline - time.monotonic()) * 1000)
    if remaining <= 0:
        raise RuntimeServiceError("runtime services did not report SERVING before the health deadline")
    return remaining


def _module_serving_probe(module: Any) -> Callable[[float], str]:
    """Adapt a native module's ``check_health()`` to ``wait_for_serving``.

    The OVRTX (ovsensors) client reports gRPC serving state through the module
    level ``check_health()`` mapping; its ``Client`` has no ``health`` method,
    unlike the ovphysx client. Map the boolean ``serving`` flag onto the gRPC
    health status strings the serving wait understands.
    """

    def probe(_timeout: float) -> str:
        state = module.check_health()
        serving = isinstance(state, Mapping) and bool(state.get("serving"))
        return "SERVING" if serving else "NOT_SERVING"

    return probe


def _process_alive(module: Any, result: Any) -> Callable[[], bool] | None:
    if not isinstance(result, Mapping) or not bool(result.get("worker_process_alive")):
        return None
    health = getattr(module, "check_health", None)
    if not callable(health):
        return None

    def alive() -> bool:
        state = health()
        return isinstance(state, Mapping) and bool(state.get("worker_process_alive"))

    return alive


def _apply_environment(values: Mapping[str, str]) -> dict[str, str | None]:
    previous = {
        name: os.environ.get(name)
        for name, value in values.items()
        if os.environ.get(name) != value
    }
    os.environ.update(values)
    return previous


owner = RuntimeServiceOwner()


__all__ = [
    "RuntimeServiceError",
    "RuntimeServiceOwner",
    "ServiceHealth",
    "health_timeout_seconds",
    "owner",
    "wait_for_serving",
]
