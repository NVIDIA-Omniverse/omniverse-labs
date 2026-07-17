# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cooperative node-local lease for exclusive OVRTX GPU sessions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

from .session_lifecycle import pid_running


LOCK_DIR_ENV = "OV_BLENDER_EXAMPLE_OVRTX_GPU_LOCK_DIR"
LEASE_ID_ENV = "OV_BLENDER_EXAMPLE_OVRTX_GPU_LEASE_ID"
LEASE_OWNER_PID_ENV = "OV_BLENDER_EXAMPLE_OVRTX_GPU_LEASE_OWNER_PID"
LEASE_TOKEN_ENV = "OV_BLENDER_EXAMPLE_OVRTX_GPU_LEASE_TOKEN"
LEASE_WAIT_ENV = "OV_BLENDER_EXAMPLE_OVRTX_GPU_LEASE_WAIT_S"
DEFAULT_WAIT_S = 0.0


class OvrtxGpuLeaseError(RuntimeError):
    """Base error for OVRTX GPU lease failures."""


class OvrtxGpuLeaseBusy(OvrtxGpuLeaseError):
    """Raised when another process already owns the effective OVRTX GPU."""

    def __init__(self, gpu_id: str, owner: Mapping[str, Any] | None) -> None:
        self.gpu_id = gpu_id
        self.owner = dict(owner or {})
        owner_text = _owner_text(self.owner)
        super().__init__(
            f"OVRTX GPU lease is busy for {gpu_id}"
            + (f" ({owner_text})" if owner_text else "")
        )


@dataclass
class OvrtxGpuLease:
    gpu_id: str
    lock_path: Path
    metadata_path: Path
    metadata: Mapping[str, Any]
    _file: Any
    _token: str = ""
    _owned: bool = True
    _closed: bool = False

    def diagnostics(self) -> dict[str, Any]:
        return {
            "status": (
                "released" if self._closed else "held" if self._owned else "inherited"
            ),
            "gpu_id": self.gpu_id,
            "lock_path": str(self.lock_path),
            "metadata_path": str(self.metadata_path),
            "metadata": dict(self.metadata),
        }

    def child_environment(self) -> dict[str, str]:
        if not self._owned or not self._token:
            return {}
        return {
            LEASE_ID_ENV: self.gpu_id,
            LEASE_OWNER_PID_ENV: str(os.getpid()),
            LEASE_TOKEN_ENV: self._token,
        }

    def close(self) -> None:
        if self._closed:
            return
        if not self._owned:
            self._closed = True
            return
        try:
            try:
                self.metadata_path.unlink()
            except OSError:
                pass
            if not getattr(self._file, "closed", False):
                _unlock_file(self._file)
        finally:
            if not getattr(self._file, "closed", False):
                self._file.close()
            self._closed = True

    def __enter__(self) -> "OvrtxGpuLease":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def acquire(
    *,
    metadata: Mapping[str, Any] | None = None,
    timeout_s: float | None = None,
    env: Mapping[str, str] | None = None,
) -> OvrtxGpuLease:
    environment = os.environ if env is None else env
    gpu_id = resolve_gpu_id(environment)
    lock_dir = _lock_dir(environment)
    lock_path = lock_dir / f"{_safe_name(gpu_id)}.lock"
    metadata_path = lock_path.with_suffix(lock_path.suffix + ".json")
    inherited = _inherited_lease(environment, gpu_id, lock_path, metadata_path)
    if inherited is not None:
        return inherited
    stream: Any | None = None
    locked = False
    try:
        lock_dir.mkdir(parents=True, exist_ok=True)
        wait_s = _wait_timeout(environment, timeout_s)
        deadline = time.monotonic() + wait_s
        stream = lock_path.open("a+b")
        _ensure_windows_lock_byte(stream)
        while True:
            try:
                _try_lock_file(stream)
                locked = True
                break
            except BlockingIOError:
                if wait_s <= 0 or time.monotonic() >= deadline:
                    owner = _read_metadata(metadata_path)
                    stream.close()
                    stream = None
                    raise OvrtxGpuLeaseBusy(gpu_id, owner)
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

        token = secrets.token_urlsafe(32)
        record = _metadata_record(gpu_id, metadata)
        record["child_token_sha256"] = _token_sha256(token)
        metadata_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OvrtxGpuLeaseBusy:
        raise
    except Exception as exc:
        if stream is not None and not getattr(stream, "closed", False):
            if locked:
                try:
                    _unlock_file(stream)
                except Exception:
                    pass
            stream.close()
        raise OvrtxGpuLeaseError(
            f"Could not acquire OVRTX GPU lease for {gpu_id}: {exc}"
        ) from exc
    return OvrtxGpuLease(
        gpu_id=gpu_id,
        lock_path=lock_path,
        metadata_path=metadata_path,
        metadata=record,
        _file=stream,
        _token=token,
    )


def resolve_gpu_id(env: Mapping[str, str] | None = None) -> str:
    environment = os.environ if env is None else env
    explicit = _first_token(environment.get(LEASE_ID_ENV, ""))
    if explicit:
        return explicit
    active_cuda_gpus = environment.get("OVRTX_ACTIVE_CUDA_GPUS")
    if active_cuda_gpus is not None:
        value = active_cuda_gpus.strip()
        if not value or value.lower() == "all" or "," in value:
            return "node"
        if value.startswith(("GPU-", "MIG-")):
            return value
        uuid = _nvidia_smi_gpu_uuids().get(value)
        return uuid or f"gpu-{value}"
    for name in ("CUDA_VISIBLE_DEVICES",):
        value = _first_token(environment.get(name, ""))
        if not value or value.lower() in {"all", "none", "void"}:
            continue
        if value.startswith(("GPU-", "MIG-")):
            return value
        uuid = _nvidia_smi_gpu_uuids().get(value)
        return uuid or f"gpu-{value}"
    uuids = _nvidia_smi_gpu_uuids()
    return uuids.get("0") or "node"


def busy_diagnostics(error: OvrtxGpuLeaseBusy) -> dict[str, Any]:
    return {
        "status": "busy",
        "gpu_id": error.gpu_id,
        "owner": dict(error.owner),
        "error": str(error),
    }


def probe(*, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Report lease availability without changing its owner metadata."""

    environment = os.environ if env is None else env
    gpu_id = resolve_gpu_id(environment)
    lock_path = _lock_dir(environment) / f"{_safe_name(gpu_id)}.lock"
    if not lock_path.exists():
        return {"status": "available", "gpu_id": gpu_id}
    try:
        stream = lock_path.open("r+b")
    except FileNotFoundError:
        return {"status": "available", "gpu_id": gpu_id}
    try:
        try:
            _try_lock_file(stream)
        except BlockingIOError:
            error = OvrtxGpuLeaseBusy(
                gpu_id,
                _read_metadata(lock_path.with_suffix(lock_path.suffix + ".json")),
            )
            return busy_diagnostics(error)
        else:
            _unlock_file(stream)
            return {"status": "available", "gpu_id": gpu_id}
    except OSError as exc:
        return {"status": "error", "gpu_id": gpu_id, "error": str(exc)}
    finally:
        stream.close()


def _inherited_lease(
    env: Mapping[str, str],
    gpu_id: str,
    lock_path: Path,
    metadata_path: Path,
) -> OvrtxGpuLease | None:
    token = env.get(LEASE_TOKEN_ENV, "")
    owner_pid = env.get(LEASE_OWNER_PID_ENV, "")
    if not token or not owner_pid:
        return None
    owner = _read_metadata(metadata_path)
    try:
        pid = int(owner_pid)
    except ValueError:
        return None
    if (
        owner.get("pid") != pid
        or owner.get("gpu_id") != gpu_id
        or owner.get("child_token_sha256") != _token_sha256(token)
        or not pid_running(pid)
    ):
        return None
    return OvrtxGpuLease(
        gpu_id=gpu_id,
        lock_path=lock_path,
        metadata_path=metadata_path,
        metadata=owner,
        _file=None,
        _owned=False,
    )


def _lock_dir(env: Mapping[str, str]) -> Path:
    return Path(env.get(LOCK_DIR_ENV) or Path(tempfile.gettempdir()) / "ovrtx-gpu-locks")


def _wait_timeout(env: Mapping[str, str], timeout_s: float | None) -> float:
    if timeout_s is not None:
        return max(0.0, float(timeout_s))
    value = env.get(LEASE_WAIT_ENV, "")
    if not value:
        return DEFAULT_WAIT_S
    try:
        return max(0.0, float(value))
    except ValueError:
        return DEFAULT_WAIT_S


def _metadata_record(gpu_id: str, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "gpu_id": gpu_id,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "argv": list(sys.argv),
        "started_at_ns": time.time_ns(),
        **dict(metadata or {}),
    }


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _owner_text(owner: Mapping[str, Any]) -> str:
    parts = []
    if owner.get("pid") is not None:
        parts.append(f"pid={owner['pid']}")
    if owner.get("entrypoint"):
        parts.append(f"entrypoint={owner['entrypoint']}")
    if owner.get("cwd"):
        parts.append(f"cwd={owner['cwd']}")
    return ", ".join(parts)


def _first_token(value: str) -> str:
    return next((part.strip() for part in value.split(",") if part.strip()), "")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value or "node")


def _nvidia_smi_gpu_uuids() -> dict[str, str]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode != 0:
        return {}
    result: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        index, separator, uuid = line.partition(",")
        if separator:
            result[index.strip()] = uuid.strip()
    return result


def _ensure_windows_lock_byte(stream: Any) -> None:
    if os.name != "nt":
        return
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"\0")
        stream.flush()
    stream.seek(0)


def _try_lock_file(stream: Any) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise BlockingIOError(str(error)) from error
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(stream: Any) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


__all__ = [
    "LOCK_DIR_ENV",
    "LEASE_ID_ENV",
    "LEASE_OWNER_PID_ENV",
    "LEASE_TOKEN_ENV",
    "LEASE_WAIT_ENV",
    "OvrtxGpuLease",
    "OvrtxGpuLeaseBusy",
    "OvrtxGpuLeaseError",
    "acquire",
    "busy_diagnostics",
    "probe",
    "resolve_gpu_id",
]
