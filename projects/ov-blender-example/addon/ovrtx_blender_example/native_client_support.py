# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for add-on-owned native runtime client adapters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Mapping, TypeVar


TError = TypeVar("TError", bound=Exception)


def exception_protocol_diagnostics(exc: BaseException) -> dict[str, Any] | None:
    diagnostics = getattr(exc, "protocol_diagnostics", None)
    if isinstance(diagnostics, Mapping):
        return dict(diagnostics)
    diagnostics = getattr(exc, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        return dict(diagnostics)
    return None


def rpc_status_exception(
    name: str,
    exc: BaseException,
    *,
    client_label: str,
    error_type: type[TError],
) -> TError:
    diagnostics = exception_protocol_diagnostics(exc) or {"error": f"{type(exc).__name__}: {exc}"}
    method = str(diagnostics.get("protocol_method") or name)
    status = str(diagnostics.get("grpc_status") or "UNKNOWN")
    message = f"{client_label} {method} failed with gRPC status {status}"
    detail = str(diagnostics.get("grpc_message") or "").strip()
    if detail:
        # The worker's own reason ("A different simulation is already
        # loaded", "Stage load failed for ...") is the actionable part of
        # the report; a bare status name is not diagnosable from the UI.
        message = f"{message}: {detail}"
    error = error_type(message)
    error.protocol_diagnostics = diagnostics  # type: ignore[attr-defined]
    return error


def rpc_status_error_type(
    native_module: Any,
    *,
    client_label: str,
    error_type: type[TError],
) -> type[BaseException] | None:
    value = getattr(native_module, "RpcStatusError", None)
    if value is None:
        return None
    if isinstance(value, type) and issubclass(value, BaseException):
        return value
    raise error_type(f"{client_label} RpcStatusError is not an exception class")


def call_native_rpc(
    name: str,
    function: Callable[[Any], Any],
    argument: Any,
    *,
    rpc_status_error: type[BaseException] | None,
    client_label: str,
    error_type: type[TError],
) -> Mapping[str, Any]:
    try:
        response = function(argument)
    except Exception as exc:
        if rpc_status_error is not None and isinstance(exc, rpc_status_error):
            raise rpc_status_exception(
                name,
                exc,
                client_label=client_label,
                error_type=error_type,
            ) from exc
        raise
    return response if isinstance(response, Mapping) else {}


def require_callable(
    native_module: Any,
    name: str,
    *,
    client_label: str,
    error_type: type[TError],
) -> Callable[..., Any]:
    value = getattr(native_module, name, None)
    if not callable(value):
        raise error_type(f"{client_label} is missing callable {name}")
    return value


def optional_callable(native_module: Any, name: str) -> Callable[..., Any] | None:
    value = getattr(native_module, name, None)
    return value if callable(value) else None


def capability_names(capabilities: Mapping[str, Any], key: str) -> set[str]:
    values = capabilities.get(key, ())
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        return set()
    return {str(value) for value in values}


def native_response_diagnostics(response: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = dict(response)
    diagnostics.pop("response_handle", None)
    return diagnostics


def coerce_mapping_int(mapping: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(mapping.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def coerce_mapping_float(mapping: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)
