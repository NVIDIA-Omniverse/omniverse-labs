# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python implementation of the backend-neutral :class:`ovfmi.FmiHost`."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from .backend import FmpyBackend, MasterBackend
from .population import deserialise_instances, parse_source
from .routing import InputRouter, OutputRouter, write_stage_group
from .types import (
    AttributeWrite,
    InstanceInfo,
    FmiHostConfig,
    OperationIndex,
    PopulationReport,
    ReadResult,
)


class FmiHost:
    """Own FMI/SSP instances connected to one caller-owned ovstage.

    A host is initially detached. Call :meth:`attach_ovstage` before data-plane
    or simulation methods, and :meth:`release` when finished. The host owns its
    FMI backend instances but never owns the supplied ovstage object.

    Args:
        config: Discovery, validation, and missing-input behavior. Defaults to
            :class:`FmiHostConfig`.

    The private ``_backend_factory`` keyword is reserved for ovfmi tests and
    backend integration; applications must not depend on it.
    """

    def __init__(
        self,
        config: FmiHostConfig | None = None,
        *,
        _backend_factory=FmpyBackend,
    ) -> None:
        self._config = config or FmiHostConfig()
        self._backend_factory = _backend_factory
        self._backend: MasterBackend | None = None
        self._input_router: InputRouter | None = None
        self._output_router: OutputRouter | None = None
        self._stage = None
        self._time = 0.0
        self._instances: tuple[InstanceInfo, ...] = ()
        self._next_operation = 1
        self._completed_operations: set[int] = set()
        self._released = False

    @property
    def time(self) -> float:
        """Current simulation time in seconds.

        The value starts at zero on attachment, advances after each successful
        step, and returns to zero after :meth:`reset`.
        """
        return self._time

    @property
    def instances(self) -> tuple[InstanceInfo, ...]:
        """Currently attached instances, or an empty tuple while detached."""
        return self._instances

    def _require_attached(self) -> None:
        if self._released:
            raise RuntimeError("FmiHost object is released")
        if self._stage is None or self._backend is None:
            raise RuntimeError("FmiHost object is not attached")

    def attach_ovstage(
        self,
        stage,
        *,
        source_asset: str | Path | None = None,
    ) -> PopulationReport:
        """Attach an ovstage and instantiate its authored FMI/SSP models.

        Args:
            stage: A populated, caller-owned ovstage object. ovfmi borrows it
                until detachment and does not release it.
            source_asset: USD source used to discover custom FMI schema
                attributes not present in the ovstage population.

        Returns:
            A report describing every discovered instance.

        Raises:
            ValueError: If ``stage`` is ``None``.
            RuntimeError: If the host was released or no source asset is
                supplied.
            Exception: Parsing and FMU/SSP instantiation errors are propagated
                after partially created backend state is released.

        Attaching again first detaches the current stage. Simulation time and
        outstanding operation tokens are reset.
        """
        if self._released:
            raise RuntimeError("FmiHost object is released")
        if stage is None:
            raise ValueError("stage must not be None")
        if source_asset is None:
            raise RuntimeError(
                "source_asset is required because ovstage population does "
                "not include custom FMI schema attributes"
            )
        if self._stage is not None:
            self.detach_ovstage()

        parsed = parse_source(str(source_asset))
        raw_instances = parsed.get("instances", {})
        parser_instances = deserialise_instances(
            raw_instances,
            root_prim=self._config.root_prim,
            enable_ssp=self._config.enable_ssp,
        )
        initial_values = parsed.get("initial_values", {})
        input_router = InputRouter(initial_values, self._config.missing_input_policy)
        output_router = OutputRouter(
            initial_values,
            input_router,
            strict=self._config.strict_schema_validation,
        )
        backend = self._backend_factory(input_router, output_router)

        try:
            backend.populate(parser_instances)
        except Exception:
            backend.release()
            raise

        infos = []
        for path, parser_instance in parser_instances.items():
            raw = raw_instances[path]
            is_ssp = bool(raw.get("ssp"))
            infos.append(
                InstanceInfo(
                    prim_path=path,
                    source_asset=str(parser_instance.fmu),
                    kind="ssp" if is_ssp else "fmu",
                    enabled=bool(parser_instance.enabled),
                )
            )

        self._stage = stage
        self._input_router = input_router
        self._output_router = output_router
        self._backend = backend
        self._instances = tuple(infos)
        self._time = 0.0
        self._completed_operations.clear()
        return PopulationReport(instances=self._instances)

    def detach_ovstage(self) -> None:
        """Release backend instances and forget the stage.

        This method is idempotent. It does not release the caller-owned
        ovstage and does not permanently release the host.
        """
        if self._backend is not None:
            self._backend.release()
        self._backend = None
        self._input_router = None
        self._output_router = None
        self._stage = None
        self._instances = ()
        self._completed_operations.clear()

    def update_from_ovstage(self, from_ordinal: int, to_ordinal: int) -> None:
        """Refresh mapped FMI inputs from the latest value at ``to_ordinal``.

        ``from_ordinal`` and ``to_ordinal`` describe the caller's valid ordinal
        window and must be ordered. The current Python backend samples the
        latest value at the inclusive upper bound; it does not replay every
        ordinal in the range.
        """
        self._require_attached()
        self._input_router.update_from_stage(
            self._stage, int(from_ordinal), int(to_ordinal)
        )

    def write(self, writes: Sequence[AttributeWrite]) -> OperationIndex:
        """Supply USD-identified values as FMI inputs.

        The Python backend performs the write synchronously and returns a
        completion token for API uniformity. Consume the token with
        :meth:`wait_op`, or all outstanding tokens with :meth:`wait_all`.
        """
        self._require_attached()
        self._input_router.write(list(writes))
        return self._complete_operation()

    def step(self, dt: float) -> OperationIndex:
        """Advance all instances by ``dt`` seconds and return an operation token.

        ``dt`` must be positive and finite. The current backend completes the
        step before returning, while retaining token-based completion semantics
        to preserve the backend-neutral API's completion model.
        """
        self._require_attached()
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a positive finite number")
        target_time = self._time + float(dt)
        self._backend.step(target_time)
        self._time = target_time
        return self._complete_operation()

    def step_sync(self, dt: float) -> None:
        """Advance by ``dt`` seconds and consume the completion token."""
        self.wait_op(self.step(dt))

    def _complete_operation(self) -> int:
        operation = self._next_operation
        self._next_operation += 1
        self._completed_operations.add(operation)
        return operation

    def wait_op(self, operation: OperationIndex, timeout_ns: int | None = None) -> None:
        """Consume one completed operation token.

        ``timeout_ns`` must be non-negative when supplied. It has no timing
        effect in the synchronous Python backend. Unknown or already-consumed
        tokens raise :class:`ValueError`.
        """
        self._require_attached()
        if timeout_ns is not None and timeout_ns < 0:
            raise ValueError("timeout_ns must be non-negative")
        if int(operation) not in self._completed_operations:
            raise ValueError(f"unknown operation index {operation}")
        self._completed_operations.remove(int(operation))

    def wait_all(self, timeout_ns: int | None = None) -> None:
        """Consume all completed operation tokens.

        ``timeout_ns`` must be non-negative when supplied. It has no timing
        effect in the synchronous Python backend.
        """
        self._require_attached()
        if timeout_ns is not None and timeout_ns < 0:
            raise ValueError("timeout_ns must be non-negative")
        self._completed_operations.clear()

    def read(
        self,
        prim_paths: Sequence[str] | None = None,
        attribute_names: Sequence[str] | None = None,
    ) -> ReadResult:
        """Snapshot latest FMI outputs, optionally filtered in USD space.

        Args:
            prim_paths: Exact prim paths to retain, or ``None`` for all.
            attribute_names: Exact USD attribute names to retain, or ``None``
                for all.

        Returns:
            An owned, context-managed :class:`ReadResult`. Filters that match
            nothing produce an empty ``groups`` tuple.
        """
        self._require_attached()
        return ReadResult(self._output_router.snapshot(prim_paths, attribute_names))

    def write_to_ovstage(
        self,
        ordinal: int,
        prim_paths: Sequence[str] | None = None,
        attribute_names: Sequence[str] | None = None,
    ) -> int:
        """Write latest FMI outputs into an ovstage ordinal.

        The caller owns ordinal allocation and sealing. ``prim_paths`` and
        ``attribute_names`` use the same exact-match filtering as :meth:`read`.

        Returns:
            Number of output groups written.
        """
        self._require_attached()
        with self.read(prim_paths, attribute_names) as result:
            for group in result.groups:
                write_stage_group(self._stage, group, int(ordinal))
            return len(result.groups)

    def reset(self, start_time: float = 0.0) -> None:
        """Reset attached models and clear operation tokens.

        The FMPy backend currently supports only ``start_time=0.0``; another
        value raises :class:`NotImplementedError`.
        """
        self._require_attached()
        if start_time != 0.0:
            raise NotImplementedError("the fmpy backend currently resets only to t=0")
        self._backend.reset()
        self._time = 0.0
        self._completed_operations.clear()

    def release(self) -> None:
        """Permanently release this host.

        Repeated calls are harmless. An attached stage is detached first, and
        no subsequent attachment or simulation operation is permitted.
        """
        if self._released:
            return
        self.detach_ovstage()
        self._released = True

    def __enter__(self) -> "FmiHost":
        """Return this host for a managed lifetime."""
        if self._released:
            raise RuntimeError("FmiHost object is released")
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()
