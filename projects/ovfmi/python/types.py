# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral value types exported by :mod:`ovfmi`."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence


ArrayLike = Any
OperationIndex = int


class MissingInputPolicy(Enum):
    """Select how :class:`FmiHost` initializes an unavailable mapped FMI input."""

    #: Reject attachment when a mapped USD value is unavailable.
    ERROR = "error"
    #: Leave the variable at the start value declared by its FMU.
    USE_FMU_START_VALUE = "use_fmu_start_value"
    #: Initialize an unavailable scalar or vector component to zero.
    ZERO = "zero"


@dataclass(frozen=True)
class FmiHostConfig:
    """Configuration applied when a :class:`FmiHost` attaches to an ovstage.

    Attributes:
        root_prim: Absolute USD prim path limiting FMI/SSP discovery.
        missing_input_policy: Policy for mapped inputs without an authored
            initial USD value.
        strict_schema_validation: Reject overlapping output mappings when
            true. When false, the last discovered mapping owns the component.
        enable_ssp: Discover and instantiate ``SspInstance`` prims in addition
            to ``FmuInstance`` prims.
    """

    root_prim: str = "/"
    missing_input_policy: MissingInputPolicy = MissingInputPolicy.USE_FMU_START_VALUE
    strict_schema_validation: bool = True
    enable_ssp: bool = True


@dataclass(frozen=True)
class InstanceInfo:
    """Description of one FMI or SSP instance created during attachment.

    Attributes:
        prim_path: Absolute path of the authoring prim in the USD source.
        source_asset: Resolved or authored path of the referenced FMU/SSP.
        kind: ``"fmu"`` or ``"ssp"``.
        enabled: Whether the authored instance is enabled.
    """

    prim_path: str
    source_asset: str
    kind: str
    enabled: bool


@dataclass(frozen=True)
class PopulationReport:
    """Result returned by :meth:`FmiHost.attach_ovstage`.

    Attributes:
        instances: Instances discovered below :attr:`FmiHostConfig.root_prim`, in
            source discovery order.
    """

    instances: tuple[InstanceInfo, ...]


@dataclass(frozen=True)
class AttributeWrite:
    """One batch of USD-identified values supplied as FMI inputs.

    ``values`` must contain one row for each entry in ``prim_paths``. Scalars
    may be represented as a one-dimensional sequence. Ragged-array writes are
    not implemented and raise :class:`NotImplementedError`.

    Attributes:
        prim_paths: Target USD prim paths.
        attribute_name: USD attribute identity used by authored FMI mappings.
        values: Array-like scalar/vector rows in the same order as
            ``prim_paths``.
        is_array: Whether rows represent ragged USD arrays. Only ``False`` is
            currently supported.
    """

    prim_paths: tuple[str, ...]
    attribute_name: str
    values: ArrayLike | tuple[ArrayLike, ...]
    is_array: bool = False


@dataclass(frozen=True)
class ReadGroup:
    """A homogeneous group of output values returned by :meth:`FmiHost.read`.

    Attributes:
        prim_paths: USD prim paths corresponding to tensor rows.
        attribute_name: USD attribute identity of the values.
        tensors: Backend-neutral array objects holding the values. A regular
            group contains one tensor; a ragged group may contain several.
        is_array: Whether the group contains ragged USD array values.
        semantic: Integer ovstage attribute semantic, or zero when none is
            required.
    """

    prim_paths: tuple[str, ...]
    attribute_name: str
    tensors: tuple[ArrayLike, ...]
    is_array: bool
    semantic: int = 0


class ReadResult:
    """Owned snapshot of the latest completed ovfmi outputs.

    Use this object as a context manager. Accessing :attr:`groups` after
    :meth:`close` raises :class:`RuntimeError`.
    """

    def __init__(self, groups: Sequence[ReadGroup]):
        self._groups = tuple(groups)
        self._closed = False

    @property
    def groups(self) -> tuple[ReadGroup, ...]:
        """Return output groups retained by this snapshot."""
        if self._closed:
            raise RuntimeError("read result is closed")
        return self._groups

    def close(self) -> None:
        """Release this snapshot; calling the method repeatedly is harmless."""
        self._closed = True

    def __enter__(self) -> "ReadResult":
        if self._closed:
            raise RuntimeError("read result is closed")
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()
