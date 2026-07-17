# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Writer contract for interactive edit persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .interactive_edit_planner import InteractiveEdit


@dataclass(frozen=True)
class WriteRequest:
    edits: tuple[InteractiveEdit, ...]
    reason: str
    usd_layer_id: str = ""


@dataclass(frozen=True)
class WriteResult:
    requested: bool
    completed: bool
    reason: str
    path: str = ""
    usd_layer_id: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


EditWriter = Callable[[WriteRequest], WriteResult]
