# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed inputs crossing from Blender into add-on-owned payloads."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar


SignalT = TypeVar("SignalT")
PayloadT = TypeVar("PayloadT")


class BlenderSignalTranslator(Protocol, Generic[SignalT, PayloadT]):
    def translate(self, signal: SignalT) -> PayloadT: ...


class BlenderRenderSignalSource(str, Enum):
    VIEW_UPDATE = "view_update"
    VIEW_DRAW = "view_draw"
    FINAL_RENDER = "final_render"


class BlenderRenderIntent(str, Enum):
    VIEWPORT = "viewport"
    FINAL_RENDER = "final_render"


@dataclass(frozen=True)
class BlenderRenderSignal:
    source: BlenderRenderSignalSource
    intent: BlenderRenderIntent
    scene: Any
    input_usd_path: str
    camera_prim_path: str
    render_product_path: str
    context: Any | None = None
    engine_id: str = ""
    current_scene_generation: bool = False


class BlenderEditSignalSource(str, Enum):
    DEPSGRAPH = "depsgraph"
    OPERATOR = "operator"
    SELECTION = "selection"


@dataclass(frozen=True)
class BlenderEditSignal:
    source: BlenderEditSignalSource
    id_items: tuple[Any, ...]
    context: Any | None = None
    input_usd_path: str | None = None
    ignored_layer_identifiers: tuple[str, ...] = ()
