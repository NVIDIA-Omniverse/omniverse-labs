# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live RTPT render-quality change application (task01-04).

A quality-setting change on a running viewport session is applied as a
runtime attribute write on the active ``RenderProduct``, executed on the
session-owning render thread. This module builds the value-update intent
and routes it to the active viewport engines; the actual write happens on
the render thread (``ViewUpdateStream.apply_pending``), followed by the
existing refinement restart (render at ``min_samples``, refine to
``max_samples``) which is the warm-up.

The retired ``live_bridge.set_render_setting()`` implementation is not used:
application flows through the current session/render-thread ownership model
(``ViewUpdateStream`` / ``RuntimeScheduler`` / the viewport render loop),
reusing the same latest-wins value-update plumbing as material/light edits.

Importable without ``bpy`` so the composition/plumbing tests share the same
attribute-name/dtype contract as the runtime.
"""

from __future__ import annotations

from typing import Any

from .interactive_edit_planner import (
    DataAuthority,
    EditShape,
    InteractiveEdit,
    RENDER_SETTING_VALUE_SOURCE,
    edit_location,
)
from .properties import RTPT_RENDER_SETTINGS


#: RTPT dtype -> USD value-update type role (the string the worker maps via
#: ``Sdf.ValueTypeNames``). USD ``Int`` is 32-bit, satisfying the ``int32``
#: contract without widening to ``Int64``; ``bool`` maps to ``Bool``.
_RTPT_VALUE_UPDATE_TYPES = {"int32": "Int", "bool": "Bool"}


def value_update_type(dtype: str) -> str:
    """USD value-update type role for an RTPT dtype string."""

    try:
        return _RTPT_VALUE_UPDATE_TYPES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported RTPT dtype: {dtype!r}") from exc


def render_setting_edit(
    property_name: str,
    value: Any,
    render_product_path: str,
) -> InteractiveEdit:
    """Build the live render-setting value edit for one RTPT property.

    Targets the active ``RenderProduct`` prim with the exact documented
    attribute name and dtype from ``RTPT_RENDER_SETTINGS`` (task01-01's
    single source of truth); the value-update type role rides provenance so
    the render-thread applier writes the exact dtype.

    ``value`` is the artist-facing UI value; the edit's value is the wire value
    OVRTX consumes (``spec.to_wire`` applies the Max Bounces +2 camera-ray
    offset, sub-caps pass through), so the live runtime write matches the USD
    and worker-config channels. The original UI value rides provenance
    (``ui_value``) for diagnostics.
    """

    spec = RTPT_RENDER_SETTINGS[property_name]
    update_type = value_update_type(spec.dtype)
    return InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path=str(render_product_path),
            usd_attribute=spec.attribute,
            provenance={
                "source": RENDER_SETTING_VALUE_SOURCE,
                "value_type": update_type,
                "property_name": property_name,
                # Exact documented dtype contract ("int32"/"bool"), carried so
                # the diagnostics record (task01-05) states the authored dtype
                # without re-deriving it from RTPT_RENDER_SETTINGS.
                "dtype": spec.dtype,
                # Artist-facing UI value carried alongside the wire value so the
                # diagnostics record can report both what the artist set and
                # what was actually sent.
                "ui_value": spec.from_wire(spec.to_wire(value)),
            },
        ),
        value=spec.to_wire(value),
    )


def render_setting_edit_for_request(
    property_name: str,
    value: Any,
    request: Any,
) -> InteractiveEdit | None:
    """Live edit for a session request, or ``None`` when no session is active.

    A change with no active session (no request, or a request without a
    render product path) authors no runtime write — the property update
    alone stands, and the value reaches the next session through the
    composition authoring (task01-03).
    """

    if request is None:
        return None
    render_product_path = str(getattr(request, "render_product_path", "") or "")
    if not render_product_path:
        return None
    if property_name not in RTPT_RENDER_SETTINGS:
        return None
    return render_setting_edit(property_name, value, render_product_path)


def dispatch_render_setting_change(property_name: str, value: Any) -> Any:
    """Route a live RTPT change to active viewport engines (Blender runtime).

    Lazy-imports the engine module (which requires ``bpy``) so this module
    stays importable in the plain test lane; the engine fills in the active
    render product path and submits on the session-owning render thread.
    """

    from . import engine

    return engine.submit_render_setting_change_to_active_viewports(
        property_name, value
    )


__all__ = [
    "RENDER_SETTING_VALUE_SOURCE",
    "dispatch_render_setting_change",
    "render_setting_edit",
    "render_setting_edit_for_request",
    "value_update_type",
]
