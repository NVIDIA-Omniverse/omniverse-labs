# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author the RTPT render-quality values into the OVRTX worker's startup
``ovrtx.config.json`` carb-settings file.

Why this exists: a real-GPU A/B (runtime measurements, "OVRTX worker ignores
omni:rtx:rtpt:* render-quality attributes on the RenderProduct" / "How to make
the RTPT quality sliders take effect") proved that this OVRTX worker build
ignores the ``omni:rtx:rtpt:*`` attributes authored on the ``RenderProduct``
prim and rejects live writes to them, but *does* honor the same values when they
arrive as carb settings under ``/rtx/rtpt/*`` in the worker package's
``ovrtx.config.json`` read once at worker-process launch. Lowering
``/rtx/rtpt/maxBounces`` from 8 to 0 took a Cornell-box render from luma 86.3 to
0.0 deterministically, while the RenderProduct attribute did nothing.

This module maps the four RTPT quality values (the ``RTPT_RENDER_SETTINGS``
single source of truth) into that carb-settings tree and merges them into the
config file at the worker package root, preserving the other opinions
(``log``/``app``/``crashreporter``). It is intentionally free of ``bpy`` and of
any render RPCs so it is unit-testable and callable from the session-ensure path
before the worker launches.

Launch-only channel: the worker reads the file once at process start and a
running worker is reused across session re-keys, so a value change only reaches
the renderer when the worker process (re)starts. Callers author the current
values before every launch; the UI tells the artist a restart is required.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from .command_line import split_command
from .properties import (
    DLSS_DISABLED_EXECMODE,
    DLSS_EXECMODE_ATTRIBUTE,
    RTPT_RENDER_SETTINGS,
)


WORKER_CONFIG_FILENAME = "ovrtx.config.json"

#: carb-settings namespace segment that ``omni:rtx:rtpt:*`` attribute names carry
#: as their first (vendor) token; dropped when converting to a carb path so
#: ``omni:rtx:rtpt:maxBounces`` becomes ``/rtx/rtpt/maxBounces``.
_ATTRIBUTE_VENDOR_PREFIX = "omni"


def _tolerant_config_load(text: str) -> dict[str, Any]:
    """Parse an existing ``ovrtx.config.json``.

    carb's JSON serializer tolerates trailing commas that ``json`` rejects, so
    strip them before parsing. A missing/blank/corrupt file yields an empty
    settings tree the caller can still author into.
    """

    stripped = (text or "").strip()
    if not stripped:
        return {}
    try:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", stripped))
    except (ValueError, TypeError):
        return {}


def _carb_path_segments(attribute: str) -> list[str]:
    """carb-settings path segments for one RTPT ``RenderProduct`` attribute name.

    ``omni:rtx:rtpt:maxBounces`` -> ``["rtx", "rtpt", "maxBounces"]`` and
    ``omni:rtx:rtpt:fireflyFilter:enabled`` ->
    ``["rtx", "rtpt", "fireflyFilter", "enabled"]``. These match the carb setting
    paths the RealTimePathTracing engine reads (verified in the worker's
    ``rtx.raytracing.plugin`` binary).
    """

    segments = [segment for segment in str(attribute).split(":") if segment]
    if segments and segments[0] == _ATTRIBUTE_VENDOR_PREFIX:
        segments = segments[1:]
    return segments


def _set_nested(tree: dict[str, Any], segments: list[str], value: Any) -> None:
    cursor = tree
    for segment in segments[:-1]:
        existing = cursor.get(segment)
        if not isinstance(existing, dict):
            existing = {}
            cursor[segment] = existing
        cursor = existing
    cursor[segments[-1]] = value


def _remove_nested(tree: dict[str, Any], segments: list[str]) -> None:
    """Delete the leaf at ``segments`` and prune any now-empty ancestor dicts.

    Used to express "author nothing" for a key the add-on owns: removing the
    leaf (rather than leaving it or overwriting it) restores the engine default,
    and pruning the containers this add-on created keeps the file clean. Absent
    paths and non-dict ancestors are left untouched.
    """

    stack: list[tuple[dict[str, Any], str]] = []
    cursor = tree
    for segment in segments[:-1]:
        nxt = cursor.get(segment)
        if not isinstance(nxt, dict):
            return
        stack.append((cursor, segment))
        cursor = nxt
    cursor.pop(segments[-1], None)
    for parent, key in reversed(stack):
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            del parent[key]
        else:
            break


def rtpt_carb_overrides(rtpt_quality: Mapping[str, Any] | None) -> dict[str, Any]:
    """The ``/rtx/rtpt/*`` carb-settings subtree for the given RTPT quality values.

    Iterates ``RTPT_RENDER_SETTINGS`` (single source of truth) so the four
    controls have one attribute-name/dtype/default definition and one UI->wire
    conversion. ``rtpt_quality`` carries the artist-facing UI values; each is
    converted to the wire value via ``spec.to_wire`` (Max Bounces adds the
    camera-ray +2 offset, the sub-caps pass through). A value absent from
    ``rtpt_quality`` falls back to the documented UI default, so the authored
    config is always complete and deterministic and its wire defaults stay
    3/3/15/true. Returns e.g. ``{"rtx": {"rtpt": {"maxBounces": 10, ...}}}``.
    """

    quality = rtpt_quality or {}
    overrides: dict[str, Any] = {}
    for name, spec in RTPT_RENDER_SETTINGS.items():
        value = spec.to_wire(quality.get(name, spec.default))
        _set_nested(overrides, _carb_path_segments(spec.attribute), value)
    return overrides


def dlss_carb_overrides(dlss_enabled: bool) -> dict[str, Any]:
    """The ``/rtx/post/dlss/*`` carb subtree for the DLSS toggle, or empty.

    ``dlss_enabled=True`` leaves the engine default (no override); ``False``
    writes the DLSS Performance-preset execMode value so a freshly launched
    worker honors the toggle from its startup config. The carb path is derived
    from the same ``omni:rtx:post:dlss:execMode`` attribute the RenderProduct
    channel authors, so both channels stay in lockstep. (This worker exposes no
    full DLSS off; ``False`` selects the Performance execution mode.)
    """

    if dlss_enabled:
        return {}
    overrides: dict[str, Any] = {}
    _set_nested(
        overrides,
        _carb_path_segments(DLSS_EXECMODE_ATTRIBUTE),
        int(DLSS_DISABLED_EXECMODE),
    )
    return overrides


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        existing = base.get(key)
        if isinstance(value, Mapping) and isinstance(existing, dict):
            _deep_merge(existing, value)
        else:
            base[key] = dict(value) if isinstance(value, Mapping) else value
    return base


def compose_worker_config(
    config_text: str,
    rtpt_quality: Mapping[str, Any] | None,
    dlss_enabled: bool = True,
) -> str:
    """Merge the RTPT + DLSS carb overrides into an existing config, return JSON.

    Preserves every other opinion in the file and overwrites only the
    ``/rtx/rtpt/*`` leaves this add-on owns, plus ``/rtx/post/dlss/execMode``
    when the DLSS toggle is OFF. When the toggle is ON the add-on authors
    nothing for DLSS: any ``/rtx/post/dlss/execMode`` this add-on wrote in a
    previous OFF state is REMOVED so a freshly launched worker returns to the
    engine default rather than staying stuck in the Performance preset. Emits
    standard JSON (no trailing commas), which carb also accepts.
    """

    tree = _tolerant_config_load(config_text)
    _deep_merge(tree, rtpt_carb_overrides(rtpt_quality))
    if dlss_enabled:
        _remove_nested(tree, _carb_path_segments(DLSS_EXECMODE_ATTRIBUTE))
    else:
        _deep_merge(tree, dlss_carb_overrides(dlss_enabled))
    return json.dumps(tree, indent=4) + "\n"


def worker_config_path(worker_command: str) -> Path | None:
    """Locate ``ovrtx.config.json`` under the worker command's ``--package-root``.

    Returns ``None`` when the command carries no package root (e.g. an
    env-configured external worker), in which case there is no add-on-owned
    config file to author and the caller silently skips.
    """

    try:
        parts = split_command(worker_command)
    except ValueError:
        return None
    package_root: str | None = None
    for index, part in enumerate(parts):
        if part == "--package-root" and index + 1 < len(parts):
            package_root = parts[index + 1]
            break
        if part.startswith("--package-root="):
            package_root = part.split("=", 1)[1]
            break
    if not package_root:
        return None
    root = Path(package_root).expanduser()
    if not root.is_absolute():
        root = root.resolve()
    return root / WORKER_CONFIG_FILENAME


def author_worker_config(
    worker_command: str,
    rtpt_quality: Mapping[str, Any] | None,
    dlss_enabled: bool = True,
) -> dict[str, Any]:
    """Write the current RTPT quality values into the worker's startup config.

    Idempotent and best-effort: reads any existing config, merges the RTPT
    ``/rtx/rtpt/*`` overrides, and rewrites the file atomically only when the
    content actually changes. Never raises — a failure to author the config must
    not break session startup; it just means the launch-time channel was not
    updated. Returns a small diagnostics mapping describing the outcome.
    """

    path = worker_config_path(worker_command)
    if path is None:
        return {"status": "skipped", "reason": "no_package_root"}
    try:
        # errors="replace" so a corrupt config whose bytes are not valid UTF-8
        # cannot raise UnicodeDecodeError (a ValueError, not an OSError) out of
        # this best-effort path: the undecodable text fails the tolerant JSON
        # parse and is treated as an empty tree, exactly like corrupt JSON, so a
        # fresh valid config is authored instead of breaking session startup.
        existing = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        composed = compose_worker_config(existing, rtpt_quality, dlss_enabled)
        if existing == composed:
            return {"status": "unchanged", "path": str(path)}
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(composed, encoding="utf-8")
        os.replace(tmp, path)
        return {"status": "written", "path": str(path)}
    except (OSError, ValueError) as exc:
        return {"status": "failed", "path": str(path), "error": f"{type(exc).__name__}: {exc}"}


__all__ = [
    "WORKER_CONFIG_FILENAME",
    "author_worker_config",
    "compose_worker_config",
    "dlss_carb_overrides",
    "rtpt_carb_overrides",
    "worker_config_path",
]
