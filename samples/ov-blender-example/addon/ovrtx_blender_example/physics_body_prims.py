# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physics body prim discovery for shared-stage composition."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any


DEFAULT_DYNAMIC_BODY_ROOT = "/World/PhysicsIsland/DynamicBodies"


def discover_dynamic_body_prims(input_usd_path: str, root: str = DEFAULT_DYNAMIC_BODY_ROOT) -> tuple[str, ...]:
    path = Path(input_usd_path).expanduser()
    if not path.is_file():
        return ()
    discovered = _discover_dynamic_body_prims_with_usd(path, root)
    if discovered:
        return discovered
    if path.suffix.lower() == ".usda":
        return _discover_dynamic_body_prims_from_usda(path, root)
    return ()


def _discover_dynamic_body_prims_with_usd(path: Path, root: str) -> tuple[str, ...]:
    try:
        from pxr import Usd  # type: ignore
    except Exception:
        return ()
    try:
        stage = Usd.Stage.Open(str(path))
    except Exception:
        return ()
    if stage is None:
        return ()
    prefix = root.rstrip("/") + "/"
    body_paths: list[str] = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path != root and not prim_path.startswith(prefix):
            continue
        try:
            schemas = set(str(schema) for schema in prim.GetAppliedSchemas())
        except Exception:
            schemas = set()
        if "PhysicsRigidBodyAPI" not in schemas:
            continue
        kinematic_attr = prim.GetAttribute("physics:kinematicEnabled")
        is_kinematic = bool(kinematic_attr and kinematic_attr.Get())
        if not is_kinematic:
            body_paths.append(prim_path)
    return tuple(body_paths)


_USDA_PRIM_RE = re.compile(r'^\s*(?:def|over|class)(?:\s+\w+)?\s+"([^"]+)"')


def _discover_dynamic_body_prims_from_usda(path: Path, root: str) -> tuple[str, ...]:
    class _PrimBlock:
        def __init__(self, path: str, close_depth: int, has_rigid_body: bool, is_kinematic: bool) -> None:
            self.path = path
            self.close_depth = close_depth
            self.has_rigid_body = has_rigid_body
            self.is_kinematic = is_kinematic

    lines = path.read_text(encoding="utf-8").splitlines()
    prefix = root.rstrip("/") + "/"
    depth = 0
    stack: list[_PrimBlock] = []
    pending: dict[str, Any] | None = None
    body_paths: list[str] = []

    def _current_parent_path() -> str:
        return stack[-1].path if stack else ""

    def _push(item: dict[str, Any], close_depth: int) -> None:
        stack.append(
            _PrimBlock(
                path=str(item["path"]),
                close_depth=close_depth,
                has_rigid_body=bool(item["has_rigid_body"]),
                is_kinematic=bool(item["is_kinematic"]),
            )
        )

    def _close_completed_blocks() -> None:
        while stack and depth <= stack[-1].close_depth:
            block = stack.pop()
            if (
                (block.path == root or block.path.startswith(prefix))
                and block.has_rigid_body
                and not block.is_kinematic
            ):
                body_paths.append(block.path)

    for line in lines:
        open_count = line.count("{")
        close_count = line.count("}")
        rigid = "PhysicsRigidBodyAPI" in line
        kinematic = "physics:kinematicEnabled" in line and "true" in line.lower()

        if pending is not None:
            pending["has_rigid_body"] = bool(pending["has_rigid_body"] or rigid)
            pending["is_kinematic"] = bool(pending["is_kinematic"] or kinematic)
            if open_count:
                _push(pending, depth)
                pending = None
        else:
            match = _USDA_PRIM_RE.match(line)
            if match:
                name = match.group(1)
                parent = _current_parent_path()
                prim_path = f"{parent}/{name}" if parent else f"/{name}"
                pending = {
                    "path": prim_path,
                    "has_rigid_body": rigid,
                    "is_kinematic": kinematic,
                }
                if open_count:
                    _push(pending, depth)
                    pending = None
            elif stack:
                stack[-1].has_rigid_body = bool(stack[-1].has_rigid_body or rigid)
                stack[-1].is_kinematic = bool(stack[-1].is_kinematic or kinematic)

        depth += open_count - close_count
        _close_completed_blocks()

    return tuple(body_paths)


__all__ = [
    "DEFAULT_DYNAMIC_BODY_ROOT",
    "discover_dynamic_body_prims",
]
