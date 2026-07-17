# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable add-on USD opinion records for scene generations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha256
from types import MappingProxyType
from typing import Any


class SceneGenerationError(RuntimeError):
    """Raised when current Blender state cannot form a valid generation."""

    def __init__(self, message: str, diagnostics: tuple[Mapping[str, Any], ...] = ()) -> None:
        super().__init__(message)
        self.diagnostics = tuple(MappingProxyType(dict(item)) for item in diagnostics)


class AddOnUsdOpinionRecord:
    """One immutable complete add-on contribution rooted at a USD prim."""

    __slots__ = (
        "_layer",
        "authored_paths",
        "diagnostics",
        "digest",
        "layer_text",
        "usd_prim_path",
    )

    def __init__(
        self,
        usd_prim_path: str,
        layer: Any,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        from pxr import Sdf  # type: ignore

        root = Sdf.Path(str(usd_prim_path or ""))
        if (
            not root.IsAbsolutePath()
            or not root.IsPrimPath()
            or root == Sdf.Path.absoluteRootPath
        ):
            raise SceneGenerationError(
                "add-on USD opinion record root must be an absolute prim path",
                ({"usd_prim_path": str(root), "reason": "invalid_record_root"},),
            )
        if not isinstance(layer, Sdf.Layer):
            raise TypeError("add-on USD opinion record layer must be an SdfLayer")
        copied = Sdf.Layer.CreateAnonymous(".usda")
        copied.TransferContent(layer)
        authored_paths = _authored_paths(Sdf, copied, root)
        if str(root) not in authored_paths:
            raise SceneGenerationError(
                "add-on USD opinion record does not author its root prim",
                ({"usd_prim_path": str(root), "reason": "record_root_missing"},),
            )
        text = copied.ExportToString()
        object.__setattr__(self, "usd_prim_path", str(root))
        object.__setattr__(self, "authored_paths", authored_paths)
        object.__setattr__(self, "digest", sha256(
            _length_delimited(
                b"add-on-usd-opinion-record-v1",
                self.usd_prim_path.encode("utf-8"),
                text.encode("utf-8"),
            )
        ).hexdigest())
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(diagnostics or {})))
        object.__setattr__(self, "layer_text", text)
        copied.SetPermissionToEdit(False)
        object.__setattr__(self, "_layer", copied)

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("add-on USD opinion records are immutable")


class SparseAddOnOpinionChange:
    """Opinion-record replacements and removals relative to a predecessor."""

    __slots__ = ("removed_usd_prim_paths", "replacement_records")

    def __init__(
        self,
        replacement_records: Iterable[AddOnUsdOpinionRecord] = (),
        removed_usd_prim_paths: Iterable[str] = (),
    ) -> None:
        object.__setattr__(self, "replacement_records", tuple(
            sorted(replacement_records, key=lambda record: record.usd_prim_path)
        ))
        object.__setattr__(
            self,
            "removed_usd_prim_paths",
            tuple(sorted(set(removed_usd_prim_paths))),
        )

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("sparse add-on opinion changes are immutable")


def _authored_paths(Sdf: Any, layer: Any, root: Any) -> tuple[str, ...]:
    authored = []
    invalid = []

    def visit(path: Any) -> None:
        if path == Sdf.Path.absoluteRootPath:
            return
        if path.HasPrefix(root):
            authored.append(str(path))
            return
        if root.HasPrefix(path) and path.IsPrimPath():
            spec = layer.GetObjectAtPath(path)
            if spec is not None and set(spec.ListInfoKeys()).difference({"specifier"}):
                invalid.append(str(path))
            return
        invalid.append(str(path))

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    if invalid:
        raise SceneGenerationError(
            "add-on USD opinion record authors outside its root",
            tuple(
                {
                    "usd_prim_path": str(root),
                    "authored_path": path,
                    "reason": "record_path_outside_root",
                }
                for path in sorted(set(invalid))
            ),
        )
    return tuple(sorted(set(authored)))


def _length_delimited(*parts: bytes) -> bytes:
    return b"".join(len(part).to_bytes(8, "big") + part for part in parts)


__all__ = [
    "AddOnUsdOpinionRecord",
    "SceneGenerationError",
    "SparseAddOnOpinionChange",
]
