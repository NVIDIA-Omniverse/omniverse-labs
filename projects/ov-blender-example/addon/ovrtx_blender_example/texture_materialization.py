# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize Blender images as on-disk files for external runtimes.

The OVRTX worker reads texture assets from the filesystem; Blender images
packed into the ``.blend`` have no such file, so their original bytes are
written once to a content-addressed cache the composed USD can reference.
Real user scenes (e.g. downloaded splash files) commonly pack everything.
"""

from __future__ import annotations

import os
import re
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def texture_cache_directory() -> Path:
    """Content-addressed image cache beside the authoring workspace.

    ``.resolve()`` is load-bearing on Windows: the POSIX-style default
    expands drive-relative otherwise (same class as the composed-sublayer
    regression, 2026-07-07).
    """

    work_root = Path(
        os.environ.get("OV_BLENDER_EXAMPLE_AUTHORING_WORK_DIR")
        or Path(tempfile.gettempdir()) / "ov-blender-example" / "authored-scenes"
    ).expanduser().resolve()
    return work_root.parent / "texture-cache"


def materialized_image_path(image: Any) -> str:
    """An on-disk file for a Blender image, or empty string.

    On-disk images resolve through their (library-aware) filepath. Packed
    images export their original packed bytes once to the texture cache,
    keyed by content digest — lossless, format-preserving, and stable
    across reconciles so composed layer digests do not churn. Images with
    neither a readable file nor packed bytes (generated/render results)
    return empty and the caller degrades.
    """

    if image is None:
        return ""
    packed = getattr(image, "packed_file", None)
    if packed is None:
        return _resolved_disk_path(image)
    data = getattr(packed, "data", None)
    if not data:
        return ""
    try:
        payload = bytes(data)
    except Exception:
        return ""
    digest = sha256(payload).hexdigest()[:16]
    suffix = Path(str(getattr(image, "filepath", "") or "")).suffix or ".png"
    name = _SAFE_NAME.sub("_", str(getattr(image, "name", "") or "image"))[:64]
    directory = texture_cache_directory()
    target = directory / f"{name}-{digest}{suffix}"
    if not target.is_file():
        try:
            directory.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(target)
        except OSError:
            return ""
    return str(target)


def _resolved_disk_path(image: Any) -> str:
    filepath = str(getattr(image, "filepath", "") or "")
    if not filepath:
        return ""
    resolved = filepath
    try:
        import bpy  # type: ignore

        resolved = bpy.path.abspath(filepath, library=getattr(image, "library", None))
    except Exception:
        pass
    try:
        path = Path(resolved).expanduser()
        return str(path.resolve()) if path.is_file() else ""
    except OSError:
        return ""


__all__ = ["materialized_image_path", "texture_cache_directory"]
