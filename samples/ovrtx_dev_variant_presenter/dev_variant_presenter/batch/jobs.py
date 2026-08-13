# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch job + matrix-mode models and the folder-safe {set}-{variant} naming.

Pure data + naming; no ovrtx/pxr. The naming convention is shared with the post-
processing module, so its parsers can split on `_` and `-` unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from dev_variant_presenter.models import QualitySpec, Selection, VariantChoice  # noqa: F401 (re-exported shape)


class MatrixMode(str, Enum):
    FULL_CARTESIAN = "full_cartesian"
    ONE_AT_A_TIME = "one_at_a_time"
    CURATED = "curated"


@dataclass(frozen=True)
class BatchJob:
    """A grid/matrix batch render request.

    base_selection : the pinned live look — every set NOT swept stays here.
    included       : per-set chosen variants for cherry-pick (set_name -> variants).
                     A whole-set include is just every variant of that set.
    curated        : explicit Selections, only used when mode == CURATED.
    """
    mode: MatrixMode
    base_selection: Selection
    included: dict[str, tuple[str, ...]]
    cameras: list[str]
    quality: QualitySpec
    frame_mode: Literal["single", "animation_range"]
    out_dir: str
    curated: tuple[Selection, ...] = ()
    # animation_range only; None falls back to the stage's start/end. step renders every
    # Nth frame (preview/turntable subsampling).
    frame_start: int | None = None
    frame_end: int | None = None
    frame_step: int = 1


def _folder_safe(text: str) -> str:
    """Make a path/variant token safe for a folder/file name."""
    for ch in ("/", "\\", " ", ":", "*", "?", '"', "<", ">", "|"):
        text = text.replace(ch, "_")
    return text


def permutation_name(selection: Selection) -> str:
    """Folder-safe `{set}-{variant}` name joined by `_`, matching the convention the
    post-processing label parsers expect."""
    if not selection:
        return "default"
    parts = [f"{_folder_safe(c.set_name)}-{_folder_safe(c.variant)}" for c in selection]
    return "_".join(parts)
