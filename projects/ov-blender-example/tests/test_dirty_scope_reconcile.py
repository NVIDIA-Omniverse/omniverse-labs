# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dirty-ID reconcile scoping (ADR 0014 targeted dirty-ID conversion).

A settings-slider drag reports only the Scene ID; a heavy scene must not
pay a full reconversion per depsgraph event (Junk Shop: ~9 s per no-op
reconcile before scoping, 2026-07-07)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import engine  # noqa: E402


def _update(id_object):
    return SimpleNamespace(id=id_object)


class _Id:
    def __init__(self, name: str) -> None:
        self.name_full = name
        self.name = name
        self.original = self


class _SceneId(_Id):
    pass


class _ObjectId(_Id):
    pass


def test_descriptors_report_updated_id_types_and_names() -> None:
    depsgraph = SimpleNamespace(
        updates=(
            _update(_SceneId("Scene")),
            _update(_ObjectId("Suzanne")),
        )
    )
    assert engine._depsgraph_updated_id_descriptors(depsgraph) == {
        ("_SceneId", "Scene"),
        ("_ObjectId", "Suzanne"),
    }


def test_unreadable_updates_widen_to_everything_dirty() -> None:
    assert (
        engine._depsgraph_updated_id_descriptors(
            SimpleNamespace(updates=(_update(None),))
        )
        is None
    )

    class _NamelessId:
        original = None
        name_full = ""
        name = ""

    assert (
        engine._depsgraph_updated_id_descriptors(
            SimpleNamespace(updates=(_update(_NamelessId()),))
        )
        is None
    )


def test_empty_update_list_scopes_to_nothing() -> None:
    # An empty (but readable) update list dirties no conversion sources;
    # reachability reconciliation still owns removals regardless of scope.
    assert engine._depsgraph_updated_id_descriptors(SimpleNamespace(updates=())) == set()
