# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NanousdStageAdapter — read-only StageAdapter backed by libnanousdapi.so.

Implements every abstract method from :class:`ovgear.adapters.StageAdapter`
using a nanousd-loaded USD stage. Edits (rename, reparent, visibility
toggle) are stubbed since the C API does support them but threading the
edits through ovgear's undo system is out of scope for the read-only
first cut. The Stage Browser still works — it just can't author.
"""

from __future__ import annotations

import re
import weakref
from contextlib import contextmanager
from typing import Callable, Iterable, List, Optional

from ovgear.adapters import (
    AdapterItem,
    BadgeFlags,
    ChangeEvent,
    ChangeEventType,
    ContextManager,
    ItemFlags,
    ReparentPosition,
    StageAdapter,
    VisibilityState,
)
from ovgear.settings import Subscription

from nusd_gles._nanousd import NanousdStage, _Prim


_NAME_RE = re.compile(r"[^A-Za-z0-9_]")

# Map nanousd typeName → ovgear type-category bucket. Anything not listed
# falls through to "Other" which becomes the generic prim icon.
_TYPE_CATEGORY_MAP = {
    "Mesh": "Mesh", "Sphere": "Mesh", "Cube": "Mesh", "Cone": "Mesh",
    "Cylinder": "Mesh", "Capsule": "Mesh", "Plane": "Mesh",
    "BasisCurves": "Mesh", "Points": "Mesh",
    "NurbsCurves": "Mesh", "NurbsPatch": "Mesh",
    "Light": "Light", "DomeLight": "Light", "DistantLight": "Light",
    "DiskLight": "Light", "RectLight": "Light", "SphereLight": "Light",
    "CylinderLight": "Light",
    "Camera": "Camera",
    "Xform": "Xform",
    "Scope": "Scope",
}


def _icon_for_type(t: str) -> str:
    return _TYPE_CATEGORY_MAP.get(t, "Prim")


class _PseudoRoot(_Prim):
    """Synthetic ``/`` root for stages with multiple top-level prims.

    Subclasses :class:`_Prim` so all the typical adapter accessors keep
    working without isinstance branches. Its handle is None, its path is
    ``/``, and its children are produced by the owning stage.
    """

    __slots__ = ("_stage",)

    def __init__(self, stage: NanousdStage) -> None:
        super().__init__(handle=0, path="/")
        self._stage = stage

    @property
    def name(self) -> str:  # type: ignore[override]
        return "/"

    @property
    def type_name(self) -> str:  # type: ignore[override]
        return ""

    @property
    def is_active(self) -> bool:  # type: ignore[override]
        return True

    def children(self) -> Iterable[_Prim]:  # type: ignore[override]
        return list(self._stage.root_children())

    def n_children(self) -> int:  # type: ignore[override]
        return len(list(self._stage.root_children()))

    def __repr__(self) -> str:
        return "<NanousdPseudoRoot />"


class NanousdStageAdapter(StageAdapter):
    """Read-only StageAdapter backed by a :class:`NanousdStage`."""

    def __init__(self, stage: NanousdStage) -> None:
        self._stage = stage
        self._root = _PseudoRoot(stage)
        self._subscribers: List[Callable[[ChangeEvent], None]] = []
        self._suppress = False
        # Cache the default-prim path so get_item_flags doesn't re-walk
        # the C side on every redraw.
        try:
            dp = stage.default_prim()
            self._default_prim_path = dp.path if dp else ""
        except Exception:
            self._default_prim_path = ""

    # Expose the stage so launcher code can pull out the file path / etc.
    @property
    def stage(self) -> NanousdStage:
        return self._stage

    # ── Hierarchy ─────────────────────────────────────────────────────────

    def get_root(self) -> AdapterItem:
        return self._root

    def get_children(self, item: AdapterItem) -> List[_Prim]:
        if isinstance(item, _PseudoRoot):
            return list(self._stage.root_children())
        if isinstance(item, _Prim):
            return list(item.children())
        return []

    def can_have_children(self, item: AdapterItem) -> bool:
        # Any prim could in principle have children — the read-only
        # adapter doesn't distinguish.
        return True

    def get_item_path(self, item: AdapterItem) -> str:
        if isinstance(item, _Prim):
            return item.path
        return "/"

    def get_item_at_path(self, path: str) -> Optional[_Prim]:
        if path in ("", "/"):
            return self._root
        return self._stage.prim_at_path(path)

    # ── Display ───────────────────────────────────────────────────────────

    def get_display_name(self, item: AdapterItem) -> str:
        if isinstance(item, _PseudoRoot):
            # Stage browser shows a single header row — use the file name.
            from pathlib import Path
            try:
                return Path(self._stage.filepath).name or "/"
            except Exception:
                return "/"
        if isinstance(item, _Prim):
            return item.name or item.path
        return ""

    def get_type_name(self, item: AdapterItem) -> str:
        if isinstance(item, _PseudoRoot):
            return ""
        if isinstance(item, _Prim):
            return item.type_name
        return ""

    def get_type_category(self, item: AdapterItem) -> str:
        return _TYPE_CATEGORY_MAP.get(self.get_type_name(item), "Other")

    def get_icon_name(self, item: AdapterItem) -> str:
        return _icon_for_type(self.get_type_name(item))

    def get_badge_flags(self, item: AdapterItem) -> BadgeFlags:
        if not isinstance(item, _Prim) or isinstance(item, _PseudoRoot):
            return BadgeFlags.NONE
        flags = BadgeFlags.NONE
        try:
            if item.is_instanceable:
                flags |= BadgeFlags.INSTANCE
            # nanousd_prim_listop reports authored composition arcs.
            if item.has_listop("references"):
                flags |= BadgeFlags.REFERENCE
            if item.has_listop("payloads"):
                flags |= BadgeFlags.PAYLOAD
            if item.has_listop("inheritPaths"):
                flags |= BadgeFlags.INHERITS
            if item.has_listop("specializes"):
                flags |= BadgeFlags.SPECIALIZES
        except Exception:
            pass
        return flags

    def get_item_flags(self, item: AdapterItem) -> ItemFlags:
        if not isinstance(item, _Prim) or isinstance(item, _PseudoRoot):
            return ItemFlags.NONE
        flags = ItemFlags.NONE
        try:
            if item.is_abstract:
                # USD class prims surface as abstract; expose both flags
                # so the Type column highlight matches the reference USD
                # adapter behaviour.
                flags |= ItemFlags.IS_ABSTRACT | ItemFlags.IS_CLASS
            if not item.is_active:
                flags |= ItemFlags.IS_INACTIVE
            if self._default_prim_path and item.path == self._default_prim_path:
                flags |= ItemFlags.IS_DEFAULT_PRIM
        except Exception:
            pass
        return flags

    # ── Visibility ────────────────────────────────────────────────────────

    def compute_visibility(self, item: AdapterItem) -> VisibilityState:
        if not isinstance(item, _Prim) or isinstance(item, _PseudoRoot):
            return VisibilityState.VISIBLE
        # USD's visibility composes top-down: an ancestor's "invisible"
        # token hides every descendant regardless of their authored
        # opinions. Walk up parent paths reading the visibility token
        # until we hit the pseudo-root.
        try:
            tok = item.get_attrib_str("visibility", "inherited")
        except Exception:
            tok = "inherited"
        if tok == "invisible":
            return VisibilityState.INVISIBLE
        # Climb ancestors via path-string slicing — cheaper than
        # nanousd_parent + freeprim per hop.
        path = item.path
        while True:
            slash = path.rfind("/")
            if slash <= 0:
                break
            path = path[:slash]
            parent = self._stage.prim_at_path(path)
            if parent is None:
                break
            try:
                ptok = parent.get_attrib_str("visibility", "inherited")
            except Exception:
                ptok = "inherited"
            if ptok == "invisible":
                return VisibilityState.INHERITED_INVISIBLE
        return VisibilityState.VISIBLE

    def set_visibility(self, item: AdapterItem, visible: bool) -> None:
        if not isinstance(item, _Prim) or isinstance(item, _PseudoRoot):
            return
        token = "inherited" if visible else "invisible"
        try:
            item.set_attrib_token("visibility", token)
        except Exception:
            return
        self._notify(ChangeEvent(
            changed_paths=(item.path,),
            resynced_paths=(),
            event_type=ChangeEventType.INFO_CHANGE,
        ))

    def can_edit_visibility(self, item: AdapterItem) -> bool:
        if not isinstance(item, _Prim) or isinstance(item, _PseudoRoot):
            return False
        return True

    # ── Rename / reparent — disabled in read-only adapter ─────────────────

    def can_rename(self, item: AdapterItem) -> bool:
        return False

    def rename(self, item: AdapterItem, new_name: str) -> str:
        # Returning the unchanged name keeps the rename controller happy.
        if isinstance(item, _Prim):
            return item.name
        return new_name

    def normalize_name(self, name: str) -> str:
        return _NAME_RE.sub("_", name)

    def can_reparent(self, items: List[AdapterItem], new_parent: AdapterItem) -> bool:
        return False

    def reparent(
        self,
        items: List[AdapterItem],
        new_parent: AdapterItem,
        position: ReparentPosition,
    ) -> None:
        return

    # ── Filter / change subscription ──────────────────────────────────────

    def filter_items(
        self,
        items: List[AdapterItem],
        predicate: Callable[[AdapterItem], bool],
    ) -> List[AdapterItem]:
        return [item for item in items if predicate(item)]

    def subscribe_changes(
        self, callback: Callable[[ChangeEvent], None]
    ) -> Subscription:
        self._subscribers.append(callback)
        return Subscription(weakref.ref(self), "changes", callback)

    def _remove_subscriber(self, key: str, callback: Callable) -> None:
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    def _notify(self, event: ChangeEvent) -> None:
        if self._suppress:
            return
        for cb in list(self._subscribers):
            try:
                cb(event)
            except Exception:
                pass

    # ── Undo / suppression — no-ops ───────────────────────────────────────

    def begin_undo_group(self, label: str) -> None:
        return

    def end_undo_group(self) -> None:
        return

    @contextmanager
    def suppress_change_notifications(self) -> ContextManager:
        old = self._suppress
        self._suppress = True
        try:
            yield
        finally:
            self._suppress = old
