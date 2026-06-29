# SPDX-License-Identifier: Apache-2.0
"""
Pure-Python stand-in for Pixar's `_usdviewq` C++ extension.

The original is built into the OpenUSD distribution as a pybind11 module
that exposes `Utils` (small free-function helpers) plus the Hydra-only
HydraObserver/DataSourceLocator/DebugLocator types. We don't ship Hydra,
so we replace it with a minimal Python `Utils` covering the calls that
appController.py and primViewItem.py make.

Import path stays identical: `from ._usdviewq import Utils, …` so the
vendored usdview source needs no edits.
"""
from __future__ import annotations


class Utils:
    """Subset of Pixar's _usdviewq.Utils used by usdview's data model."""

    @staticmethod
    def GetPrimInfo(prim, time):
        """Return Pixar-style 14-tuple: primViewItem._extractInfo unpacks
        (hasArcs, active, imageable, defined, abstract, isInPrototype,
         isInstance, supportsGuides, supportsDrawMode,
         isVisibilityInherited, visVaries, name, typeName, displayName).
        """
        try:
            type_name = prim.GetTypeName() or ""
        except Exception:
            type_name = ""
        try:
            name = prim.GetName() or ""
        except Exception:
            name = ""
        # Pseudo-root has empty name; usdview displays it as "/".
        if not name:
            try:
                if hasattr(prim, "IsPseudoRoot") and prim.IsPseudoRoot():
                    name = "/"
                else:
                    p = prim.GetPath()
                    if str(p) == "/":
                        name = "/"
            except Exception:
                pass
        active = True
        try:
            if hasattr(prim, "IsActive"):
                active = prim.IsActive()
        except Exception:
            pass
        imageable = bool(type_name)
        is_in_prototype = bool(getattr(prim, "IsInPrototype", lambda: False)())
        is_instance = bool(getattr(prim, "IsInstance", lambda: False)())
        return (
            False,        # hasArcs
            active,       # active
            imageable,    # imageable
            True,         # defined
            False,        # abstract
            is_in_prototype,
            is_instance,
            False,        # supportsGuides
            False,        # supportsDrawMode
            True,         # isVisibilityInherited
            False,        # visVaries
            name,
            type_name,
            "",           # displayName
        )

    @staticmethod
    def _GetAllPrimsOfType(stage, type_name):
        """Walk the stage and return prims whose `IsA(type_name)` is True."""
        try:
            type_str = type_name if isinstance(type_name, str) \
                else type_name.GetTypeName()
        except Exception:
            type_str = str(type_name)
        out = []
        for prim in stage.Traverse():
            try:
                if hasattr(prim, "IsA") and prim.IsA(type_str):
                    out.append(prim)
                elif prim.GetTypeName() == type_str:
                    out.append(prim)
            except Exception:
                continue
        return out


# HydraObserver and friends are Hydra-only — never importable here.
# Code that tries `from ._usdviewq import HydraObserver` will get an
# ImportError, which Phase 3 will route around by deleting that import
# (in hydraSceneDebugger.py).
