# SPDX-License-Identifier: Apache-2.0
"""Read ``depsgraph.updates`` once per depsgraph_update_post invocation.

Blender rebuilds the ``Depsgraph.updates`` collection on every attribute
access, at a cost that grows with the scene (about 0.4 ms per read at 10k
objects). The live edit handler snapshots it once and passes the tuple to
each callee that iterates it, next to the real depsgraph: callees still need
the real ``bpy.types.Depsgraph`` for ``evaluated_get``, so a stand-in
depsgraph cannot carry the snapshot.

Callees accept ``updates=None``, meaning "read the depsgraph", so existing
callers keep their behaviour. Collaborators that may be replaced (module
seams, injected builders, engine resolvers) receive the snapshot only when
their signature accepts an ``updates`` keyword.
"""

from __future__ import annotations

import inspect
import weakref
from typing import Any, Callable

_ACCEPTS_UPDATES: "weakref.WeakKeyDictionary[Any, bool]" = weakref.WeakKeyDictionary()


def snapshot(depsgraph: Any) -> tuple[Any, ...]:
    """The handler's single read of ``depsgraph.updates``, in Blender's order."""

    return tuple(getattr(depsgraph, "updates", ()))


def accepts_updates(fn: Callable[..., Any]) -> bool:
    """Whether ``fn`` can take an ``updates=`` keyword argument."""

    target = getattr(fn, "__func__", fn)
    try:
        return _ACCEPTS_UPDATES[target]
    except (KeyError, TypeError):
        pass
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    accepted = any(
        parameter.kind == parameter.VAR_KEYWORD
        or (parameter.name == "updates"
            and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY))
        for parameter in parameters
    )
    try:
        _ACCEPTS_UPDATES[target] = accepted
    except TypeError:
        pass
    return accepted


def keyword(fn: Callable[..., Any], updates: tuple[Any, ...] | None) -> dict[str, Any]:
    """``{"updates": updates}`` when there is a snapshot and ``fn`` takes it."""

    if updates is None or not accepts_updates(fn):
        return {}
    return {"updates": updates}
