# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Central user-facing message reporting for the OVRTX add-on.

Every user-facing status/error the add-on produces must reach three sinks so
nothing is overlay-only:

* stdout/stderr, immediately (stdout for INFO/status, stderr for
  WARNING/ERROR), each line carrying an ``[ovrtx]`` prefix, and
* the Blender Info editor log.

``RenderEngine.report()`` does not populate the Info window from viewport-draw
or render-thread contexts, and Info-window writes must happen on the main
thread. The reliable pattern is a small operator (:class:`OVRTX_OT_report`)
whose ``execute`` calls ``self.report(level, message)``, invoked from a
``bpy.app.timers`` main-thread pump that drains a thread-safe queue. Errors
raised on render/worker threads are enqueued here, never dropped.

The core (:class:`UserMessageBus`) is bpy-free and unit-testable. The Blender
wiring (operator + timer pump) lives behind :func:`register`/:func:`unregister`
and no-ops when ``bpy`` is unavailable. ``update_stats``/viewport-overlay
behaviour is untouched: callers keep drawing the overlay and additionally emit
through this module.
"""

from __future__ import annotations

import sys
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

try:
    import bpy  # type: ignore
except ModuleNotFoundError:
    bpy = None  # type: ignore[assignment]


INFO = "INFO"
WARNING = "WARNING"
ERROR = "ERROR"

#: Blender report levels, ordered most severe first (used to fold a report-style
#: level set such as ``{'ERROR'}`` down to a single channel level).
_LEVELS: tuple[str, ...] = (ERROR, WARNING, INFO)

#: Prefix stamped on every console line.
PREFIX = "[ovrtx]"

#: Main-thread pump cadence, in seconds.
_PUMP_INTERVAL_S = 0.1


@dataclass(frozen=True)
class UserMessage:
    """A single user-facing message queued for the Info window."""

    level: str
    text: str
    context: str = ""


def normalize_level(level: Any) -> str:
    """Fold any accepted level form to one of ``INFO``/``WARNING``/``ERROR``.

    Accepts a plain string (case-insensitive) or a Blender report-style set of
    strings (e.g. ``{'ERROR'}`` or ``{'WARNING', 'INFO'}``), for which the most
    severe recognised level wins. Anything unrecognised degrades to ``INFO``.
    """

    if isinstance(level, str):
        candidate = level.strip().upper()
        return candidate if candidate in _LEVELS else INFO
    if isinstance(level, (set, frozenset, tuple, list)):
        found = {str(item).strip().upper() for item in level}
        for severity in _LEVELS:
            if severity in found:
                return severity
        return INFO
    return INFO


def stream_for_level(level: str, *, stdout: Any = None, stderr: Any = None) -> Any:
    """Return the console stream a level writes to (stdout for INFO)."""

    if level == INFO:
        return stdout if stdout is not None else sys.stdout
    return stderr if stderr is not None else sys.stderr


class UserMessageBus:
    """Thread-safe fan-out of user messages to console + an Info-window queue.

    Console writes happen immediately (on whatever thread emitted). Info-window
    delivery is deferred: messages accumulate in a thread-safe queue that a
    main-thread pump drains (see :func:`register`). Change-only de-duplication
    keeps a per-``(context, channel)`` record of the last emitted text so a
    status string repeated every draw tick is written exactly once until it
    changes.
    """

    def __init__(self, *, stdout: Any = None, stderr: Any = None) -> None:
        self._lock = threading.Lock()
        self._queue: deque[UserMessage] = deque()
        self._last_console: dict[str, tuple[str, str]] = {}
        self._last_info: dict[str, tuple[str, str]] = {}
        self._stdout = stdout
        self._stderr = stderr

    def emit(
        self,
        level: Any,
        text: Any,
        *,
        context: str = "",
        dedup: bool = True,
        to_console: bool = True,
        to_info: bool = True,
    ) -> bool:
        """Fan a message out to the console and the Info-window queue.

        ``level`` may be a string or a Blender report-style set. ``context`` is
        a dedup bucket (typically per engine + purpose): when ``dedup`` is true,
        a channel is written only if this message differs from the last one it
        received for that context. Returns ``True`` if anything was emitted.
        """

        level = normalize_level(level)
        text = "" if text is None else str(text)
        if not text:
            return False
        context = str(context)
        value = (level, text)

        with self._lock:
            write_console = to_console and not (
                dedup and self._last_console.get(context) == value
            )
            write_info = to_info and not (
                dedup and self._last_info.get(context) == value
            )
            if write_console:
                self._last_console[context] = value
            if write_info:
                self._last_info[context] = value
                self._queue.append(UserMessage(level=level, text=text, context=context))

        if write_console:
            self._write_console(level, text)
        return write_console or write_info

    def _write_console(self, level: str, text: str) -> None:
        stream = stream_for_level(level, stdout=self._stdout, stderr=self._stderr)
        try:
            stream.write(f"{PREFIX} {text}\n")
            flush = getattr(stream, "flush", None)
            if callable(flush):
                flush()
        except Exception:
            # The console must never take down a render/draw path.
            pass

    def take_pending(self) -> list[UserMessage]:
        """Atomically remove and return every queued Info-window message."""

        with self._lock:
            pending = list(self._queue)
            self._queue.clear()
        return pending

    def drain(self, sink: Callable[[str, str], Any]) -> int:
        """Drain the Info-window queue, handing each message to ``sink``.

        ``sink`` is called as ``sink(level, text)`` on the draining (main)
        thread. A sink exception never strands the remaining messages.
        """

        count = 0
        for message in self.take_pending():
            try:
                sink(message.level, message.text)
            except Exception:
                pass
            count += 1
        return count

    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    def reset(self) -> None:
        """Clear the queue and dedup state (tests / add-on reload)."""

        with self._lock:
            self._queue.clear()
            self._last_console.clear()
            self._last_info.clear()


# --- Process-wide default bus + convenience API ---------------------------

_DEFAULT_BUS = UserMessageBus()


def default_bus() -> UserMessageBus:
    return _DEFAULT_BUS


def report(
    level: Any,
    text: Any,
    *,
    context: str = "",
    dedup: bool = True,
    to_console: bool = True,
    to_info: bool = True,
) -> bool:
    """Emit a message through the process-wide bus (see :meth:`UserMessageBus.emit`)."""

    return _DEFAULT_BUS.emit(
        level,
        text,
        context=context,
        dedup=dedup,
        to_console=to_console,
        to_info=to_info,
    )


def report_info(text: Any, *, context: str = "", dedup: bool = True, to_info: bool = True) -> bool:
    return report(INFO, text, context=context, dedup=dedup, to_info=to_info)


def report_warning(text: Any, *, context: str = "", dedup: bool = True, to_info: bool = True) -> bool:
    return report(WARNING, text, context=context, dedup=dedup, to_info=to_info)


def report_error(text: Any, *, context: str = "", dedup: bool = True, to_info: bool = True) -> bool:
    return report(ERROR, text, context=context, dedup=dedup, to_info=to_info)


def mirror_to_console(level: Any, text: Any, *, context: str = "", dedup: bool = False) -> bool:
    """Mirror a message to the console only, leaving Info-window delivery alone.

    Used where ``self.report(...)`` already reaches the Info window natively
    (final-render context) so the console gets the message without double-posting
    to Info.
    """

    return report(level, text, context=context, dedup=dedup, to_info=False)


def report_for_operator(operator: Any, level: Any, text: Any) -> None:
    """``operator.report`` plus the console mirror, in one call.

    Operator ``execute`` runs on the main thread, so ``operator.report``
    reaches the Info window natively — only the console needs mirroring
    (stdout for INFO, stderr for WARNING/ERROR), never a second Info post.
    Operator invocations are discrete user-triggered events: repeats are
    deliberate and are not de-duplicated. Every operator report should go
    through this helper so no message stays Info-only.
    """

    normalized = normalize_level(level)
    levels = level if isinstance(level, (set, frozenset)) else {normalized}
    try:
        operator.report(set(levels), "" if text is None else str(text))
    except Exception:
        # The console mirror below still runs; a report failure (e.g. a
        # stale operator wrapper) must not swallow the message entirely.
        pass
    mirror_to_console(normalized, text, dedup=False)


# --- Blender wiring: report operator + main-thread pump -------------------

#: Only a real ``bpy`` supplies the operator base class and property
#: descriptors. Tests that inject a partial fake ``bpy`` (no ``types.Operator``)
#: still exercise the bpy-free bus; the operator simply stays undefined there.
_HAS_BPY_OPERATOR = (
    bpy is not None
    and getattr(getattr(bpy, "types", None), "Operator", None) is not None
    and getattr(bpy, "props", None) is not None
)


if _HAS_BPY_OPERATOR:

    class OVRTX_OT_report(bpy.types.Operator):  # type: ignore[misc]
        """Post a single message into the Blender Info editor log.

        Invoked from the main-thread pump so render/worker-thread messages reach
        the Info window (which ``RenderEngine.report`` cannot from those
        contexts).
        """

        bl_idname = "ovrtx_example.report"
        bl_label = "OVRTX Report"
        bl_options = {"INTERNAL"}

        level: bpy.props.StringProperty(default=INFO)  # type: ignore[valid-type]
        message: bpy.props.StringProperty(default="")  # type: ignore[valid-type]

        def execute(self, context: Any) -> set[str]:
            level = self.level if self.level in _LEVELS else INFO
            if self.message:
                self.report({level}, self.message)
            return {"FINISHED"}

else:
    OVRTX_OT_report = None  # type: ignore[assignment]


def _post_to_info_window(level: str, text: str) -> None:
    if bpy is None:
        return
    ops = getattr(getattr(bpy, "ops", None), "ovrtx_example", None)
    report_op = getattr(ops, "report", None)
    if report_op is None:
        return
    try:
        report_op(level=level, message=text)
    except Exception:
        # A transient operator-context failure must not strand the queue; the
        # message already reached the console synchronously at emit time.
        pass


def _pump() -> float:
    _DEFAULT_BUS.drain(_post_to_info_window)
    return _PUMP_INTERVAL_S


def register(bpy_module: Any = None) -> bool:
    """Register the report operator and the main-thread Info-window pump."""

    module = bpy_module if bpy_module is not None else bpy
    if module is None:
        return False
    operator = OVRTX_OT_report
    if operator is not None and not getattr(operator, "is_registered", False):
        module.utils.register_class(operator)
    timers = getattr(getattr(module, "app", None), "timers", None)
    if timers is None:
        return False
    try:
        if callable(getattr(timers, "is_registered", None)) and timers.is_registered(_pump):
            return True
        timers.register(_pump, first_interval=_PUMP_INTERVAL_S)
    except Exception:
        return False
    return True


def unregister(bpy_module: Any = None) -> bool:
    """Unregister the pump and report operator; drain any residual messages."""

    module = bpy_module if bpy_module is not None else bpy
    if module is None:
        return False
    timers = getattr(getattr(module, "app", None), "timers", None)
    if timers is not None:
        try:
            if callable(getattr(timers, "is_registered", None)) and timers.is_registered(_pump):
                timers.unregister(_pump)
        except Exception:
            pass
    operator = OVRTX_OT_report
    if operator is not None and getattr(operator, "is_registered", False):
        # Flush anything still queued into the Info window while the operator
        # is still registered (this runs on Blender's main thread), so a
        # teardown-time message is not lost and, critically, is not carried
        # in the queue into a later re-register (extension rebuild) where it
        # would surface belatedly in the next session's Info log.
        _DEFAULT_BUS.drain(_post_to_info_window)
        try:
            module.utils.unregister_class(operator)
        except (RuntimeError, ValueError):
            pass
    else:
        # No operator to deliver through: discard residual messages rather
        # than let the queue grow unbounded across register/unregister cycles.
        _DEFAULT_BUS.take_pending()
    return True


__all__ = [
    "INFO",
    "WARNING",
    "ERROR",
    "PREFIX",
    "UserMessage",
    "UserMessageBus",
    "OVRTX_OT_report",
    "default_bus",
    "normalize_level",
    "stream_for_level",
    "report",
    "report_info",
    "report_warning",
    "report_error",
    "mirror_to_console",
    "report_for_operator",
    "register",
    "unregister",
]
