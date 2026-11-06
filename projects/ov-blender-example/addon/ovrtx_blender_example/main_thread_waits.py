# SPDX-License-Identifier: Apache-2.0
"""Worker-thread waits on Blender's main thread that a stopping owner can cancel.

A thread blocked on a main-thread call cannot be joined from the main thread:
the queued call never runs while the main thread is in ``join``. Owners
cancel the waits of the thread they are about to join; the cancellation is
sticky, so a wait the thread starts afterwards fails at once.
"""
import threading
import weakref

_lock = threading.Lock()
_waits = {}  # thread -> set of pending waits
_cancelled = weakref.WeakKeyDictionary()  # thread -> reason


class MainThreadWaitCancelled(RuntimeError):
    """The owner of the waiting thread is stopping it."""


class Wait:
    def __init__(self):
        self.done = threading.Event()
        self.outcome = {}
        self.cancelled = False
        self.claimed = False

    def claim(self):
        """Main thread: True if the call should still run."""
        with _lock:
            if self.cancelled:
                return False
            self.claimed = True
            return True

    def cancel(self, reason):
        """Cancel this queued call without poisoning later calls on the thread.

        A running Blender callback must finish before its caller tears down
        the depsgraph. The caller discards that result after completion.
        """
        with _lock:
            if self.claimed or self.done.is_set():
                return False
            self.cancelled = True
            self.outcome.setdefault("error", MainThreadWaitCancelled(reason))
            self.done.set()
            return True

    def __enter__(self):
        with _lock:
            reason = _cancelled.get(threading.current_thread())
            if reason is not None:
                raise MainThreadWaitCancelled(reason)
            _waits.setdefault(threading.current_thread(), set()).add(self)
        return self

    def __exit__(self, *_exc):
        self.cancel("Main-thread call was abandoned")
        with _lock:
            pending = _waits.get(threading.current_thread())
            if pending is not None:
                pending.discard(self)
                if not pending:
                    del _waits[threading.current_thread()]


def cancel_waits_of(thread, reason):
    """Release every main-thread wait of ``thread``, now and later, with a cancellation error."""
    with _lock:
        _cancelled[thread] = reason
        pending = tuple(_waits.get(thread, ()))
        for wait in pending:
            if not wait.done.is_set():
                wait.cancelled = True
                wait.outcome.setdefault("error", MainThreadWaitCancelled(reason))
    for wait in pending:
        wait.done.set()
    return len(pending)
