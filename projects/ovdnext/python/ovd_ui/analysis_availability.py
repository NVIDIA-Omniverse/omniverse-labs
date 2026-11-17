# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analysis action state from owned presentation facts and a private native probe.

No GUI polling reads the engine. Context changes invalidate pending results;
View-open or a new capture starts at most one check per cached scene/frame.
Messages remain available for empty OVD streams. Legacy presence is not inferred
from file extension, empty current-frame profile scopes, or error record count.
A capped native scan may return unknown, which stays disabled with its reason.
"""
from collections import OrderedDict
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import tempfile

import _pvd_native as native
from .ingestion import retain_capture, submit_capture

ACTIONS = ("profile", "memory", "errors", "frame_timing", "messages")


@dataclass(frozen=True)
class ActionAvailability:
    enabled: bool
    reason: str = ""
    checking: bool = False
    requestable: bool = False


@dataclass(frozen=True)
class AvailabilityContext:
    capture_key: int
    segment_epoch: int
    path: Path | None
    scene_id: int
    frame: int
    ovd: bool
    active: bool
    clip_frames: int

    @property
    def key(self):
        return (self.capture_key, self.segment_epoch, self.path, self.scene_id, self.frame)

    @property
    def capture_identity(self):
        return self.capture_key, self.segment_epoch, self.path

    @classmethod
    def from_display(cls, path, presentation, clip):
        if presentation is None:
            return None
        info = presentation.info()
        return cls(presentation.capture_key, presentation.segment_epoch,
                   Path(path) if path is not None else None,
                   presentation.scene_id, presentation.frame, info.ovd,
                   bool(info.live and not info.ended), clip.stop-clip.start if clip else 0)


def read_availability(context, control):
    """Private native load/seek; neither displayed Session nor capture is mutated.

    LoadControl also cancels native opening at its documented safe points.
    Legacy indexed opening/indexing remains non-interruptible inside that stage.
    """
    path = context.path
    if path is None or not path.is_file():
        raise ValueError("Analysis requires an existing saved capture")
    with retain_capture(path), tempfile.TemporaryDirectory(prefix="pvd-availability-") as temporary:
        before = path.stat()
        session = native.Session(str(path), temporary, control)
        try:
            if control.cancelled:
                raise CancelledError()
            if not session.seek(context.scene_id, context.frame):
                raise ValueError("The requested analysis frame is not published")
            result = native.analysis_availability(session, context.scene_id, control)
            after = path.stat()
            fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
            if fields(before) != fields(after):
                raise ValueError("Capture changed during the analysis availability check; reopen it")
            return result
        finally:
            del session


class AnalysisAvailability:
    """UI-thread controller. sync/poll never wait for a worker; close joins it.

    Native data and errors are cached for up to 32 frame keys of one capture.
    Revisiting an uncached frame requires an explicit request (View or action).
    Errors/budget uncertainty are cached too, avoiding retry loops on menu-open.
    Owned whole-file zone/timestamp/nonzero memory-metadata facts are reusable
    across frames; error logs and zero-metadata memory events are not.
    Worker completions are accepted only for the exact current capture/epoch/frame.
    """
    def __init__(self, *, executor=None, reader=read_availability, cache_size=32):
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="pvd-availability")
        self.reader, self.cache_size = reader, cache_size
        self.context = None
        self.cache = OrderedDict()
        self.capture_facts = None
        self.pending = None
        self.closed = False

    def _cancel(self):
        if self.pending is not None:
            _, control, future = self.pending
            control.cancel()
            future.cancel()
            self.pending = None

    def sync(self, context):
        if context == self.context:
            return False
        previous = self.context
        self.context = context
        if previous is None or context is None or previous.capture_identity != context.capture_identity:
            self.cache.clear()
            self.capture_facts = None
        if self.pending and (context is None or self.pending[0] != context.key):
            self._cancel()
        return True

    def request(self):
        context = self.context
        if self.closed or context is None or context.ovd or context.active or context.path is None or not context.scene_id:
            return
        if context.key in self.cache or self.pending is not None:
            return
        control = native.LoadControl()
        try:
            future = submit_capture(self.executor, context.path, self.reader, context, control)
        except Exception as error:
            self._store(context.key, "Analysis availability check failed: " + str(error))
            return
        self.pending = context.key, control, future

    def poll(self):
        if self.pending is None or not self.pending[2].done():
            return False
        key, control, future = self.pending
        self.pending = None
        if self.closed or self.context is None or key != self.context.key or control.cancelled:
            return False
        try:
            result = future.result()
        except CancelledError:
            return False
        except Exception as error:
            result = "Analysis availability check failed: " + str(error)
        self._store(key, result)
        return True

    def _store(self, key, result):
        self.cache[key] = result
        if not isinstance(result,str):self.capture_facts = result
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def state(self, action):
        if action not in ACTIONS:
            raise ValueError("Unknown analysis action")
        context = self.context
        if context is None:
            return ActionAvailability(False, "Open a capture to inspect analysis data")
        if action == "messages":
            return ActionAvailability(context.ovd, "" if context.ovd else "Legacy captures use Errors; Capture messages requires OVD")
        if context.ovd:
            return ActionAvailability(False, {
                "profile": "OVD does not contain legacy profile event streams",
                "memory": "OVD does not contain legacy memory event streams",
                "errors": "OVD uses Capture messages; it has no legacy Error Stream",
                "frame_timing": "OVD does not record legacy frame timing measurements",
            }[action])
        if context.active:
            return ActionAvailability(False, "Detach and save the producer before inspecting analysis data")
        if context.path is None:
            return ActionAvailability(False, "Analysis requires a saved capture")
        if not context.scene_id:
            return ActionAvailability(False, "This capture has no selected scene timeline")
        if action == "frame_timing" and context.clip_frames < 2:
            return ActionAvailability(False, "Frame graph requires at least two frames in the current clip")
        result = self.cache.get(context.key)
        if result is None and self.capture_facts is not None:
            # These owned facts originate in whole-file stream metadata. Error
            # logs and zero-metadata memory activity remain frame-specific.
            if action in ("profile","frame_timing") or (action=="memory" and self.capture_facts.memory_entries):
                result = self.capture_facts
        if result is None:
            checking = self.pending is not None and self.pending[0] == context.key
            return ActionAvailability(False, "Checking recorded analysis data…" if checking else
                "Recorded data for this frame has not been checked yet", checking, requestable=True)
        if isinstance(result, str):
            return ActionAvailability(False, result)
        presence = getattr(result, action)
        return ActionAvailability(bool(presence.known and presence.available), presence.reason)

    def close(self):
        self.closed = True
        self._cancel()
        self.context = None
        self.cache.clear()
        self.capture_facts = None
        self.executor.shutdown(wait=True, cancel_futures=True)
