# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Toolkit-independent capture lifecycle over the validated C++ Session engine.

Call mutations and ``poll`` on the application thread. Native ingest runs on its
own worker. ``adopt_file`` accepts a Session created by an asynchronous loader;
failed/cancelled loads must never be adopted. No binary parsing occurs here.

``poll()`` drains immutable WorkflowEvents queued by commands and live updates:
* ``display`` replaces/clears the displayed session. The frontend resets its
  selection/adapter/clip state and loads event.scenes. session=None means New.
* ``frames`` refreshes the displayed live session's models. scenes, serial,
  segment_epoch, current_frame and max_frame are one native publication snapshot.
  Seek only to the corresponding completed frame (respecting the frontend's
  active scene and Pre/Post mode). segment_changed invalidates geometry caches.
* ``recordings`` and ``recents`` rebuild the history list; ``state`` updates capture
  controls and status; ``error`` displays a recoverable error. Other fields have
  neutral defaults.

Opening history leaves the native listener running. Follow latest promotes only
a useful publication, never a schema fragment or empty/rejected connection. An
accepted producer may finish after acceptance is disabled. Detach/New/shutdown
explicitly stop it. Keep recordings is sampled when each listener is armed,
and updated while that listener has not yet completed a producer handshake.
Every connection is captured in RAM; optional disk saves happen only after a
useful capture completes and never determine whether the listener can start.
"""
from dataclasses import dataclass, replace
import textwrap
from pathlib import Path
import tempfile
import time

from _pvd_native import Session
from .recordings import RecordingLibrary, absolute_path, copy_capture, validate_capture_destination
from .settings import Preferences, parse_port
from .ingestion import load_capture
from .capture_control import CaptureSelectionModel


@dataclass(frozen=True)
class WorkflowEvent:
    kind: str
    session: object = None
    path: Path | None = None
    data_path: Path | None = None
    message: str = ""
    scenes: tuple = ()
    serial: int = 0
    segment_epoch: int = 0
    segment_changed: bool = False
    segment_transition_serial: int = 0
    segment_transition_count: int = 0
    current_frame: int = 0
    max_frame: int = 0


@dataclass(frozen=True)
class SessionRecording:
    """One receiving or retained native session, independent of its file."""
    id: int
    session: object
    recorded_at: float
    status: str
    size_bytes: int
    path: Path | None = None


def wrap_connection_tooltip(text):
    """Bound string-tooltip lines without hiding filenames or listener errors.

    OVUI's public string tooltip constructs an unconstrained Label. Explicit
    newlines retain that API/state contract while keeping the tooltip compact.
    """
    return "\n".join(textwrap.fill(line,width=64,replace_whitespace=False,
                                 break_long_words=True,break_on_hyphens=False)
                     for line in text.split("\n"))


@dataclass(frozen=True)
class ConnectionStatus:
    """Last observed transport state, independent of transient operation messages.

    Waiting means a native listener was requested, not a claim that its
    asynchronous socket bind succeeded. poll() observes rejection and retry.
    No Session or borrowed native metadata is retained by this snapshot.
    """
    state: str
    port: int
    accepting: bool
    background: bool
    error: str = ""

    @property
    def text(self):
        return {"inactive": "Live inactive", "paused": "Live paused", "closed": "Live closed"}.get(
            self.state, f"{self.port} · {self.state.capitalize()}")

    @property
    def tooltip(self):
        description = {
            "inactive": "Live listening has not been started.",
            "waiting": f"Listening requested on TCP port {self.port}.\nWaiting for a producer.",
            "receiving": f"Receiving a producer on TCP port {self.port}.",
            "retrying": f"Waiting to retry listening on TCP port {self.port}.",
            "paused": f"Live listening is paused on TCP port {self.port}.",
            "closed": "Live capture is closed.",
        }[self.state]
        if self.background:
            description += ("\nThe displayed capture is separate;\nlive capture continues in the background."
                            if self.state == "receiving" else
                            "\nThe displayed capture is separate;\nlistening is requested in the background.")
        if not self.accepting and self.state == "receiving":
            description += "\nNew connections are paused;\nthe current producer continues."
        elif self.state not in ("inactive", "closed"):
            description += "\nNew connections are enabled." if self.accepting else "\nNew connections are paused."
        if self.error:
            description += "\nListener error:\n" + self.error
        return wrap_connection_tooltip(description)


class CaptureWorkflow:
    def __init__(self, settings=None, recordings=None, *, session_type=Session,
                 temporary_directory=None, clock=time.monotonic):
        self.settings = settings if settings is not None else Preferences()
        self.recordings = recordings if recordings is not None else RecordingLibrary()
        self._session_type, self._clock = session_type, clock
        self._temporary = None if temporary_directory else tempfile.TemporaryDirectory(prefix="pvd-session-")
        self.temporary_directory = str(temporary_directory or self._temporary.name)
        self.display_session = None
        self.display_path = None
        self.display_data_path = None
        self._loaded = None
        self._session_recordings = []
        self._next_session_recording_id = 1
        self._preview_fallback = ()
        self.live_session = None
        self.capture_selection = CaptureSelectionModel()
        self.port = self.settings.get("live_port", 5425)
        self.accept_connections = True  # Per-run switch; never persist OFF.
        self.follow_live = True
        self._live_poll_holds = set()
        self.status = "Ready"
        self.error = ""
        self.closed = False
        self._events = []
        self._listening_requested = False
        self._retry_at = 0.
        self._auto_save_active = False
        self._recorded_at = 0.
        self._ever_connected = self._useful = False
        self._connection_connected = False
        self._listener_error = ""
        self._live_update = None
        self._serial = self._epoch = self._transition_serial = 0
        if self.settings.load_error:
            self._fail(self.settings.load_error)
        if self.recordings.load_error:
            self._fail(f"Could not load recordings: {self.recordings.load_error}")

    @property
    def keep_recordings(self):
        return self.settings.get("keep_recordings", False)

    @property
    def session_recordings(self):
        return tuple(reversed(self._session_recordings))

    @property
    def live_recording(self):
        if self.live_session is None or not self._ever_connected:
            return None
        return SessionRecording(self._next_session_recording_id, self.live_session,
            self._recorded_at, "Live", self.live_session.info().recording_size)

    @property
    def displayed_session_recording_id(self):
        for recording in self._session_recordings:
            if recording.session is self.display_session:
                return recording.id
        return None

    @property
    def recent_files(self):
        return self.settings.recent_files

    def refresh_recent_files(self):
        """Match Qt's refresh-time pruning without touching capture files."""
        try:
            remaining = [path for path in self.recent_files if Path(path).exists()]
        except OSError as error:
            self._fail(f"Could not refresh recent files: {error}")
            return self.recent_files
        if tuple(remaining) != self.recent_files:
            self._persist("recent_files", remaining)
        return self.recent_files

    def reload_recordings(self):
        try:
            self.recordings.reload()
        except OSError as error:
            return self._fail(f"Could not load recordings: {error}")
        return True

    @property
    def active_recording_path(self):
        # Active captures have no disk path; autosave runs only after completion.
        return None

    @property
    def connection_status(self):
        """Pure snapshot of existing native observations; never polls or seeks."""
        if self.closed:
            state = "closed"
        elif self.live_session is not None:
            state = "receiving" if self._connection_connected else "waiting"
        elif not self.accept_connections:
            state = "paused"
        elif self._listening_requested:
            state = "retrying"
        else:
            state = "inactive"
        background = self.live_session is not None and self.display_session is not None and self.display_session is not self.live_session
        return ConnectionStatus(state, self.port, self.accept_connections, background, self._listener_error)

    @property
    def can_save(self):
        if self.display_session is not None and self.display_session is self.live_session:
            return self._useful
        retained = any(recording.session is self.display_session
                       and recording.status == "Complete"
                       for recording in self._session_recordings)
        return self.display_session is not None and (self.display_path is not None or retained)

    def _emit(self, kind, **values):
        self._events.append(WorkflowEvent(kind, **values))

    def _state(self, message):
        self.status = message
        self._emit("state", message=message)

    def _fail(self, message):
        self.error = str(message)
        self._emit("error", message=self.error)
        self._state(self.error)
        return False

    def _persist(self, key, value):
        try:
            self.settings.set(key, value)
            return True
        except (OSError, ValueError) as error:
            return self._fail(f"Could not save preference: {error}")

    def startup(self, capture=None, endpoint=None, *, explicit_live=False):
        """Explicit live wins over positional file; empty startup listens on 5425/saved port."""
        if explicit_live or endpoint is not None or not capture:
            return self.attach(endpoint if endpoint is not None else self.port)
        return self.open_file(capture)

    def open_file(self, path):
        if self.closed:
            return False
        source = absolute_path(path)
        try:
            directory = self.temporary_directory if self._temporary is None else None
            loaded = load_capture(source, temporary_directory=directory, session_type=self._session_type)
            return self.adopt_loaded(loaded)
        except (RuntimeError, ValueError, OSError) as error:
            return self._fail(f"Unable to open {source.name}: {error}")

    @property
    def verified_payload(self):
        loaded = self._loaded
        if (loaded is not None and self.display_session is not None and loaded.session is self.display_session
                and loaded.data_path == self.display_data_path):
            return loaded.verified_payload
        return None

    def _release_loaded(self):
        if self._loaded:
            self._loaded.close()
            self._loaded = None

    def adopt_loaded(self, loaded):
        """Adopt an opening result, preserving archive provenance and payload lifetime."""
        try:
            accepted = self.adopt_file(loaded.source_path, loaded.session,
                                       data_path=loaded.data_path, loaded=loaded)
        except Exception:
            loaded.close()
            raise
        if not accepted:loaded.close()
        return accepted

    def adopt_file(self, path, session, *, data_path=None, loaded=None):
        """Commit a successfully loaded file without interrupting a background capture."""
        if self.closed:
            return False
        if session is None:
            return self._fail("Capture opening was cancelled")
        scenes = tuple(session.scenes())
        if not scenes or not any(scene.frame_count for scene in scenes):
            return self._fail("This capture contains no displayable scene")
        self.follow_live = False
        self._preview_fallback = ()
        self._release_loaded()
        self._loaded = loaded
        self.display_path = absolute_path(path)
        self.display_data_path = absolute_path(data_path or path)
        self.display_session = session
        self.error = ""
        self._emit("display", session=session, path=self.display_path, data_path=self.display_data_path, scenes=scenes)
        try:
            self.settings.add_recent(self.display_path)
            self.refresh_recent_files()
            self._emit("recents")
            self.settings.set("open_directory", str(self.display_path.parent))
        except (OSError, ValueError) as error:
            self._fail(f"Capture opened, but recent history could not be saved: {error}")
        self._state(f"Opened {self.display_path.name}" + (f"; capturing on {self.port} in background" if self.live_session else ""))
        return True

    def open_recording(self, path):
        return self.open_file(path)

    def attach(self, endpoint=5425):
        if self.closed:
            return False
        try:
            port = parse_port(endpoint)
        except ValueError as error:
            return self._fail(error)
        if not self._persist("live_port", port):
            return False
        if self.live_session:
            self._finish_live(rearm=False)
        self.port = port
        self.accept_connections = True
        self.follow_live = True
        self._listening_requested = True
        self._retry_at = 0.
        self._arm()
        return self.live_session is not None

    def _arm(self):
        if (self.closed or not self.accept_connections or not self._listening_requested
                or self.live_session is not None or self._clock() < self._retry_at):
            return
        try:
            session = self._session_type.listen(self.port, "")
        except (RuntimeError, ValueError, OSError) as error:
            self._retry_at = self._clock() + 1.
            self._listener_error = f"Could not listen on port {self.port}: {error}; retrying"
            self._fail(self._listener_error)
            return
        self.live_session = session
        self.capture_selection.bind(session)
        self._connection_connected = False
        self._listener_error = ""
        self._auto_save_active = self.keep_recordings
        self._ever_connected = self._useful = False
        self._recorded_at = 0.
        self._live_update = None
        self._serial = self._epoch = self._transition_serial = 0
        self._state(f"Listening on port {self.port}; waiting for a producer")

    def _mark_connected(self, info):
        self._connection_connected = bool(info.connected and not info.ended)
        if info.connected and not self._ever_connected:
            self._ever_connected = True
            self._recorded_at = time.time()
            self._state(f"Receiving on port {self.port}" + ("; next connection disabled" if not self.accept_connections else ""))

    def _sample(self):
        update = self.live_session.refresh_live()
        if not update.ready:
            # A background engine may never have built a render snapshot. Its
            # old metadata publication alone cannot prove that a current complete
            # viewport is available; preserve the displayed capture until a safe
            # publication can provide both data and geometry.
            return False
        self._live_update = update
        self._useful = True
        promoted = self.follow_live and self.display_session is not self.live_session
        if promoted:
            self._preview_fallback = (self.display_session, self.display_path,
                                      self.display_data_path)
            self.display_session = self.live_session
            self.display_path = None
            self.display_data_path = self.display_path
            self._emit("display", session=self.live_session, path=self.display_path,
                       data_path=self.display_data_path, scenes=tuple(update.scenes))
        transitions = max(0, update.segment_transition_serial - self._transition_serial)
        changed = update.serial != self._serial or update.segment_epoch != self._epoch or transitions
        if self.display_session is self.live_session and (changed or promoted):
            self._emit("frames", session=self.live_session, path=self.display_path,
                       scenes=tuple(update.scenes), serial=update.serial,
                       segment_epoch=update.segment_epoch,
                       segment_changed=bool(transitions or (self._serial and self._epoch != update.segment_epoch)),
                       segment_transition_serial=update.segment_transition_serial,
                       segment_transition_count=transitions,
                       current_frame=update.current_frame, max_frame=update.max_frame)
        self._serial, self._epoch = update.serial, update.segment_epoch
        self._transition_serial = update.segment_transition_serial
        return True

    def _retain_session_recording(self, session, info):
        size = getattr(info, "recording_size", 0)
        if not self._ever_connected and not size:
            return None
        status = "Rejected" if info.error else "Complete" if self._useful else "Empty"
        recording = SessionRecording(self._next_session_recording_id, session,
            self._recorded_at or time.time(), status, size)
        self._next_session_recording_id += 1
        self._session_recordings.append(recording)
        self._emit("recordings")
        return recording

    def _finish_live(self, *, rearm, unexpected=False):
        session = self.live_session
        if session is None:
            return
        session.stop_live()  # Join before index publication or optional disk save.
        info = session.info()
        self._mark_connected(info)
        self._sample()
        recording = self._retain_session_recording(session, info)
        auto_save = recording is not None and recording.status == "Complete" and self._auto_save_active
        displayed = self.display_session is session
        if displayed:
            self.display_path = self.display_data_path = None
        connected, useful = self._ever_connected, self._useful
        self.live_session = None
        self.capture_selection.bind(None)
        self._connection_connected = False
        self._auto_save_active = False
        self._live_update = None
        self._ever_connected = self._useful = False
        if info.error and displayed and self._preview_fallback:
            self.display_session, self.display_path, self.display_data_path = self._preview_fallback
            self._preview_fallback = ()
            self._emit("display", session=self.display_session, path=self.display_path,
                       data_path=self.display_data_path,
                       scenes=tuple(self.display_session.scenes()) if self.display_session else ())
        elif displayed:
            self._preview_fallback = ()
            self._release_loaded()
        if info.error:
            self._listener_error = f"Last connection rejected: {info.error}"
            self._fail(self._listener_error)
        elif unexpected and not connected:
            self._listener_error = f"Listener on port {self.port} stopped before a valid stream; the port may be in use or the producer handshake was rejected"
            self._fail(self._listener_error)
        else:
            self._state("Capture complete" if useful else "Last connection contained no displayable scene" if connected else "Listener stopped")
        if rearm and self.accept_connections:
            self._retry_at = self._clock() + (1. if not connected else 0.)
            self._arm()
        # Rearm before the synchronous copy/fsync so the next producer can connect.
        # _arm resets per-connection state; use only the finished clip's locals here.
        if auto_save:
            listening_status = self.status if self.live_session is not None else None
            try:
                self.recordings.ensure_directory()
                destination = self.recordings.candidate_path(recording.recorded_at)
                session.save(str(destination), False)
                index = self._session_recording_index(recording.id)
                if index is not None:
                    self._session_recordings[index] = replace(recording, path=destination)
                self.recordings.upsert(destination, "Complete", recording.recorded_at)
                if displayed:
                    self.display_path = self.display_data_path = destination
            except (RuntimeError, ValueError, OSError) as error:
                self._fail(f"Automatic save failed: {error}")
                if listening_status is not None:
                    self._state(listening_status)
            self._emit("recordings")

    def suspend_live_polling(self):
        """Keep the displayed save target stable while native reception continues.

        Defer promotion, finalization and rearm together, as the original viewer
        does while its save dialog is open. The returned release is idempotent;
        the next normal poll adopts any useful completed background capture.
        """
        token = object()
        self._live_poll_holds.add(token)
        def release():
            self._live_poll_holds.discard(token)
        return release

    def poll(self):
        # Control metadata is independent of viewport/save-dialog publication.
        if not self.closed:
            try:
                self.capture_selection.poll()
            except (RuntimeError, ValueError, OSError) as error:
                self._fail(f"Recording selection update failed: {error}")
        if not self.closed and not self._live_poll_holds:
            if self.live_session is None:
                self._arm()
            else:
                try:
                    info = self.live_session.info()
                    self._mark_connected(info)
                    if info.ended:
                        self._finish_live(rearm=True, unexpected=True)
                    else:
                        self._sample()
                except (RuntimeError, ValueError, OSError) as error:
                    self._fail(f"Capture update failed: {error}")
        events = tuple(self._events)
        self._events.clear()
        return events

    def set_accept_connections(self, enabled):
        if self.closed:
            return
        self.accept_connections = bool(enabled)
        self._listening_requested = True
        if enabled:
            self._retry_at = 0.
            self._arm()
        elif self.live_session:
            info = self.live_session.info()
            self._connection_connected = bool(info.connected and not info.ended)
            if info.connected and not info.ended:
                self._state("Next connection disabled; current producer continues")
            else:
                self._finish_live(rearm=False)
        else:
            self._state("Accepting live connections is off")

    def set_keep_recordings(self, enabled):
        if not self._persist("keep_recordings", bool(enabled)):
            return False
        if self.live_session:
            info = self.live_session.info()
            if not self._ever_connected and not info.connected:
                self._auto_save_active = bool(enabled)
        self._state("New useful captures will be saved when they complete" if enabled else "New connections will not be written to disk")
        return True

    def follow_latest(self):
        if self.closed:
            return False
        self.follow_live = True
        if self.live_session and not self._sample():
            self._state("Waiting for the next complete live frame")
            return True
        self._state("Following the latest useful live capture")
        return True

    def _session_recording_index(self, recording_id):
        return next((index for index, recording in enumerate(self._session_recordings)
                     if recording.id == recording_id), None)

    def select_session_recording(self, recording_id):
        index = self._session_recording_index(recording_id)
        if index is None:
            return self._fail("The session recording is no longer available")
        recording = self._session_recordings[index]
        if recording.status != "Complete":
            return self._fail(f"This session recording is {recording.status.lower()} and cannot be displayed")
        self.follow_live = False
        self._preview_fallback = ()
        self._release_loaded()
        self.display_session = recording.session
        self.display_path = recording.path
        self.display_data_path = recording.path
        self._emit("display", session=recording.session, path=recording.path,
                   data_path=recording.path, scenes=tuple(recording.session.scenes()))
        self._state("Displaying retained session recording")
        return True

    def remove_session_recording(self, recording_id):
        index = self._session_recording_index(recording_id)
        if index is None:
            return False
        recording = self._session_recordings[index]
        if self._preview_fallback and self._preview_fallback[0] is recording.session:
            self._preview_fallback = (None, None, None)
        if self.display_session is recording.session:
            self.display_session = self.display_path = self.display_data_path = None
            self._emit("display")
        del self._session_recordings[index]
        self._emit("recordings")
        self._state("Session recording removed from memory; saved files are unchanged")
        return True

    def save_session_recording(self, recording_id, destination, *, overwrite=False):
        index = self._session_recording_index(recording_id)
        if index is None:
            return self._fail("The session recording is no longer available")
        recording = self._session_recordings[index]
        if recording.status != "Complete":
            return self._fail(f"This session recording is {recording.status.lower()} and cannot be saved")
        destination = absolute_path(destination)
        if not destination.suffix:
            destination = destination.with_suffix(".ovd")
        elif destination.suffix.lower() != ".ovd":
            return self._fail("Session recordings must be saved as .ovd")
        try:
            saved_size = recording.session.save(str(destination), overwrite)
            saved = destination
        except (RuntimeError, ValueError, OSError) as error:
            return self._fail(f"Could not save session recording: {error}")
        updated = replace(recording, path=saved,
                          size_bytes=recording.size_bytes or saved_size)
        self._session_recordings[index] = updated
        if self.display_session is recording.session:
            self.display_path = self.display_data_path = saved
        self._persist("save_directory", str(saved.parent))
        self._emit("recordings")
        self._state(f"Saved {saved}")
        return saved

    def detach(self):
        self.accept_connections = False
        self.follow_live = False
        self._finish_live(rearm=False)
        self._state("Detached; displayed capture is frozen and can be scrubbed")

    def new(self):
        if self.closed:
            return
        self.follow_live = False
        self._finish_live(rearm=False)
        self._release_loaded()
        self.display_session = None
        self.display_path = None
        self.display_data_path = None
        self.follow_live = True
        self._emit("display")
        self._state("New capture")
        self._listening_requested = True
        self._retry_at = 0.
        self._arm()

    def save_capture(self, destination, *, overwrite=False):
        """Save displayed capture bytes; an active displayed producer is frozen first.

        A later background listener is preserved. Completed memory recordings use
        save_session_recording; this path preserves existing file-copy behavior.
        """
        if not self.can_save:
            return self._fail("No completed capture is available to save")
        proof = self.verified_payload
        source = proof.path if proof is not None else self.display_path
        destination = absolute_path(destination)
        extension = ".ovd" if self.display_session.info().ovd else self.display_path.suffix.lower()
        if not destination.suffix:
            destination = destination.with_name(destination.name + extension)
        elif destination.suffix.lower() != extension:
            return self._fail(f"This capture must be saved as {extension}; changing its extension does not convert the native data")
        recording_id = self.displayed_session_recording_id
        if recording_id is not None:
            return self.save_session_recording(recording_id, destination, overwrite=overwrite)
        # Reject same-source and overwrite mistakes before changing capture state.
        if destination == source or destination == self.display_path:
            return self._fail("Choose a destination different from the recording source")
        try:
            validate_capture_destination(source, destination, overwrite=overwrite)
        except (OSError, ValueError) as error:
            return self._fail(f"Could not save capture: {error}")
        if self.display_session is self.live_session:
            requested = self.live_session
            self.detach()
            if self.display_session is not requested:
                return self._fail(self.error or "The requested live capture is no longer available to save")
            recording_id = self.displayed_session_recording_id
            if recording_id is None:
                return self._fail(self.error or "The requested live capture is no longer available to save")
            return self.save_session_recording(recording_id, destination, overwrite=overwrite)
        try:
            saved = copy_capture(source, destination, overwrite=overwrite)
            self._persist("save_directory", str(saved.parent))
            self._state(f"Saved {saved}")
            return saved
        except (OSError, ValueError) as error:
            return self._fail(f"Could not save capture: {error}")

    def delete_recording(self, path):
        try:
            self.recordings.delete(path, protected=(self.display_path,))
            self._emit("recordings")
            self._state(f"Deleted {Path(path).name}")
            return True
        except (OSError, ValueError) as error:
            return self._fail(f"Could not delete recording: {error}")

    def clear_recent(self):
        saved = self._persist("recent_files", [])
        if saved:
            self._emit("recents")
        return saved

    def shutdown(self):
        if self.closed:
            return
        self.accept_connections = False
        self.follow_live = False
        self._finish_live(rearm=False)
        self._release_loaded()
        self.display_session = None
        self.display_path = None
        self.display_data_path = None
        self._session_recordings.clear()
        self._preview_fallback = ()
        self.closed = True
        if self._temporary:
            self._temporary.cleanup()
        self._state("Closed")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()
