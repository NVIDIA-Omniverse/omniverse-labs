# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture preparation, owned native loading and local-file drop interpretation.

Archive packaging and bounded OVD snapshots are handled in Python; capture bytes go through
the native Session validator. A .pvd archive must contain one safe .pxd2 member,
matching the original Windows viewer. Its original path remains title/recent/
Save-As provenance while data_path supplies the private payload to analysis.

CaptureOpener.submit() returns OpenJob. Poll done()/progress on the UI thread,
cancel() to abandon work, then take_result() exactly once and pass its result to
CaptureWorkflow.adopt_loaded(). An ignored/cancelled result is never adopted.
close() cancels and joins outstanding workers. Native indexed loading and DOM
indexing report interruptible=False: cancellation is honored after that stage.

LoadedCapture.close() releases its ownership. Native Session references and
retain_capture(data_path) leases keep temporary payloads alive independently.
Analysis workers must retain such a lease for their entire read/stat sweep.
"""
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
from threading import Event, Lock
from urllib.parse import unquote, urlsplit
import weakref
import zipfile

from _pvd_native import LoadControl, Session

from .recordings import absolute_path


SUPPORTED_SUFFIXES = (".ovd", ".pxd2", ".pxd3", ".pvd")
MAX_VERIFIED_OVD_BYTES = 512 * 1024 * 1024
_leases = weakref.WeakValueDictionary()
_leases_lock = Lock()


@dataclass(frozen=True)
class OpenProgress:
    stage: str
    processed: int = 0
    total: int = 0
    interruptible: bool = True

    @property
    def fraction(self):
        return min(1., self.processed / self.total) if self.total else None


class _Resources:
    def __init__(self, directory=None):
        self.temporary = tempfile.TemporaryDirectory(prefix="pvd-open-", dir=directory)
        self.path = Path(self.temporary.name)


@contextmanager
def retain_capture(path):
    """Pin an extracted payload while a private native analysis session uses it."""
    with _leases_lock:
        lease = _leases.get(str(absolute_path(path)))
    try:
        yield lease
    finally:
        # Do not retain our own local through a preserved exception frame.
        lease = None


def submit_capture(executor, path, function, *args, **kwargs):
    """Pin a payload at submission, including time spent queued behind another job."""
    with _leases_lock:
        retained = [_leases.get(str(absolute_path(path)))]
    def work():
        try:
            return function(*args, **kwargs)
        finally:
            # Futures preserve failure tracebacks and the work function. Clear
            # the holder after native readers unwind, leaving those errors
            # intact without keeping an extracted payload alive indefinitely.
            retained.clear()
    try:
        future = executor.submit(work)
    except BaseException:
        retained.clear()
        raise
    # A cancelled queued Future never enters work(). Terminal callbacks cannot
    # run while its native work is active: running Future.cancel() returns False.
    future.add_done_callback(lambda completed: retained.clear())
    return future


@dataclass(frozen=True)
class VerifiedPayload:
    path: Path
    sha256: str
    byte_length: int
    file_identity: tuple


@dataclass
class LoadedCapture:
    source_path: Path
    data_path: Path
    session: object
    _resources: object
    verified_payload: VerifiedPayload | None = None
    evidence_reason: str = ""

    @property
    def archived(self):
        return self.source_path.suffix.lower() == ".pvd" and self.source_path != self.data_path

    def close(self):
        self.session = None
        self._resources = None


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def safe_archive_member(name):
    """Same lexical component policy as PvdInputValidation's Windows archive path."""
    if not isinstance(name, str) or not name or name.startswith(("/", "\\")) or ":" in name:
        return False
    for part in name.replace("\\", "/").split("/"):
        if (not part or part in (".", "..") or len(part.encode("utf-16-le", errors="surrogatepass")) // 2 > 255
                or part.endswith((".", " "))
                or any(ord(c) < 32 or c in '<>"|?*' for c in part)):
            return False
        device = part.split(".", 1)[0].rstrip(". ").upper()
        if device in ("CON", "PRN", "AUX", "NUL", "CLOCK$", "CONIN$", "CONOUT$"):
            return False
        if len(device) == 4 and device[:3] in ("COM", "LPT") and device[3] in "123456789¹²³":
            return False
    return True


def first_dropped_capture(payload):
    """First supported local path in a path list or text/uri-list; ignore extras.

    Network URLs are never downloaded. URI fragments/queries and remote file
    authorities are rejected. Plain absolute filesystem paths are also accepted
    for toolkits that supply paths directly instead of URI text.
    """
    items = payload.splitlines() if isinstance(payload, str) else payload
    for item in items or ():
        value = os.fspath(item).strip()
        if not value or value.startswith("#") or "\x00" in value:
            continue
        if value.lower().startswith("file:"):
            try:parsed = urlsplit(value)
            except ValueError:continue
            if parsed.netloc.lower() not in ("", "localhost") or parsed.query or parsed.fragment:
                continue
            value = unquote(parsed.path)
            if os.name == "nt" and len(value) > 2 and value[0] == "/" and value[2] == ":":
                value = value[1:]
        elif "://" in value:
            continue
        if "\x00" in value or not Path(value).is_absolute():
            continue
        path = absolute_path(value)
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            return path
    return None


class OpenJob:
    def __init__(self):
        self.future = None
        self.control = LoadControl()
        self._cancel = Event()
        self._native_started = False
        self._taken = False
        self._phase = OpenProgress("Preparing capture")
        self._lock = Lock()

    @property
    def cancelled(self):
        return self._cancel.is_set()

    @property
    def progress(self):
        if self._native_started and not self.done():
            value = self.control.progress()
            if value.stage:
                return OpenProgress(value.stage, value.processed, value.total, value.interruptible)
        with self._lock:
            return self._phase

    def done(self):
        return self.future is not None and self.future.done()

    def cancel(self):
        with self._lock:
            if self._taken:
                return False
            self._cancel.set()
            self.control.cancel()
        if self.future:
            self.future.cancel()
            if self.future.done() and not self.future.cancelled():
                try:self.future.result().close()
                except Exception:pass
        return True

    def take_result(self):
        result = self.future.result()
        with self._lock:
            if self.cancelled:
                result.close()
                raise CancelledError("Capture opening cancelled")
            if self._taken:
                raise RuntimeError("Capture result has already been adopted")
            self._taken = True
        return result

    def _check(self):
        if self.cancelled:
            raise CancelledError("Capture opening cancelled")

    def _update(self, stage, processed=0, total=0):
        with self._lock:
            self._phase = OpenProgress(stage, processed, total)

    def _extract(self, stream, resources):
        self._check()
        with zipfile.ZipFile(stream) as archive:
            payloads = [member for member in archive.infolist()
                        if member.filename.lower().endswith(".pxd2")
                        or member.orig_filename.lower().endswith(".pxd2")]
            if len(payloads) != 1:
                raise ValueError("PVD archive must contain exactly one PXD2 payload")
            member = payloads[0]
            mode = member.external_attr >> 16
            if (not safe_archive_member(member.orig_filename) or member.is_dir()
                    or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG))
                    or member.external_attr & 0x400):
                raise ValueError("PVD archive contains an unsafe PXD2 member")
            if member.flag_bits & 1:
                raise ValueError("Encrypted PVD archives are not supported")
            if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError("PVD archives must use stored or deflate compression")
            free = shutil.disk_usage(resources.path).free
            reserve = min(64 * 1024 * 1024, free // 20)
            if member.file_size > free - reserve:
                raise OSError("Not enough free space to extract this PVD capture")
            destination = resources.path / "capture.pxd2"
            self._update("Extracting PVD payload", 0, member.file_size)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            descriptor = os.open(destination, flags, 0o600)
            with os.fdopen(descriptor, "wb") as output, archive.open(member) as input_stream:
                processed = 0
                while True:
                    self._check()
                    block = input_stream.read(1024 * 1024)
                    if not block:
                        break
                    processed += len(block)
                    if processed > member.file_size:
                        raise ValueError("PVD payload exceeds its declared size")
                    output.write(block)
                    self._update("Extracting PVD payload", processed, member.file_size)
                if processed != member.file_size:
                    raise ValueError("PVD archive payload is incomplete")
                output.flush()
                os.fsync(output.fileno())
            return destination

    def _snapshot(self, stream, resources, size):
        free = shutil.disk_usage(resources.path).free
        if size > free - min(64 * 1024 * 1024, free // 20):
            raise OSError("Not enough free space for the OVD evidence snapshot")
        destination = resources.path / "capture.ovd"
        self._update("Snapshotting OVD payload", 0, size)
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        digest, processed = hashlib.sha256(), 0
        with os.fdopen(descriptor, "wb") as output:
            while True:
                self._check()
                block = stream.read(1024 * 1024)
                if not block:
                    break
                processed += len(block)
                if processed > size:
                    raise ValueError("Capture changed while copying its OVD payload")
                output.write(block);digest.update(block)
                self._update("Snapshotting OVD payload", processed, size)
            if processed != size:
                raise ValueError("Capture changed while copying its OVD payload")
            output.flush();os.fsync(output.fileno())
        return VerifiedPayload(destination, digest.hexdigest(), processed, _identity(destination.stat()))

    def _run(self, path, temporary_directory=None, session_type=Session):
        self._check()
        source = absolute_path(path)
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError("Open an OVD, PXD2, PXD3 or PVD capture")
        # Reject named pipes/devices before opening: a FIFO open can block forever.
        if not stat.S_ISREG(source.stat().st_mode):
            raise ValueError("Capture source must be a regular file")
        resources = _Resources(temporary_directory)
        session = None
        verified = None
        evidence_reason = "Evidence navigation requires a finalized OVD file"
        try:
            with source.open("rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError("Capture source must be a regular file")
                data_path = self._extract(stream, resources) if source.suffix.lower() == ".pvd" else source
                if source.suffix.lower() == ".ovd":
                    if before.st_size <= MAX_VERIFIED_OVD_BYTES:
                        verified = self._snapshot(stream, resources, before.st_size)
                        data_path, evidence_reason = verified.path, ""
                    else:
                        evidence_reason = "OVD snapshot limit is 512 MiB; evidence inspection remains available"
                self._check()
                if _identity(before) != _identity(os.fstat(stream.fileno())) or _identity(before) != _identity(source.stat()):
                    raise ValueError("Capture changed while it was opening; try a completed file")
                self._native_started = True
                session = session_type(str(data_path), str(resources.path), self.control)
                self._check()
                if _identity(before) != _identity(os.fstat(stream.fileno())) or _identity(before) != _identity(source.stat()):
                    raise ValueError("Capture changed while it was opening; try a completed file")
            # The callback's argument keeps resources alive until the last native
            # Session reference disappears, independently of UI result ownership.
            weakref.finalize(session, lambda retained: None, resources)
            with _leases_lock:
                _leases[str(data_path)] = resources
            self._update("Ready", 1, 1)
            return LoadedCapture(source, data_path, session, resources, verified, evidence_reason)
        except BaseException as error:
            session = None
            resources.temporary.cleanup()
            if self.cancelled:
                raise CancelledError("Capture opening cancelled") from None
            if isinstance(error, zipfile.BadZipFile):
                raise ValueError(f"Invalid PVD archive: {error}") from error
            raise


class CaptureOpener:
    def __init__(self, temporary_directory=None, *, session_type=Session):
        self.temporary_directory = temporary_directory
        self.session_type = session_type
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pvd-open")
        self.jobs = weakref.WeakSet()
        self.closed = False

    def submit(self, path):
        if self.closed:
            raise RuntimeError("Capture opener is closed")
        job = OpenJob()
        self.jobs.add(job)
        job.future = self.executor.submit(job._run, path, self.temporary_directory, self.session_type)
        weak_job = weakref.ref(job)
        def discard_cancelled(future):
            owner = weak_job()
            if owner is not None and owner.cancelled and not future.cancelled():
                try:future.result().close()
                except Exception:pass
        job.future.add_done_callback(discard_cancelled)
        return job

    def close(self):
        self.closed = True
        for job in tuple(self.jobs):
            job.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)


def load_capture(path, *, temporary_directory=None, session_type=Session):
    """Synchronous entry point for the non-GUI workflow and command-line callers."""
    return OpenJob()._run(path, temporary_directory, session_type)
